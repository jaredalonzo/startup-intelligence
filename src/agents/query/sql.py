"""Hybrid-retrieval SQL builders for the query head. Pure functions, no DB, no LLM.

Each builder turns a QueryPlan into one parameterized query: pgvector cosine
ordering (`embedding <=> %(qvec)s`) combined with typed-column filters and, for
the growing-eng filter, a snapshots-delta CTE — hybrid retrieval in a single
round trip, per the graph contract.

Injection safety: SQL text is assembled ONLY from the static fragments below.
Plan values (LLM output) never enter the SQL string — they are bound through
named psycopg placeholders, so a hostile value can only ever be a parameter.
The caller supplies the query vector as ``params["qvec"]`` (a pgvector Vector).
"""
from __future__ import annotations

from typing import Any, Callable

from agents.query.state import QueryPlan

# One whitelist entry: (static SQL fragment, params contributed when active).
# A contributor returns None when the plan doesn't set its filter.
_Contributor = Callable[[QueryPlan], dict[str, Any] | None]
_Filter = tuple[str, _Contributor]


def _ilike_patterns(values: list[str]) -> list[str]:
    return [f"%{v}%" for v in values]


_POSTING_FILTERS: list[_Filter] = [
    (
        "p.company_slug = ANY(%(companies)s)",
        lambda p: {"companies": [c.lower() for c in p.companies]} if p.companies else None,
    ),
    (
        "p.department ILIKE ANY(%(departments)s)",
        lambda p: {"departments": _ilike_patterns(p.departments)} if p.departments else None,
    ),
    (
        "p.team ILIKE ANY(%(teams)s)",
        lambda p: {"teams": _ilike_patterns(p.teams)} if p.teams else None,
    ),
    (
        "p.seniority = ANY(%(seniorities)s)",
        lambda p: {"seniorities": [s.lower() for s in p.seniorities]} if p.seniorities else None,
    ),
    (
        "p.location ILIKE ANY(%(locations)s)",
        lambda p: {"locations": _ilike_patterns(p.locations)} if p.locations else None,
    ),
    (
        "p.remote = %(remote)s",
        lambda p: {"remote": p.remote} if p.remote is not None else None,
    ),
    (
        "p.posted_at >= NOW() - make_interval(days => %(posted_after_days)s)",
        lambda p: (
            {"posted_after_days": p.posted_after_days}
            if p.posted_after_days is not None
            else None
        ),
    ),
    (
        "p.first_seen_at >= NOW() - make_interval(days => %(first_seen_after_days)s)",
        lambda p: (
            {"first_seen_after_days": p.first_seen_after_days}
            if p.first_seen_after_days is not None
            else None
        ),
    ),
    (
        # The posting's range must reach comp_min: prefer the max bound, fall
        # back to the min bound when that's all the source published.
        "COALESCE(p.compensation_max, p.compensation_min) >= %(comp_min)s",
        lambda p: {"comp_min": p.comp_min} if p.comp_min is not None else None,
    ),
    (
        "EXISTS (SELECT 1 FROM unnest(e.skills) sk WHERE lower(sk) = ANY(%(skills)s))",
        lambda p: {"skills": [s.lower() for s in p.skills]} if p.skills else None,
    ),
    (
        "EXISTS (SELECT 1 FROM unnest(e.platforms) pf WHERE lower(pf) = ANY(%(platforms)s))",
        lambda p: {"platforms": [x.lower() for x in p.platforms]} if p.platforms else None,
    ),
]

# CTE over the append-only snapshot series: companies whose latest eng_count in
# the lookback window exceeds their oldest — the same latest-vs-oldest window
# delta the tracker trends on. A snapshots filter, never a semantic match.
_ENG_GROWTH_CTE = """\
eng_growth AS (
    SELECT company_slug
    FROM snapshots
    WHERE snapshot_at >= NOW() - make_interval(days => %(growth_days)s)
      AND eng_count IS NOT NULL
    GROUP BY company_slug
    HAVING (array_agg(eng_count ORDER BY snapshot_at DESC))[1]
         > (array_agg(eng_count ORDER BY snapshot_at ASC))[1]
)"""

# Latest extraction per posting (the aggregation layer's read discipline).
_LATEST_EXTRACTIONS_CTE = """\
latest_extractions AS (
    SELECT DISTINCT ON (ats, posting_id) ats, posting_id, skills, platforms
    FROM extractions
    ORDER BY ats, posting_id, extracted_at DESC
)"""


def build_postings_query(
    plan: QueryPlan, *, top_k: int, growth_days: int, snippet_chars: int
) -> tuple[str, dict[str, Any]]:
    """One hybrid query over postings: cosine order + whitelisted filters.

    Returns (sql, params) with named placeholders; the caller must add
    ``params["qvec"]``. Only fragments whose plan field is set are included.
    """
    params: dict[str, Any] = {
        "top_k": top_k,
        "snippet_chars": snippet_chars,
    }
    where = ["p.embedding IS NOT NULL"]
    for fragment, contribute in _POSTING_FILTERS:
        contribution = contribute(plan)
        if contribution is not None:
            where.append(fragment)
            params.update(contribution)

    ctes: list[str] = []
    joins = ["JOIN companies c ON c.slug = p.company_slug"]
    if plan.skills or plan.platforms:
        ctes.append(_LATEST_EXTRACTIONS_CTE)
        joins.append("JOIN latest_extractions e ON e.ats = p.ats AND e.posting_id = p.id")
    if plan.growing_eng:
        ctes.append(_ENG_GROWTH_CTE)
        joins.append("JOIN eng_growth g ON g.company_slug = p.company_slug")
        params["growth_days"] = growth_days

    with_clause = f"WITH {', '.join(ctes)}\n" if ctes else ""
    sql = (
        f"{with_clause}"
        "SELECT 'posting' AS kind, p.company_slug, c.name AS company_name,\n"
        "       p.title, p.url,\n"
        "       LEFT(p.description_text, %(snippet_chars)s) AS snippet,\n"
        "       p.posted_at, p.first_seen_at, p.last_seen_at,\n"
        "       p.embedding <=> %(qvec)s AS distance\n"
        "FROM postings p\n"
        f"{chr(10).join(joins)}\n"
        f"WHERE {' AND '.join(where)}\n"
        "ORDER BY distance\n"
        "LIMIT %(top_k)s"
    )
    return sql, params


def build_dossiers_query(
    plan: QueryPlan, *, top_k: int, growth_days: int, snippet_chars: int
) -> tuple[str, dict[str, Any]]:
    """One hybrid query over dossiers (latest per company; the table is append-only).

    Dossiers are company-level, so only the company and growing-eng filters
    apply; posting-only filters are deliberately ignored here.
    """
    params: dict[str, Any] = {
        "top_k": top_k,
        "snippet_chars": snippet_chars,
    }
    where = ["d.embedding IS NOT NULL"]
    if plan.companies:
        where.append("d.company_slug = ANY(%(companies)s)")
        params["companies"] = [c.lower() for c in plan.companies]

    ctes = [
        # Latest dossier per company — never surface a stale one.
        "latest_dossiers AS (\n"
        "    SELECT DISTINCT ON (company_slug) *\n"
        "    FROM dossiers\n"
        "    ORDER BY company_slug, generated_at DESC\n"
        ")"
    ]
    joins = ["JOIN companies c ON c.slug = d.company_slug"]
    if plan.growing_eng:
        ctes.append(_ENG_GROWTH_CTE)
        joins.append("JOIN eng_growth g ON g.company_slug = d.company_slug")
        params["growth_days"] = growth_days

    sql = (
        f"WITH {', '.join(ctes)}\n"
        "SELECT 'dossier' AS kind, d.company_slug, c.name AS company_name,\n"
        "       'Dossier — ' || c.name || ' (' || COALESCE(d.classification, 'unscored') || ')' AS title,\n"
        "       d.notion_url AS url,\n"
        "       LEFT(d.dossier_markdown, %(snippet_chars)s) AS snippet,\n"
        "       d.generated_at,\n"
        "       d.embedding <=> %(qvec)s AS distance\n"
        "FROM latest_dossiers d\n"
        f"{chr(10).join(joins)}\n"
        f"WHERE {' AND '.join(where)}\n"
        "ORDER BY distance\n"
        "LIMIT %(top_k)s"
    )
    return sql, params
