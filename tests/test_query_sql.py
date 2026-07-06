"""Unit tests for the query head's whitelist SQL builders.

The builders are the injection boundary between LLM output (the QueryPlan) and
the database: these tests pin that plan values only ever land in the params
dict, never in the SQL text, and that each filter/CTE appears iff its plan
field is set.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.query.sql import build_dossiers_query, build_postings_query  # noqa: E402
from agents.query.state import QueryPlan  # noqa: E402

_KW = {"top_k": 12, "growth_days": 30, "snippet_chars": 1200}

HOSTILE = "a'; DROP TABLE postings; --"


def _placeholders(sql: str) -> set[str]:
    return set(re.findall(r"%\((\w+)\)s", sql))


# ---------------------------------------------------------------------------
# postings builder
# ---------------------------------------------------------------------------

def test_semantic_only_plan_is_bare():
    sql, params = build_postings_query(QueryPlan(semantic_terms="rust"), **_KW)
    assert "WITH" not in sql                      # no CTEs
    assert "WHERE p.embedding IS NOT NULL\n" in sql  # no extra filters
    assert "ORDER BY distance" in sql and "LIMIT %(top_k)s" in sql
    assert params == {"top_k": 12, "snippet_chars": 1200}
    # qvec is the caller's to bind
    assert "qvec" in _placeholders(sql)


def test_each_filter_lands_in_params_not_sql():
    cases: list[tuple[QueryPlan, str, str, object]] = [
        (QueryPlan(semantic_terms="x", companies=["Acme"]),
         "p.company_slug = ANY(%(companies)s)", "companies", ["acme"]),
        (QueryPlan(semantic_terms="x", departments=["Eng"]),
         "p.department ILIKE ANY(%(departments)s)", "departments", ["%Eng%"]),
        (QueryPlan(semantic_terms="x", teams=["Infra"]),
         "p.team ILIKE ANY(%(teams)s)", "teams", ["%Infra%"]),
        (QueryPlan(semantic_terms="x", seniorities=["Senior"]),
         "p.seniority = ANY(%(seniorities)s)", "seniorities", ["senior"]),
        (QueryPlan(semantic_terms="x", locations=["London"]),
         "p.location ILIKE ANY(%(locations)s)", "locations", ["%London%"]),
        (QueryPlan(semantic_terms="x", remote=True),
         "p.remote = %(remote)s", "remote", True),
        (QueryPlan(semantic_terms="x", posted_after_days=14),
         "p.posted_at >= NOW() - make_interval(days => %(posted_after_days)s)",
         "posted_after_days", 14),
        (QueryPlan(semantic_terms="x", first_seen_after_days=7),
         "p.first_seen_at >= NOW() - make_interval(days => %(first_seen_after_days)s)",
         "first_seen_after_days", 7),
        (QueryPlan(semantic_terms="x", comp_min=150000),
         "COALESCE(p.compensation_max, p.compensation_min) >= %(comp_min)s",
         "comp_min", 150000),
        (QueryPlan(semantic_terms="x", skills=["Kubernetes"]),
         "unnest(e.skills) sk WHERE lower(sk) = ANY(%(skills)s)", "skills", ["kubernetes"]),
        (QueryPlan(semantic_terms="x", platforms=["AWS"]),
         "unnest(e.platforms) pf WHERE lower(pf) = ANY(%(platforms)s)", "platforms", ["aws"]),
    ]
    for plan, fragment, key, expected in cases:
        sql, params = build_postings_query(plan, **_KW)
        assert fragment in sql, fragment
        assert params[key] == expected


def test_unset_filters_leave_no_trace():
    sql, _ = build_postings_query(QueryPlan(semantic_terms="x", remote=True), **_KW)
    assert "seniority" not in sql and "ILIKE" not in sql and "comp_min" not in sql


def test_extractions_cte_only_with_skills_or_platforms():
    bare_sql, _ = build_postings_query(QueryPlan(semantic_terms="x"), **_KW)
    assert "latest_extractions" not in bare_sql
    for plan in (QueryPlan(semantic_terms="x", skills=["rust"]),
                 QueryPlan(semantic_terms="x", platforms=["gcp"])):
        sql, _ = build_postings_query(plan, **_KW)
        assert "latest_extractions AS (" in sql
        assert "JOIN latest_extractions e ON e.ats = p.ats AND e.posting_id = p.id" in sql
        assert "DISTINCT ON (ats, posting_id)" in sql


def test_eng_growth_cte_only_when_flagged():
    bare_sql, bare_params = build_postings_query(QueryPlan(semantic_terms="x"), **_KW)
    assert "eng_growth" not in bare_sql and "growth_days" not in bare_params

    sql, params = build_postings_query(
        QueryPlan(semantic_terms="x", growing_eng=True), **_KW
    )
    assert "eng_growth AS (" in sql
    assert "JOIN eng_growth g ON g.company_slug = p.company_slug" in sql
    # latest-vs-oldest eng_count within the window — a snapshots delta
    assert "array_agg(eng_count ORDER BY snapshot_at DESC))[1]" in sql
    assert params["growth_days"] == 30


def test_combined_plan_placeholders_all_bound():
    plan = QueryPlan(
        semantic_terms="x", companies=["acme"], departments=["eng"], teams=["infra"],
        seniorities=["senior"], locations=["nyc"], remote=False, posted_after_days=30,
        first_seen_after_days=10, comp_min=1, skills=["rust"], platforms=["aws"],
        growing_eng=True,
    )
    sql, params = build_postings_query(plan, **_KW)
    # every placeholder in the SQL is bound (qvec is the caller's) — no dangling params
    assert _placeholders(sql) - {"qvec"} == set(params)


def test_injection_lands_only_in_params():
    hostile_plan = QueryPlan(
        semantic_terms=HOSTILE, companies=[HOSTILE], departments=[HOSTILE],
        teams=[HOSTILE], seniorities=[HOSTILE], locations=[HOSTILE],
        skills=[HOSTILE], platforms=[HOSTILE],
    )
    clean_plan = QueryPlan(
        semantic_terms="x", companies=["a"], departments=["a"], teams=["a"],
        seniorities=["a"], locations=["a"], skills=["a"], platforms=["a"],
    )
    hostile_sql, hostile_params = build_postings_query(hostile_plan, **_KW)
    clean_sql, _ = build_postings_query(clean_plan, **_KW)

    assert "DROP TABLE" not in hostile_sql
    # the SQL text is exactly the clean plan's — values changed nothing structural
    assert hostile_sql == clean_sql
    # the payload survives only as parameter values
    assert any(HOSTILE in str(v) or HOSTILE.lower() in str(v)
               for v in hostile_params.values())


# ---------------------------------------------------------------------------
# dossiers builder
# ---------------------------------------------------------------------------

def test_dossiers_latest_per_company_and_company_filter():
    sql, params = build_dossiers_query(
        QueryPlan(semantic_terms="x", companies=["Acme"]), **_KW
    )
    assert "DISTINCT ON (company_slug)" in sql
    assert "ORDER BY company_slug, generated_at DESC" in sql
    assert "d.company_slug = ANY(%(companies)s)" in sql
    assert params["companies"] == ["acme"]
    assert "notion_url AS url" in sql


def test_dossiers_ignore_posting_only_filters():
    plan = QueryPlan(
        semantic_terms="x", seniorities=["senior"], skills=["rust"], remote=True,
        posted_after_days=5, comp_min=1, locations=["nyc"],
    )
    sql, params = build_dossiers_query(plan, **_KW)
    assert "seniority" not in sql and "skills" not in sql and "remote" not in sql
    assert "posted_at" not in sql and "comp_min" not in sql and "location" not in sql
    assert set(params) == {"top_k", "snippet_chars"}


def test_dossiers_eng_growth():
    sql, params = build_dossiers_query(
        QueryPlan(semantic_terms="x", growing_eng=True), **_KW
    )
    assert "eng_growth AS (" in sql
    assert "JOIN eng_growth g ON g.company_slug = d.company_slug" in sql
    assert params["growth_days"] == 30


def test_dossiers_injection_lands_only_in_params():
    sql_hostile, params = build_dossiers_query(
        QueryPlan(semantic_terms="x", companies=[HOSTILE]), **_KW
    )
    sql_clean, _ = build_dossiers_query(
        QueryPlan(semantic_terms="x", companies=["a"]), **_KW
    )
    assert "DROP TABLE" not in sql_hostile
    assert sql_hostile == sql_clean
    assert params["companies"] == [HOSTILE.lower()]
