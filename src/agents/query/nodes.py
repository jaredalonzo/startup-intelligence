"""Query head nodes: parse_query (LLM tool-use) → retrieve (det) → answer (LLM).

Only two LLM touchpoints, per the graph contract: parse_query turns the fuzzy
question into a structured QueryPlan, and answer synthesizes a grounded,
cited response. Everything between — query embedding (mechanical, deterministic),
SQL construction, retrieval — is deterministic code. The head is read-only:
the retrieval connection is opened read-only at the DB level.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pgvector import Vector
from pgvector.psycopg import register_vector

from agents.query.sql import build_dossiers_query, build_postings_query
from agents.query.state import QueryPlan, QueryState, RetrievedDoc
from config import (
    EMBEDDING_LLM,
    EMBEDDING_QUERY_PREFIX,
    QUERY_ANSWER_LLM,
    QUERY_ENG_GROWTH_LOOKBACK_DAYS,
    QUERY_PARSE_LLM,
    QUERY_SNIPPET_CHARS,
    QUERY_TOP_K_DOSSIERS,
    QUERY_TOP_K_POSTINGS,
)
from llm_structured import bind_tools
from store.db import get_connection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# parse_query — LLM tool-use; the only place an LLM touches retrieval
# ---------------------------------------------------------------------------

_PARSE_SYSTEM = """\
You decompose a natural-language question about startups' job postings and
company dossiers into a QueryPlan tool call. Emit exactly one QueryPlan.

Rules:
- semantic_terms is required: the topical content to search for (roles,
  technologies, themes), stripped of anything expressed as a filter.
- Only set a filter the question explicitly implies. Never guess values.
- Company references become slugs: lowercase, no spaces (e.g. "Hugging Face"
  -> "huggingface").
- Seniority values must be one of: senior, staff, principal, junior, ic, manager.
- Time references become relative day windows (posted_after_days /
  first_seen_after_days), never absolute dates.
- Set growing_eng only when the question asks for companies whose engineering
  hiring is growing.
- corpus: 'postings' for job-level questions, 'dossiers' for company-trajectory
  questions, 'both' when unsure.
"""


async def parse_query(state: QueryState) -> dict[str, Any]:
    """Decompose the question into a QueryPlan via one tool call.

    Degrades, never gates: any failure (no tool call, invalid args, backend
    error) falls back to a semantic-only plan over the raw question, flagged
    with parse_fallback so the trace and --show-plan make it visible.
    """
    question = state["question"]
    try:
        bound = bind_tools(QUERY_PARSE_LLM, [QueryPlan])
        ai = await bound.ainvoke(
            [
                {"role": "system", "content": _PARSE_SYSTEM},
                {"role": "user", "content": question},
            ]
        )
        calls = getattr(ai, "tool_calls", None) or []
        if not calls:
            raise ValueError("parse model returned no tool call")
        plan = QueryPlan.model_validate(calls[0]["args"])
        if not plan.semantic_terms.strip():
            plan = plan.model_copy(update={"semantic_terms": question})
        logger.info("query: parsed plan %s", plan.model_dump(exclude_defaults=True))
        return {"plan": plan, "parse_fallback": False}
    except Exception:
        logger.warning(
            "query: parse_query failed; falling back to semantic-only retrieval",
            exc_info=True,
        )
        return {"plan": QueryPlan(semantic_terms=question), "parse_fallback": True}


# ---------------------------------------------------------------------------
# retrieve — deterministic hybrid retrieval; no LLM
# ---------------------------------------------------------------------------

def _run_retrieval(
    plan: QueryPlan, vec: list[float]
) -> tuple[list[RetrievedDoc], list[RetrievedDoc]]:
    """Execute the per-corpus hybrid queries on a read-only connection."""
    postings: list[RetrievedDoc] = []
    dossiers: list[RetrievedDoc] = []
    with get_connection() as conn:
        # Read-only must be set before the first query (register_vector runs a
        # catalog lookup); it makes the never-writes guardrail a DB-level fact.
        conn.read_only = True
        register_vector(conn)
        qvec = Vector(vec)
        if plan.corpus in ("postings", "both"):
            sql, params = build_postings_query(
                plan,
                top_k=QUERY_TOP_K_POSTINGS,
                growth_days=QUERY_ENG_GROWTH_LOOKBACK_DAYS,
                snippet_chars=QUERY_SNIPPET_CHARS,
            )
            params["qvec"] = qvec
            rows = conn.execute(sql, params).fetchall()
            postings = [RetrievedDoc.model_validate(dict(r)) for r in rows]
        if plan.corpus in ("dossiers", "both"):
            sql, params = build_dossiers_query(
                plan,
                top_k=QUERY_TOP_K_DOSSIERS,
                growth_days=QUERY_ENG_GROWTH_LOOKBACK_DAYS,
                snippet_chars=QUERY_SNIPPET_CHARS,
            )
            params["qvec"] = qvec
            rows = conn.execute(sql, params).fetchall()
            dossiers = [RetrievedDoc.model_validate(dict(r)) for r in rows]
    return postings, dossiers


async def retrieve(state: QueryState) -> dict[str, Any]:
    """Hybrid retrieval: embed the question (mechanical), then one SQL query
    per corpus combining cosine order with the plan's structured filters."""
    plan = state["plan"] or QueryPlan(semantic_terms=state["question"])
    semantic = plan.semantic_terms.strip() or state["question"]
    qtext = f"{EMBEDDING_QUERY_PREFIX}{semantic}"
    vec = await asyncio.to_thread(EMBEDDING_LLM.embed_query, qtext)
    postings, dossiers = await asyncio.to_thread(_run_retrieval, plan, vec)
    logger.info(
        "query: retrieved %d posting(s), %d dossier(s)%s",
        len(postings),
        len(dossiers),
        f" (best distance {postings[0].distance:.3f})" if postings else "",
    )
    return {"postings": postings, "dossiers": dossiers}


# ---------------------------------------------------------------------------
# answer — LLM grounded synthesis; cite-or-abstain
# ---------------------------------------------------------------------------

ABSTAIN_MARKDOWN = """\
No documents in the corpus matched this question, so I can't answer it without
inventing something.

Try rephrasing with different terms, loosening any company/seniority/location
constraints, or asking about roles and companies the watchlist actually tracks.
"""

_ANSWER_SYSTEM = """\
You are an analyst answering a question using ONLY the numbered context
documents provided (job postings and company dossiers from a tracked corpus).

The context may open with a "Retrieval filters applied" note. Those constraints
were enforced by the deterministic retrieval layer (SQL over typed columns,
extracted skills, and the snapshot time series) — every listed document
satisfies them even where its text excerpt doesn't repeat the words. You may
assert those constraints as facts about the listed documents, attributing them
to the filters (e.g. "per the applied filters").

Hard rules — cite or abstain:
- Every other factual claim must cite its supporting document(s) as [n].
- A claim you cannot support with a provided document or an applied filter
  must not be made.
- If the context does not support an answer to the question, say exactly that
  instead of answering around it.
- Never assert a trend ("rising", "growing", "increasingly") unless a provided
  field states it (a dossier classification, or an applied growing-engineering
  filter). Individual documents are evidence of openings, not of trends.
- Absence from the context is not evidence of absence: these are the top
  retrieved matches from a fixed watchlist, not an exhaustive market scan.
- End with a "## Sources" section listing every cited [n] as:
  [n] title — company — URL (date).

Answer in concise markdown. Lead with the direct answer, then supporting detail.
"""


def describe_filters(plan: QueryPlan | None) -> str:
    """Deterministically render the plan's enforced filters for the answer prompt.

    These statements are grounded in the retrieval layer itself (every returned
    row satisfied them in SQL), so the answer model may treat them as facts
    about the listed documents. Returns "" when no filters were applied.
    """
    if plan is None:
        return ""
    lines: list[str] = []
    if plan.companies:
        lines.append(f"restricted to companies: {', '.join(plan.companies)}")
    if plan.departments:
        lines.append(f"department matches: {', '.join(plan.departments)}")
    if plan.teams:
        lines.append(f"team matches: {', '.join(plan.teams)}")
    if plan.seniorities:
        lines.append(f"seniority in: {', '.join(plan.seniorities)}")
    if plan.locations:
        lines.append(f"location matches: {', '.join(plan.locations)}")
    if plan.remote is not None:
        lines.append("remote roles only" if plan.remote else "non-remote roles only")
    if plan.posted_after_days is not None:
        lines.append(f"posted within the last {plan.posted_after_days} days")
    if plan.first_seen_after_days is not None:
        lines.append(f"first seen in the corpus within the last {plan.first_seen_after_days} days")
    if plan.comp_min is not None:
        lines.append(f"compensation range reaching at least {plan.comp_min}")
    if plan.skills:
        lines.append(
            f"every posting's extracted skills include: {', '.join(plan.skills)}"
        )
    if plan.platforms:
        lines.append(
            f"every posting's extracted platforms include: {', '.join(plan.platforms)}"
        )
    if plan.growing_eng:
        lines.append(
            "every company shown has growing engineering headcount over the "
            "tracked snapshot window (deterministic time-series delta)"
        )
    if not lines:
        return ""
    return "Retrieval filters applied (enforced in SQL; true of every document below):\n" + "\n".join(
        f"- {line}" for line in lines
    )


def format_context(
    postings: list[RetrievedDoc], dossiers: list[RetrievedDoc]
) -> str:
    """Deterministically render retrieved docs as numbered context sections.

    Numbering runs continuously across both sections so [n] citations are
    unambiguous. Each entry carries the citation fields the answer must echo:
    URL and the relevant dates (postings: posted/first seen/last confirmed
    live; dossiers: generated).
    """

    def _date(value: Any) -> str:
        return value.isoformat()[:10] if value is not None else "unknown"

    lines: list[str] = []
    n = 0
    if postings:
        lines.append("### Job postings")
        for doc in postings:
            n += 1
            lines.append(f"[{n}] {doc.company_name} — {doc.title}")
            lines.append(f"URL: {doc.url or 'none'}")
            lines.append(
                f"Posted: {_date(doc.posted_at)} | First seen: {_date(doc.first_seen_at)}"
                f" | Last confirmed live: {_date(doc.last_seen_at)}"
            )
            lines.append(doc.snippet.strip())
            lines.append("")
    if dossiers:
        lines.append("### Company dossiers")
        for doc in dossiers:
            n += 1
            lines.append(f"[{n}] {doc.title}")
            lines.append(f"URL: {doc.url or 'none'}")
            lines.append(f"Generated: {_date(doc.generated_at)}")
            lines.append(doc.snippet.strip())
            lines.append("")
    return "\n".join(lines).strip()


async def answer(state: QueryState) -> dict[str, Any]:
    """Grounded synthesis over the retrieved context; cite-or-abstain.

    Empty retrieval short-circuits to a deterministic abstain — no LLM call,
    nothing to hallucinate from.
    """
    postings = state["postings"] or []
    dossiers = state["dossiers"] or []
    if not postings and not dossiers:
        logger.info("query: empty retrieval — abstaining without an LLM call")
        return {"answer_markdown": ABSTAIN_MARKDOWN}

    context = format_context(postings, dossiers)
    filters_note = describe_filters(state["plan"])
    if filters_note:
        context = f"{filters_note}\n\n{context}"
    response = await QUERY_ANSWER_LLM.ainvoke(
        [
            SystemMessage(content=_ANSWER_SYSTEM),
            HumanMessage(
                content=f"Question: {state['question']}\n\nContext documents:\n\n{context}"
            ),
        ]
    )
    return {"answer_markdown": str(response.content)}
