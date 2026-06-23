"""Tracker agent node implementations.

resolve_board is the tracker's one genuinely agentic node: for a company whose
ATS board is not yet cached, it discovers which ATS (Greenhouse / Lever / Ashby /
Workable) and slug the company's public job board lives on, then caches the
result so it's a one-time cost.

Design (per the repo's "do not agentify deterministic work" principle):
  - The LLM only *proposes* candidate slugs — the genuinely fuzzy part, e.g.
    company "Cognition" → board slug "cognitionlabs", "Glean" → "gleanwork".
  - Verification stays deterministic: each candidate is probed against the real
    ATS endpoints via ingestion.watchlist.probe_ats. The first verified hit is
    the answer; the LLM's own narrative conclusion is never trusted as truth.
  - A cheap deterministic fast-path (watchlist hint + name-derived slugs) runs
    first; the LLM tool-use loop is entered only when that misses, keeping cost
    near zero for the common case.
"""
from __future__ import annotations

import logging
from typing import Awaitable, Callable, Literal

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from agents.tracker.state import BoardResolution, TrackerState
from config import RESOLVE_LLM, TRACKER_RESOLVE_MAX_PROBES
from ingestion.ats.models import ATSSource
from ingestion.watchlist import _BOARD_URL, get_cached, probe_ats, upsert_company
from store.db import get_connection

logger = logging.getLogger(__name__)

# A probe maps a candidate slug to (ats, ats_slug) if a live board exists, else None.
ProbeFn = Callable[[str], Awaitable[tuple[ATSSource, str] | None]]


class ProbeCandidate(BaseModel):
    """Test whether a slug hosts a live public job board on any supported ATS.

    Call this with your best guess of the company's ATS board slug. If a live
    board is found the search is over; otherwise you'll be told it missed so you
    can reason about a better slug and try again. Slugs are lowercase with no
    spaces or punctuation, e.g. 'scaleai', 'cognitionlabs', 'gleanwork'.
    """

    slug: str = Field(description="Candidate ATS board slug to verify, e.g. 'cognitionlabs'")


_SYSTEM_PROMPT = (
    "You identify which applicant-tracking system (ATS) hosts a company's public "
    "job board and under what slug. Supported ATSes: Greenhouse, Lever, Ashby, "
    "Workable. You cannot browse the web — instead, propose a candidate slug and "
    "call ProbeCandidate to test it against the real ATS endpoints. Iterate: when "
    "a probe misses, reason about a better slug (drop or add suffixes like 'labs', "
    "'hq', 'ai', 'inc'; try the website domain root or the product name; collapse "
    "spaces) and probe again. Slugs are lowercase with no spaces or punctuation. "
    "Stop the moment a probe succeeds. If several probes fail, stop — do not guess "
    "endlessly."
)


def _seed_candidates(company: dict) -> list[str]:
    """Deterministic, zero-cost slug guesses to try before invoking the LLM.

    Order matters: a trusted watchlist hint first, then the internal key, then a
    name-derived alphanumeric slug.
    """
    name = company.get("name", "") or ""
    name_slug = "".join(ch for ch in name.lower() if ch.isalnum())
    candidates: list[str] = []
    for cand in (company.get("ats_slug"), company.get("slug"), name_slug):
        if cand and cand not in candidates:
            candidates.append(cand)
    return candidates


def _describe_company(company: dict, *, already_tried: list[str]) -> str:
    lines = [
        f"Company name: {company.get('name')}",
        f"Internal key: {company.get('slug')}",
    ]
    if company.get("domain"):
        lines.append(f"Website domain: {company['domain']}")
    if company.get("github_org"):
        lines.append(f"GitHub org: {company['github_org']}")
    if already_tried:
        lines.append(f"Already tried (all missed): {', '.join(already_tried)}")
    lines.append("Find this company's live ATS job board.")
    return "\n".join(lines)


def _resolved(
    slug: str, ats: ATSSource, ats_slug: str, method: str, attempted: list[str]
) -> BoardResolution:
    return BoardResolution(
        company_slug=slug,
        resolved=True,
        ats=ats,
        ats_slug=ats_slug,
        board_url=_BOARD_URL[ats].format(slug=ats_slug),
        method=method,
        attempted_slugs=list(attempted),
    )


async def resolve_one(
    company: dict,
    *,
    probe: ProbeFn,
    llm,  # a tool-calling chat model (ChatOllama / ChatAnthropic); injected for testability
    max_probes: int = TRACKER_RESOLVE_MAX_PROBES,
) -> BoardResolution:
    """Resolve a single company's ATS board. Pure of cache/DB and HTTP setup.

    Dependencies (`probe`, `llm`) are injected so this is unit-testable without a
    network or a live model. Returns a BoardResolution; never writes to the DB.
    """
    slug = company["slug"]
    attempted: list[str] = []

    async def _probe(cand: str) -> tuple[ATSSource, str] | None:
        cand = cand.strip().lower()
        if not cand or cand in attempted:
            return None
        attempted.append(cand)
        return await probe(cand)

    # 1. Deterministic fast-path — trusted hint and obvious name-derived slugs.
    for cand in _seed_candidates(company):
        hit = await _probe(cand)
        if hit:
            method = "hint" if cand == company.get("ats_slug") else "deterministic"
            return _resolved(slug, hit[0], hit[1], method, attempted)

    # 2. Agentic escalation — the LLM proposes slugs; the probe tool verifies them.
    llm_with_tools = llm.bind_tools([ProbeCandidate])
    messages: list = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_describe_company(company, already_tried=attempted)),
    ]
    for _ in range(max_probes):
        ai: AIMessage = await llm_with_tools.ainvoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            break
        for call in ai.tool_calls:
            cand = str(call["args"].get("slug", ""))
            hit = await _probe(cand)
            if hit:
                # A verified board is ground truth — return without trusting the LLM further.
                return _resolved(slug, hit[0], hit[1], "agentic", attempted)
            messages.append(
                ToolMessage(
                    content=f"No live board found for slug '{cand.strip().lower()}'. "
                    "Try a different guess.",
                    tool_call_id=call["id"],
                )
            )

    logger.warning("resolve_board: no ATS board found for %s (tried: %s)", slug, attempted)
    return BoardResolution(
        company_slug=slug, resolved=False, method="agentic", attempted_slugs=attempted
    )


async def resolve_board(state: TrackerState) -> dict:
    """Resolve and cache the ATS board for the company in *state*.

    Conditional skip: if the board is already cached in the companies table, return
    immediately — no probing, no LLM call (the one-time-cost guarantee). Otherwise
    run resolve_one and, on success, cache the result so future runs hit the cache.
    """
    company = state["company"]
    slug = company["slug"]

    with get_connection() as conn:
        cached = get_cached(slug, conn)
    if cached:
        ats: ATSSource = cached["ats"]
        ats_slug: str = cached["ats_slug"]
        return {"resolution": _resolved(slug, ats, ats_slug, "cache", [])}

    async with httpx.AsyncClient() as client:
        resolution = await resolve_one(
            company,
            probe=lambda cand: probe_ats(cand, client),
            llm=RESOLVE_LLM,
        )

    if resolution.resolved:
        assert resolution.ats is not None and resolution.ats_slug is not None
        with get_connection() as conn:
            upsert_company(
                slug,
                company["name"],
                resolution.ats,  # type: ignore[arg-type]
                resolution.ats_slug,
                conn,
                github_org=company.get("github_org"),
                blog_url=company.get("blog_url"),
                blog_rss_url=company.get("blog_rss_url"),
            )
            conn.commit()
        logger.info("resolve_board: %s -> %s/%s (%s)", slug, resolution.ats,
                    resolution.ats_slug, resolution.method)

    return {"resolution": resolution}


def route_after_resolve(state: TrackerState) -> Literal["fetch_signals", "__end__"]:
    """Skip the rest of the per-company pipeline when the board never resolved."""
    resolution = state.get("resolution")
    if resolution is not None and resolution.resolved:
        return "fetch_signals"
    return "__end__"
