"""Shared TypedDict state objects for both LangGraph agents."""
from __future__ import annotations

import operator
from typing import Annotated

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Skills extraction schema (output of the extract_skills LLM node)
# ---------------------------------------------------------------------------

class SkillExtraction(BaseModel):
    posting_id: str
    ats: str
    company_slug: str
    skills: list[str]
    platforms: list[str]
    seniority: str | None
    years_experience: int | None
    comp_min: int | None
    comp_max: int | None
    comp_currency: str | None
    comp_interval: str | None


# ---------------------------------------------------------------------------
# Trend report (output of aggregate_trends, input to synthesize_radar)
# ---------------------------------------------------------------------------

class SkillTrend(BaseModel):
    skill: str
    count_current: int
    count_previous: int
    delta: int                    # count_current - count_previous
    pct_of_postings: float        # count_current / total postings


class TrendReport(BaseModel):
    window_days: int
    total_postings: int
    rising: list[SkillTrend]      # largest positive deltas
    falling: list[SkillTrend]     # largest negative deltas
    new: list[str]                # skills that went 0 → N
    top_platforms: list[SkillTrend]
    co_occurrences: list[tuple[str, str, int]]  # (skill_a, skill_b, count)


# ---------------------------------------------------------------------------
# Skills agent state
# ---------------------------------------------------------------------------

class SkillsState(dict):
    """State for the skills trend agent graph.

    new_postings:           postings loaded by load_deltas, awaiting extraction
    extractions:            fan-in accumulator; operator.add merges lists from Send nodes
    normalized_extractions: extractions after synonym collapse (output of normalize_taxonomy)
    unknown_skills:         skills not found in aliases.yaml; flagged for taxonomy review
    trend_report:           output of aggregate_trends
    radar_digest:           output of synthesize_radar (markdown string)
    watermark:              ISO timestamp; load_deltas reads postings updated after this
    """
    new_postings: list[dict]
    extractions: Annotated[list[SkillExtraction], operator.add]
    normalized_extractions: list[SkillExtraction]
    unknown_skills: list[str]
    trend_report: TrendReport | None
    radar_digest: str | None
    watermark: str | None


# ---------------------------------------------------------------------------
# Tracker agent — board resolution (output of resolve_board)
# ---------------------------------------------------------------------------

class BoardResolution(BaseModel):
    """Where a company's public ATS job board lives, as found by resolve_board.

    `method` records how it was determined — a cache hit and a deterministic
    probe cost nothing; 'agentic' means the LLM tool-use loop was needed. This
    is worth tracking: it tells us which companies have a non-obvious slug.
    """
    company_slug: str
    resolved: bool
    ats: str | None = None             # 'greenhouse' | 'lever' | 'ashby' | 'workable'
    ats_slug: str | None = None        # slug on that ATS (often != company_slug)
    board_url: str | None = None
    method: str | None = None          # 'cache' | 'hint' | 'deterministic' | 'agentic'
    attempted_slugs: list[str] = Field(default_factory=list)  # every slug probed, for audit


# ---------------------------------------------------------------------------
# Tracker agent state
# ---------------------------------------------------------------------------

class TrackerState(dict):
    """State for the tracker agent graph.

    The tracker maps over companies; each invocation handles one company, so
    this state is per-company (not a corpus like SkillsState).

    company:    {name, slug, ats?, ats_slug?, github_org?, blog_url?, domain?}
                — a watchlist entry, possibly with no known board yet
    resolution: output of resolve_board; None until that node runs

    Downstream fields (signals, snapshot, diff, dossier, score) are added by the
    fetch_signals / snapshot / diff / synthesize_dossier nodes in JAR-55 / JAR-56.
    """
    company: dict
    resolution: BoardResolution | None
