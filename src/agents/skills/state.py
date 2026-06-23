"""State objects for the skills trend agent graph."""
from __future__ import annotations

import operator
from typing import Annotated

from pydantic import BaseModel


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
