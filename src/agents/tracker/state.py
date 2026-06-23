"""State objects for the startup tracker agent graph."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ingestion.ats.models import ATSSource


# ---------------------------------------------------------------------------
# Board resolution (output of resolve_board)
# ---------------------------------------------------------------------------

class BoardResolution(BaseModel):
    """Where a company's public ATS job board lives, as found by resolve_board.

    `method` records how it was determined — a cache hit and a deterministic
    probe cost nothing; 'agentic' means the LLM tool-use loop was needed. This
    is worth tracking: it tells us which companies have a non-obvious slug.
    """
    company_slug: str
    resolved: bool
    ats: ATSSource | None = None       # 'greenhouse' | 'lever' | 'ashby' | 'workable'
    ats_slug: str | None = None        # slug on that ATS (often != company_slug)
    board_url: str | None = None
    method: str | None = None          # 'cache' | 'hint' | 'deterministic' | 'agentic'
    attempted_slugs: list[str] = Field(default_factory=list)  # every slug probed, for audit


# ---------------------------------------------------------------------------
# Dossier inputs (output of load_signals — the per-company time series read
# back out of the store across the five tracker metrics)
# ---------------------------------------------------------------------------

class MetricDelta(BaseModel):
    """A single current-vs-previous reading on one tracked dimension.

    `delta` is current - previous (None when there's no prior point to diff
    against); `note` carries a short qualifier, e.g. 'no signal source wired'.
    """
    label: str
    current: float | None = None
    previous: float | None = None
    delta: float | None = None
    note: str | None = None


class DossierInputs(BaseModel):
    """Everything synthesize_dossier needs, assembled deterministically from the
    persisted snapshots/signals. The agent reads the store; it never fetches.

    Fields map onto the five tracker metrics: headcount (eng-weighted hiring
    velocity), open positions, technology/product, product evolution, and
    customers. `customers` has no ingestion source yet, so it is absent here and
    the dossier notes it as pending rather than inventing a number.
    """
    company_slug: str
    company_name: str
    snapshots_available: int                      # how many snapshots back the series goes

    # Headcount proxy + open positions (from snapshots + live postings).
    # `_delta` fields are run-over-run (what changed this run); `_window_delta`
    # fields are latest-vs-oldest loaded snapshot (the trend score leans on these,
    # since adjacent-snapshot hiring deltas are too noisy to score on).
    posting_count: int | None = None
    posting_count_delta: int | None = None
    posting_count_window_delta: int | None = None
    eng_count: int | None = None
    eng_count_delta: int | None = None
    eng_count_window_delta: int | None = None
    new_postings: int = 0                          # len(new_ids) on the latest snapshot
    removed_postings: int = 0                      # len(removed_ids) on the latest snapshot
    open_by_department: list[tuple[str, int]] = Field(default_factory=list)
    sample_titles: list[str] = Field(default_factory=list)

    # Technology / product (GitHub releases + star trajectory)
    recent_releases: list[dict] = Field(default_factory=list)       # {repo, tag, name, published_at}
    star_delta_by_repo: list[tuple[str, int]] = Field(default_factory=list)
    new_release_count: int = 0                     # releases first seen since the previous snapshot

    # Product evolution (engineering blog / changelog)
    recent_blog_posts: list[dict] = Field(default_factory=list)     # {title, url, published_at}
    new_blog_count: int = 0                        # posts first seen since the previous snapshot


# ---------------------------------------------------------------------------
# Trend score (output of score_trending)
# ---------------------------------------------------------------------------

class TrendScore(BaseModel):
    """Composite momentum score plus the LLM's qualitative judgment.

    `composite` is a deterministic weighted blend of the normalized deltas;
    `classification` / `rationale` are the LLM judgment flag. `is_top_mover`
    gates whether a Linear task is created for the company.
    """
    composite: float
    classification: Literal["accelerating", "steady", "cooling"]
    rationale: str
    is_top_mover: bool


# ---------------------------------------------------------------------------
# Tracker agent state
# ---------------------------------------------------------------------------

class TrackerState(dict):
    """State for the tracker agent graph.

    The tracker maps over companies; each invocation handles one company, so
    this state is per-company (not a corpus like SkillsState).

    company:          {name, slug, ats?, ats_slug?, github_org?, blog_url?}
                      — a watchlist entry, possibly with no known board yet
    resolution:       output of resolve_board; None until that node runs
    signals:          output of load_signals (DossierInputs); None until it runs
    meaningful_change: load_signals' deterministic gate — when False the graph
                      skips the LLM synthesis tail (cost control)
    dossier_markdown: output of synthesize_dossier
    trend_score:      output of score_trending
    dossier_url:      Notion page URL written/updated by write_dossier

    The fetch_signals / snapshot / diff nodes (JAR-55) write the store that
    load_signals reads; they slot in ahead of load_signals once built.
    """
    company: dict
    resolution: BoardResolution | None
    signals: DossierInputs | None
    meaningful_change: bool
    dossier_markdown: str | None
    trend_score: TrendScore | None
    dossier_url: str | None
