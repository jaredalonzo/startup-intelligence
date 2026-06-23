"""State objects for the startup tracker agent graph."""
from __future__ import annotations

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
# Tracker agent state
# ---------------------------------------------------------------------------

class TrackerState(dict):
    """State for the tracker agent graph.

    The tracker maps over companies; each invocation handles one company, so
    this state is per-company (not a corpus like SkillsState).

    company:    {name, slug, ats?, ats_slug?, github_org?, blog_url?}
                — a watchlist entry, possibly with no known board yet
    resolution: output of resolve_board; None until that node runs

    Downstream fields (signals, snapshot, diff, dossier, score) are added by the
    fetch_signals / snapshot / diff / synthesize_dossier nodes in JAR-55 / JAR-56.
    """
    company: dict
    resolution: BoardResolution | None
