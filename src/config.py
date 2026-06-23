"""Central configuration — model objects, thresholds, and tunables."""
from langchain_anthropic import ChatAnthropic
# To switch to Ollama: swap the two import lines and update the model strings below.
# from langchain_ollama import ChatOllama

# ---------------------------------------------------------------------------
# LLM instances — import these in nodes, never instantiate a client there
# ---------------------------------------------------------------------------

# High-volume extraction node (runs once per posting — cost scales with volume)
EXTRACTION_LLM = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=1024)  # type: ignore[call-arg]

# Synthesis nodes (once per run — quality matters more than cost)
SYNTHESIS_LLM = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=2048)  # type: ignore[call-arg]

# Tracker board resolution (LLM + tool-use; one-time cost per new company) —
# a once-per-company, quality-over-cost task, so it reuses the synthesis model.
# Whatever model SYNTHESIS_LLM points to must support tool calling.
RESOLVE_LLM = SYNTHESIS_LLM

# ---------------------------------------------------------------------------
# Skills agent
# ---------------------------------------------------------------------------

# Number of days to look back when no watermark exists (first run)
SKILLS_DEFAULT_WINDOW_DAYS = 30

# Minimum number of postings a skill must appear in to be included in the radar
SKILLS_MIN_POSTING_COUNT = 3

# Top N rising / falling skills to include in the TrendReport
SKILLS_TOP_N = 20

# Threshold for creating a Linear gap task (skill appears in >= X% of postings)
SKILLS_GAP_TASK_THRESHOLD_PCT = 0.15

# ---------------------------------------------------------------------------
# Target archetypes for the skills radar
# ---------------------------------------------------------------------------

TARGET_ROLES = [
    "Field Data Engineer (FDE)",
    "Technical Account Manager (TAM)",
    "Customer Solutions Engineer (CSE)",
    "Implementation Engineer",
]

# ---------------------------------------------------------------------------
# Tracker agent
# ---------------------------------------------------------------------------

# Max ATS-slug probes the resolve_board tool-use loop may make per company
# before giving up. Caps the cost of an unresolvable company.
TRACKER_RESOLVE_MAX_PROBES = 6

# ---------------------------------------------------------------------------
# Tracker dossier (load_signals / synthesize_dossier / score_trending)
# ---------------------------------------------------------------------------

# How many snapshots back to load for the per-company time series. The two most
# recent give the run-over-run deltas (what changed this run); latest-vs-oldest
# gives the window trend that score_trending leans on, since run-over-run hiring
# deltas between adjacent snapshots are too noisy to score on.
TRACKER_DOSSIER_SNAPSHOT_LOOKBACK = 8

# Composite momentum score (score_trending). Each component is normalized to a
# bounded range, weighted (weights sum to 1), and scaled to ~0-100. Hiring
# velocity leads by design — it is the project's core growth proxy — so it
# dominates once snapshot history is deep enough for the trend to be meaningful;
# the GitHub/blog activity terms corroborate and are capped (below) so they
# cannot saturate the score on their own. (See the live-run calibration note.)
TRACKER_SCORE_WEIGHTS = {
    "eng_velocity": 0.45,     # window growth rate of eng/product/data postings
    "posting_growth": 0.20,   # window growth rate of total live postings
    "release_cadence": 0.15,  # new GitHub releases since last run (capped)
    "blog_cadence": 0.10,     # new blog posts since last run (capped)
    "star_growth": 0.10,      # GitHub star delta (capped)
}

# Caps that bound each activity component before weighting (prevent saturation).
TRACKER_SCORE_CAPS = {
    "release_cadence": 8,
    "blog_cadence": 6,
    "star_growth": 1000,
}

# Composite bands (on the ~0-100 scale) → classification. A company is a
# "top mover" iff it lands in the accelerating band, so the deterministic flag
# can never contradict the label. Calibrated against the live distribution.
TRACKER_ACCEL_BAND = 12.0
TRACKER_COOL_BAND = -3.0
