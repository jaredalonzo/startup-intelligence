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
# recent give the current-vs-previous deltas; the rest are trajectory context.
TRACKER_DOSSIER_SNAPSHOT_LOOKBACK = 8

# Weights for the deterministic composite momentum score (score_trending). Each
# multiplies a per-run signal; star_growth is applied per 100 stars gained.
TRACKER_SCORE_WEIGHTS = {
    "eng_velocity": 1.0,      # net change in eng/product/data postings
    "posting_growth": 0.5,    # net change in total live postings
    "release_cadence": 2.0,   # new GitHub releases since last run
    "blog_cadence": 1.5,      # new blog posts since last run
    "star_growth": 1.0,       # GitHub star delta, per 100 stars
}

# Composite at/above which a company is a "top mover" worth a Linear task.
TRACKER_TOP_MOVER_COMPOSITE = 5.0
