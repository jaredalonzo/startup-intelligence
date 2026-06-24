"""Central configuration — model objects, thresholds, and tunables."""
import os

# To switch to Anthropic: swap the two import lines and update the model strings below.
# from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

# ---------------------------------------------------------------------------
# LLM instances — import these in nodes, never instantiate a client there
# ---------------------------------------------------------------------------

# High-volume extraction node (runs once per posting — cost scales with volume)
EXTRACTION_LLM = ChatOllama(model="qwen2.5:14b", temperature=0)

# Synthesis nodes (once per run — quality matters more than cost)
SYNTHESIS_LLM = ChatOllama(model="qwen2.5:14b", temperature=0)

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

# Composite momentum score (score_trending). Every component is normalized to a
# common [-1, 1] scale (activity terms [0, 1]), weighted (weights sum to 1), and
# scaled to a ~[-100, 100] index. Hiring leads by design — it is the project's
# core growth proxy. The hiring weights sum to 0.65 and the activity weights to
# 0.35, so the activity-only ceiling is 35; ACCEL_BAND sits *above* that (below),
# which means activity alone can never make a company a top mover — it can only
# corroborate real hiring momentum. (Fixes the release-dominated live-run bug.)
TRACKER_SCORE_WEIGHTS = {
    "eng_velocity": 0.45,     # eng/product/data hiring growth (normalized, leads)
    "posting_growth": 0.20,   # total live-posting growth (normalized)
    "release_cadence": 0.15,  # new GitHub releases since last run (capped)
    "blog_cadence": 0.10,     # new blog posts since last run (capped)
    "star_growth": 0.10,      # GitHub star delta (capped)
}

# Reference window growth rate that earns a hiring term full credit (maps to 1.0).
# Growing eng/total postings this fast over the tracked window saturates the
# hiring term; this puts hiring on the same 0-1 scale as the activity terms,
# instead of entering as a tiny raw fraction that the activity terms drown out.
TRACKER_GROWTH_REF = 0.25

# Caps that bound each activity component before weighting (prevent saturation).
TRACKER_SCORE_CAPS = {
    "release_cadence": 8,
    "blog_cadence": 6,
    "star_growth": 1000,
}

# Composite bands → classification. A company is a "top mover" iff it lands in the
# accelerating band, so the deterministic flag can never contradict the label.
# ACCEL_BAND (40) is deliberately > the activity-only ceiling (35): a top mover
# must have genuine hiring momentum, not just an active GitHub/blog. Recalibrate
# against the live distribution as snapshot history deepens.
TRACKER_ACCEL_BAND = 40.0
TRACKER_COOL_BAND = -5.0

# ---------------------------------------------------------------------------
# Outputs — Linear (outputs/linear.py)
# ---------------------------------------------------------------------------

# Destination for agent-created issues, in the "Jared Alonzo" (JAR) team's
# "Startup & Skills Intelligence Agents" project. Defaults are the discovered
# workspace IDs (not secrets); override per-environment via env if needed.
# LINEAR_API_KEY is read from the environment in outputs/linear.py (secret).
LINEAR_TEAM_ID = os.getenv("LINEAR_TEAM_ID", "4249174f-4fad-4957-b406-93a4fd30ec2c")
LINEAR_PROJECT_ID = os.getenv("LINEAR_PROJECT_ID", "a147d3f0-638d-4ba9-af05-dc15fdc7d9e8")

# ---------------------------------------------------------------------------
# Observability — LangSmith tracing (observability.py)
# ---------------------------------------------------------------------------

# LangSmith project the two graphs' traces land in. Tracing is enabled by the
# run entrypoints via observability.init_tracing(), and only when
# LANGSMITH_API_KEY is present in the environment.
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "startup-intelligence")

# Per-run soft LLM budgets (observability.CostGuard). Crossing either logs a
# one-time WARNING; the run still finishes (alerting, not blocking). Generous by
# design — meant to catch runaway loops/fan-outs, not normal volume (a full
# skills run is ~1 call per posting). Tokens matter most on the Anthropic path;
# Ollama may not report them, so the call budget is the reliable rail.
LLM_CALL_BUDGET_PER_RUN = int(os.getenv("LLM_CALL_BUDGET_PER_RUN", "5000"))
LLM_TOKEN_BUDGET_PER_RUN = int(os.getenv("LLM_TOKEN_BUDGET_PER_RUN", "5000000"))

# ---------------------------------------------------------------------------
# Evaluation — LLM-as-judge for extraction quality (eval/extraction_quality.py)
# ---------------------------------------------------------------------------

# Model that grades extraction quality, both in the offline bake-off
# (scripts/eval_models.py, overridable via --judge-model) and the online
# evaluator over live traces (scripts/online_eval.py). A stronger model than the
# extraction model is recommended so the judge isn't graded by its own peer.
EVAL_JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "qwen2.5:14b")

# LangSmith feedback key the judge writes; also the run name it scores online.
EVAL_FEEDBACK_KEY = "extraction_quality"
EVAL_EXTRACTION_RUN_NAME = "extract_one"
