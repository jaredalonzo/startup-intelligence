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
