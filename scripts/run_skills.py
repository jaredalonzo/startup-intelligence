"""Skills trend agent entrypoint.

Loads deltas from the postings table since the last watermark, runs skill
extraction and trend synthesis, and writes the radar digest to Notion.

Usage:
    python scripts/run_skills.py
    python scripts/run_skills.py --window-days 7   # override lookback on this run
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from agents.skills.graph import compile_graph, make_checkpointer
from config import LLM_CALL_BUDGET_PER_RUN, LLM_TOKEN_BUDGET_PER_RUN
from observability import CostGuard, init_tracing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger(__name__)

init_tracing()

_THREAD_ID = "skills-agent"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the skills trend agent.")
    parser.add_argument(
        "--window-days",
        type=int,
        default=None,
        help="Force a specific lookback window in days, ignoring the stored watermark.",
    )
    args = parser.parse_args()

    log.info("Starting skills agent run (thread=%s)", _THREAD_ID)

    initial: dict = {}  # type: ignore[type-arg]
    if args.window_days:
        watermark = datetime.now(timezone.utc) - timedelta(days=args.window_days)
        initial["watermark"] = watermark.isoformat()
        log.info("Watermark overridden: looking back %d days", args.window_days)

    guard = CostGuard(LLM_CALL_BUDGET_PER_RUN, LLM_TOKEN_BUDGET_PER_RUN)
    with make_checkpointer() as checkpointer:
        checkpointer.setup()
        graph = compile_graph(checkpointer)
        result = graph.invoke(
            initial,
            config={"configurable": {"thread_id": _THREAD_ID}, "callbacks": [guard]},
        )
    guard.log_summary()

    postings_count = len(result.get("new_postings") or [])
    extractions_count = len(result.get("extractions") or [])
    digest = result.get("radar_digest")

    log.info(
        "Run complete — postings=%d extractions=%d digest=%s",
        postings_count,
        extractions_count,
        "written to Notion" if digest else "skipped (no new postings)",
    )


if __name__ == "__main__":
    main()
