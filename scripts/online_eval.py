"""Online LLM-as-judge over live skills-agent traces.

Scores recent production `extract_one` runs in LangSmith with the *same* judge
the offline bake-off uses (eval.extraction_quality), and writes the verdict back
as run feedback. This turns the offline metric into a continuous quality signal
on real traffic — regressions show up in the LangSmith project without a fixed
dataset.

Idempotent: runs that already carry the feedback key are skipped, so it is safe
to re-run or schedule (e.g. GitHub Actions cron, like the ingestion/agent jobs).

Prerequisites: LANGSMITH_API_KEY (read + write feedback), and the judge model
available (local Ollama pulled, or ANTHROPIC_API_KEY for a claude-* judge).

Usage:
    python scripts/online_eval.py                      # last 24h, default judge
    python scripts/online_eval.py --since-hours 6 --limit 200
    python scripts/online_eval.py --judge-model claude-sonnet-4-6 --dry-run
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from langsmith import Client

from config import (
    EVAL_EXTRACTION_RUN_NAME,
    EVAL_FEEDBACK_KEY,
    EVAL_JUDGE_MODEL,
    LANGSMITH_PROJECT,
)
from eval.extraction_quality import make_online_evaluator
from eval.llm import build_llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _already_scored(client: Client, run_ids: list, key: str) -> set[str]:
    """Run IDs that already carry feedback `key` — so we never double-judge."""
    if not run_ids:
        return set()
    scored = {
        str(fb.run_id)
        for fb in client.list_feedback(run_ids=run_ids, feedback_key=[key])
    }
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Score live extraction traces with the LLM judge.")
    parser.add_argument("--project", default=LANGSMITH_PROJECT,
                        help="LangSmith project whose traces to score.")
    parser.add_argument("--run-name", default=EVAL_EXTRACTION_RUN_NAME,
                        help="Trace run name to evaluate (the per-posting extraction node).")
    parser.add_argument("--judge-model", default=EVAL_JUDGE_MODEL,
                        help="Model used as the quality judge (a stronger model is recommended).")
    parser.add_argument("--since-hours", type=int, default=24,
                        help="Only score runs started within this many hours.")
    parser.add_argument("--limit", type=int, default=100,
                        help="Max runs to pull per invocation.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Judge and log, but do not write feedback to LangSmith.")
    args = parser.parse_args()

    if not os.environ.get("LANGSMITH_API_KEY"):
        log.error("LANGSMITH_API_KEY is required — online eval reads traces and writes feedback.")
        sys.exit(1)

    client = Client()
    evaluator = make_online_evaluator(build_llm(args.judge_model), key=EVAL_FEEDBACK_KEY)
    start = datetime.now(timezone.utc) - timedelta(hours=args.since_hours)

    runs = list(client.list_runs(
        project_name=args.project,
        filter=f'eq(name, "{args.run_name}")',
        start_time=start,
        limit=args.limit,
    ))
    log.info("Pulled %d %r runs from project %r (last %dh).",
             len(runs), args.run_name, args.project, args.since_hours)

    scored = _already_scored(client, [r.id for r in runs], EVAL_FEEDBACK_KEY)

    judged = skipped = 0
    for run in runs:
        if str(run.id) in scored:
            skipped += 1
            continue

        result = evaluator(run)
        if result is None:
            skipped += 1  # not a scorable extraction (no JD / extraction)
            continue

        if args.dry_run:
            log.info("[dry-run] %s  score=%.2f  %s", run.id, result["score"], result["comment"][:120])
        else:
            client.create_feedback(
                run.id,
                key=result["key"],
                score=result["score"],
                comment=result["comment"],
                feedback_source_type="model",
            )
        judged += 1

    log.info("Online eval complete: %d judged, %d skipped (already scored or not applicable)%s.",
             judged, skipped, " [dry-run, no feedback written]" if args.dry_run else "")


if __name__ == "__main__":
    main()
