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
import os
import sys
import uuid
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


def _new_thread_id() -> str:
    """A unique checkpointer thread per run.

    The skills graph accumulates the ``extractions`` channel via operator.add. A
    fixed thread_id made the Postgres checkpointer reload the prior run's channel
    and add the new run's extractions on top — inflating the current window and
    re-persisting stale rows. A fresh thread per run isolates each run's channels;
    incremental reads now hang off the stored agent watermark, not checkpointed
    state. Intra-run resumability (re-invoking the same thread_id) is unaffected.
    """
    return f"skills-agent-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"


def _write_step_summary(result: dict) -> None:
    """Append a Markdown run summary to GITHUB_STEP_SUMMARY (CI only; no-op locally).

    Built from the graph result: the TrendReport (rising/falling/new skills,
    platforms) is the run's product and is computed in-graph, so it is reported
    straight from memory rather than re-derived from the store.
    """
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return

    report = result.get("trend_report")
    postings = len(result.get("new_postings") or [])
    extractions = len(result.get("extractions") or [])
    digest = result.get("radar_digest")
    unknown = result.get("unknown_skills") or []

    lines = [f"## Skills Agent Run — {datetime.now(timezone.utc):%Y-%m-%d}", ""]

    if report is None:
        lines += ["No new postings since the last watermark — extraction and trend "
                  "synthesis were skipped.", ""]
        with open(path, "a") as f:
            f.write("\n".join(lines))
        return

    lines += [
        f"**{postings} new postings**, {extractions} extractions · "
        f"digest: {'written to Notion' if digest else 'skipped'} · "
        f"window {report.window_days}d · corpus {report.total_postings} postings",
        "",
    ]

    def _trend_table(title: str, trends: list, n: int = 10) -> list[str]:
        out = [f"### {title}", "| Skill | Now | Prev | Δ | % postings |", "|---|--:|--:|--:|--:|"]
        for t in trends[:n]:
            out.append(f"| {t.skill} | {t.count_current} | {t.count_previous} | "
                       f"{t.delta:+d} | {t.pct_of_postings:.0%} |")
        out.append("")
        return out

    if report.rising:
        lines += _trend_table("Rising skills", report.rising)
    if report.falling:
        lines += _trend_table("Falling skills", report.falling)
    if report.new:
        names = [getattr(x, "skill", x) for x in report.new]
        lines += ["### Newly appearing", ", ".join(str(n) for n in names) or "—", ""]
    if report.top_platforms:
        lines += ["### Top platforms", "| Platform | Now | % postings |", "|---|--:|--:|"]
        for t in report.top_platforms[:10]:
            lines.append(f"| {t.skill} | {t.count_current} | {t.pct_of_postings:.0%} |")
        lines.append("")
    if unknown:
        lines += [f"### Unknown skills flagged for taxonomy review ({len(unknown)})",
                  ", ".join(unknown[:40]), ""]

    with open(path, "a") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the skills trend agent.")
    parser.add_argument(
        "--window-days",
        type=int,
        default=None,
        help="Force a specific lookback window in days, ignoring the stored watermark.",
    )
    parser.add_argument(
        "--all-roles",
        action="store_true",
        help="Extract every posting, skipping the technical-only role filter "
             "(a deliberate broad analysis; the agent targets FDE/TAM/CSE/eng by default).",
    )
    args = parser.parse_args()

    thread_id = _new_thread_id()
    log.info("Starting skills agent run (thread=%s)", thread_id)

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
            # max_concurrency=1 serializes the extract_one fan-out: Ollama Cloud's
            # free tier allows a single concurrent request, so parallel extractions
            # would be rejected (429). The tracker is already a sequential map, so
            # only the skills agent's fan-out needs this guard.
            config={
                "configurable": {"thread_id": thread_id, "all_roles": args.all_roles},
                "callbacks": [guard],
                "max_concurrency": 1,
            },
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
    _write_step_summary(result)


if __name__ == "__main__":
    main()
