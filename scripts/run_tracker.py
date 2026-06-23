"""Startup tracker agent entrypoint.

Maps over the watchlist and runs the per-company tracker graph:

    resolve_board (async, LLM+tool-use, cached)
      → load_signals (det, reads the store)
      → gate: meaningful change?  (skip the LLM tail if not)
      → synthesize_dossier (LLM)
      → score_trending (det composite + deterministic classification; LLM rationale)
      → write_dossier (Notion upsert; flags top movers for Linear)

The graph is *per company*, so this script owns the map over companies (unlike
the skills agent, which runs a single corpus-wide invoke).

`resolve_board` is async, so the graph runs via `ainvoke` — which requires an
*async* checkpointer (the sync PostgresSaver in graph.py raises on aget_tuple).
We build an AsyncPostgresSaver here with the same serde registration.

Prerequisite: ingestion must have populated the store first
(`python scripts/ingest.py`). The tracker reads snapshots/signals; it never
fetches ATS endpoints itself.

Usage:
    python scripts/run_tracker.py                          # whole watchlist
    python scripts/run_tracker.py --slug anthropic --slug openai
    python scripts/run_tracker.py --limit 5                # first 5 entries
    python scripts/run_tracker.py --dry-run                # no Notion writes
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg import AsyncConnection
from psycopg.rows import dict_row

from agents.tracker import dossier as dossier_mod
from agents.tracker.graph import compile_graph
from agents.tracker.state import BoardResolution, DossierInputs, TrendScore
from config import LLM_CALL_BUDGET_PER_RUN, LLM_TOKEN_BUDGET_PER_RUN
from ingestion.watchlist import COMPANIES
from observability import CostGuard, init_tracing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger(__name__)

init_tracing()


@asynccontextmanager
async def _async_checkpointer() -> AsyncIterator[AsyncPostgresSaver]:
    """AsyncPostgresSaver the tracker's async graph needs (graph.py's sync saver
    only implements the blocking interface). Serde registration mirrors it."""
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[BoardResolution, DossierInputs, TrendScore]
    )
    async with await AsyncConnection.connect(
        os.environ["DATABASE_URL"],
        autocommit=True,
        prepare_threshold=0,
        row_factory=dict_row,
    ) as conn:
        yield AsyncPostgresSaver(conn, serde=serde)


def _select_companies(slugs: list[str] | None, limit: int | None) -> list[dict]:
    """Resolve the watchlist down to the company dicts the graph expects."""
    entries = COMPANIES
    if slugs:
        wanted = set(slugs)
        entries = [e for e in entries if e.slug in wanted]
        missing = wanted - {e.slug for e in entries}
        if missing:
            log.warning("Unknown slugs ignored: %s", ", ".join(sorted(missing)))
    if limit is not None:
        entries = entries[:limit]
    return [dataclasses.asdict(e) for e in entries]


def _install_dry_run() -> None:
    """Replace the Notion writer with a no-op so a run scores without publishing.

    Patches the symbol on the dossier module (where write_dossier looks it up),
    leaving production code untouched.
    """
    def _fake_upsert(markdown: str, company_name: str) -> str:
        log.info("[dry-run] would upsert Notion dossier for %s (%d chars)",
                 company_name, len(markdown))
        return "dry-run://notion-skipped"

    dossier_mod.upsert_company_dossier = _fake_upsert  # type: ignore[assignment]


async def _run_company(graph, company: dict, guard: CostGuard) -> dict:
    """Invoke the per-company graph once; return a compact result row.

    The shared CostGuard accumulates across all companies so the budget is
    per-run (whole watchlist), not per-company.
    """
    slug = company["slug"]
    try:
        result = await graph.ainvoke(
            {"company": company},
            config={"configurable": {"thread_id": f"tracker-{slug}"}, "callbacks": [guard]},
        )
    except Exception:
        log.exception("tracker: %s failed; continuing", slug)
        return {"slug": slug, "name": company.get("name"), "error": True}

    resolution = result.get("resolution")
    score = result.get("trend_score")
    return {
        "slug": slug,
        "name": company.get("name"),
        "resolved": bool(resolution and resolution.resolved),
        "method": resolution.method if resolution else None,
        "changed": bool(result.get("meaningful_change")),
        "composite": score.composite if score else None,
        "classification": score.classification if score else None,
        "top_mover": bool(score and score.is_top_mover),
        "dossier_url": result.get("dossier_url"),
        "error": False,
    }


def _print_summary(rows: list[dict]) -> None:
    """Per-company table sorted by composite — the calibration artifact."""
    def _key(r: dict) -> float:
        return r["composite"] if r.get("composite") is not None else float("-inf")

    print("\n=== Tracker run summary ===")
    print(f"{'company':<16}{'resolved':<14}{'changed':<9}"
          f"{'composite':>10}  {'class':<13}{'top?':<5}")
    print("-" * 72)
    for r in sorted(rows, key=_key, reverse=True):
        if r.get("error"):
            print(f"{r['slug']:<16}{'ERROR':<14}")
            continue
        comp = f"{r['composite']:.2f}" if r["composite"] is not None else "-"
        resolved = f"yes ({r['method']})" if r["resolved"] else "no"
        print(f"{r['slug']:<16}{resolved:<14}"
              f"{'yes' if r['changed'] else 'no':<9}"
              f"{comp:>10}  {(r['classification'] or '-'):<13}"
              f"{'★' if r['top_mover'] else '':<5}")

    scored = [r for r in rows if r.get("composite") is not None]
    print("-" * 72)
    print(f"companies={len(rows)}  resolved={sum(r.get('resolved', False) for r in rows)}  "
          f"scored={len(scored)}  top_movers={sum(r.get('top_mover', False) for r in rows)}  "
          f"errors={sum(r.get('error', False) for r in rows)}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the startup tracker agent.")
    parser.add_argument("--slug", action="append", dest="slugs",
                        help="Restrict to these watchlist slugs (repeatable).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N watchlist entries.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Score everything but skip the Notion dossier writes.")
    args = parser.parse_args()

    if args.dry_run:
        _install_dry_run()
        log.info("Dry run: Notion writes are disabled.")

    companies = _select_companies(args.slugs, args.limit)
    if not companies:
        log.error("No companies selected; nothing to do.")
        return
    log.info("Starting tracker run over %d companies", len(companies))

    guard = CostGuard(LLM_CALL_BUDGET_PER_RUN, LLM_TOKEN_BUDGET_PER_RUN)
    rows: list[dict] = []
    async with _async_checkpointer() as checkpointer:
        await checkpointer.setup()
        graph = compile_graph(checkpointer)
        # Sequential map: one async connection reused across invokes, in order.
        # Parallel mapping would need a connection per task — left for later.
        for company in companies:
            rows.append(await _run_company(graph, company, guard))

    guard.log_summary()
    _print_summary(rows)


if __name__ == "__main__":
    asyncio.run(main())
