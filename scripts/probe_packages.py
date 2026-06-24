"""Watchlist package health check — dry run, no DB writes.

Probes every (registry, package) pair configured in the watchlist against the
*same* fetchers ingestion uses, then reports which resolve and which don't.
Because it reuses ingestion.signals.package_downloads, results match exactly
what a real ingestion run would persist — no duplicate HTTP logic to drift.

A package is reported MISSING when the production fetcher returns nothing for it
(404, network error, or unparseable response). This is a script-side diff:
configured pairs minus the pairs that came back.

Usage:
    python scripts/probe_packages.py            # all companies
    python scripts/probe_packages.py --slug anthropic
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import httpx

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion.signals.package_downloads import fetch_package_downloads  # noqa: E402
from ingestion.watchlist import COMPANIES, WatchlistEntry  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


async def probe(entries: list[WatchlistEntry]) -> tuple[list, list]:
    """Return (found, missing) rows across *entries*.

    found:   (slug, registry, package, last_month)
    missing: (slug, registry, package)
    """
    found: list[tuple[str, str, str, int | None]] = []
    missing: list[tuple[str, str, str]] = []

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(15.0),
        transport=httpx.AsyncHTTPTransport(retries=2),
    ) as client:
        for entry in entries:
            if not entry.packages:
                continue
            packages = [(r, p) for r, p in entry.packages]
            rows = await fetch_package_downloads(entry.slug, packages, client)

            # Script-side diff: what we asked for vs what came back.
            returned = {(row.registry, row.package): row for row in rows}
            for registry, pkg in packages:
                row = returned.get((registry, pkg))
                if row is not None:
                    found.append((entry.slug, registry, pkg, row.last_month))
                else:
                    missing.append((entry.slug, registry, pkg))

    return found, missing


def _report(found: list, missing: list) -> None:
    total = len(found) + len(missing)
    print(f"\n{'='*64}")
    print(f"PROBED {total} packages: {len(found)} exist, {len(missing)} missing")
    print(f"{'='*64}")

    print("\n--- EXISTS (last_month downloads) ---")
    for slug, reg, pkg, last_month in sorted(found):
        count = f"{last_month:>13,}" if isinstance(last_month, int) else f"{'n/a':>13}"
        print(f"  {slug:14} {reg:5} {pkg:36} {count}")

    if missing:
        print("\n--- MISSING (fetcher returned nothing) ---")
        for slug, reg, pkg in sorted(missing):
            print(f"  {slug:14} {reg:5} {pkg}")
    else:
        print("\nAll configured packages resolved. ✓")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Probe watchlist packages on PyPI/npm (dry run).")
    parser.add_argument("--slug", help="probe a single company by slug")
    args = parser.parse_args()

    entries = COMPANIES
    if args.slug:
        entries = [e for e in COMPANIES if e.slug == args.slug]
        if not entries:
            parser.error(f"no watchlist company with slug {args.slug!r}")

    found, missing = await probe(entries)
    _report(found, missing)
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    asyncio.run(main())
