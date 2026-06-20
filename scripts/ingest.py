"""Scheduled ingestion entrypoint.

Fetches all watchlist companies from the DB, pulls their ATS job boards,
writes postings, snapshots, and watermarks. One company failure never aborts
the full run — errors are logged and the watermark is left unchanged so the
company is retried on the next run.

Usage:
    python scripts/ingest.py
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

load_dotenv()

from ingestion.ats import ashby, greenhouse, lever, workable
from ingestion.ats.models import ATSSource, Posting
from ingestion.diff import compute_diff
from ingestion.snapshot import update_watermark, upsert_postings, write_snapshot
from store.db import get_connection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_FETCHERS: dict[ATSSource, object] = {
    "greenhouse": greenhouse.fetch_postings,
    "lever":      lever.fetch_postings,
    "ashby":      ashby.fetch_postings,
    "workable":   workable.fetch_postings,
}


async def ingest_company(
    slug: str,
    ats: ATSSource,
    ats_slug: str,
    client: httpx.AsyncClient,
) -> dict[str, object]:
    """Fetch, diff, persist one company. Returns a summary dict."""
    fetch = _FETCHERS[ats]
    postings: list[Posting] = await fetch(ats_slug, client)  # type: ignore[operator]
    if not postings:
        # A 200 with 0 jobs is suspicious — could be a transient CDN response.
        # Skip the snapshot so we don't record a false total-closure; retry next run.
        log.warning("%s (%s:%s) returned 0 postings — skipping snapshot", slug, ats, ats_slug)
        return {"slug": slug, "ats": ats, "total": 0, "new": 0, "removed": 0, "snapshot_id": None}
    # Adapters use ats_slug as company_slug; rebind to canonical slug for FK integrity.
    if ats_slug != slug:
        postings = [p.model_copy(update={"company_slug": slug}) for p in postings]
    current_ids = {p.id for p in postings}

    with get_connection() as conn:
        new_ids, removed_ids = compute_diff(slug, ats, current_ids, conn)
        upsert_postings(postings, conn)
        snapshot_id = write_snapshot(slug, postings, new_ids, removed_ids, conn)
        update_watermark(slug, ats, conn)
        conn.commit()

    return {
        "slug": slug,
        "ats": ats,
        "total": len(postings),
        "new": len(new_ids),
        "removed": len(removed_ids),
        "snapshot_id": snapshot_id,
    }


async def main() -> None:
    with get_connection() as conn:
        companies = conn.execute(
            "SELECT slug, ats, ats_slug FROM companies ORDER BY slug"
        ).fetchall()

    log.info("Starting ingestion run for %d companies", len(companies))

    async with httpx.AsyncClient(follow_redirects=True) as client:
        for row in companies:
            slug, ats, ats_slug = row["slug"], row["ats"], row["ats_slug"]
            try:
                result = await ingest_company(slug, ats, ats_slug, client)
                log.info(
                    "%s (%s:%s)  total=%d  new=%d  removed=%d  snapshot=%s",
                    result["slug"], result["ats"], ats_slug,
                    result["total"], result["new"], result["removed"], result["snapshot_id"],
                )
            except Exception:
                log.exception("Failed to ingest %s (%s:%s) — skipping", slug, ats, ats_slug)

    log.info("Ingestion run complete")


if __name__ == "__main__":
    asyncio.run(main())
