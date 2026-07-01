"""Scheduled ingestion entrypoint.

Fetches all watchlist companies from the DB, pulls their ATS job boards,
writes postings, snapshots, and watermarks. Also fetches GitHub signals
(releases + repo stats) for companies with a github_org configured.
One company failure never aborts the full run.

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

from config import INGEST_DROP_GUARD_MAX_FRACTION, INGEST_DROP_GUARD_MIN_PREV
from ingestion.ats import ashby, greenhouse, lever, workable
from ingestion.ats.models import ATSSource, Posting
from ingestion.diff import compute_diff
from ingestion.signals.blog_rss import fetch_blog_posts, write_blog_posts
from ingestion.signals.github_org import fetch_github_signals, write_github_signals
from ingestion.signals.package_downloads import fetch_package_downloads, write_package_downloads
from ingestion.snapshot import (
    is_suspect_drop,
    latest_posting_count,
    update_watermark,
    upsert_postings,
    write_snapshot,
)
from store.db import get_connection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_FETCHERS: dict[ATSSource, object] = {
    "greenhouse": greenhouse.fetch_postings,
    "lever":      lever.fetch_postings,
    "ashby":      ashby.fetch_postings,
    "workable":   workable.fetch_postings,
}


def _untrusted_fetch_result(slug: str, ats: ATSSource, total: int) -> dict[str, object]:
    """Summary for a company whose ATS fetch was not trusted into the time series.

    Used by both skip paths (an empty fetch and an implausible-drop fetch): no
    snapshot is written and no dependent signal work runs, so all counts are zero.
    """
    return {"slug": slug, "ats": ats, "total": total, "new": 0, "removed": 0,
            "snapshot_id": None, "gh_releases": 0, "gh_repos": 0, "blog_posts": 0,
            "package_downloads": 0}


async def ingest_company(
    slug: str,
    ats: ATSSource,
    ats_slug: str,
    github_org: str | None,
    blog_rss_url: str | None,
    packages: list[tuple[str, str]],
    client: httpx.AsyncClient,
) -> dict[str, object]:
    """Fetch, diff, persist one company. Returns a summary dict."""
    fetch = _FETCHERS[ats]
    postings: list[Posting] = await fetch(ats_slug, client)  # type: ignore[operator]
    if not postings:
        log.warning("%s (%s:%s) returned 0 postings — skipping snapshot", slug, ats, ats_slug)
        return _untrusted_fetch_result(slug, ats, 0)
    if ats_slug != slug:
        postings = [p.model_copy(update={"company_slug": slug}) for p in postings]
    current_ids = {p.id for p in postings}

    # ATS + snapshot (DB write)
    with get_connection() as conn:
        # Guard the time series against a partial fetch: an implausible single-run
        # collapse (missed page, transient 5xx short body, dropped detail fetches)
        # would otherwise land as a permanent false "hiring collapse" snapshot. On a
        # suspect fetch skip the whole write — leave the last good snapshot standing
        # and re-try next run — rather than corrupting the append-only window.
        prev_count = latest_posting_count(slug, conn)
        if is_suspect_drop(
            prev_count, len(postings),
            min_prev=INGEST_DROP_GUARD_MIN_PREV,
            max_drop_fraction=INGEST_DROP_GUARD_MAX_FRACTION,
        ):
            log.warning(
                "%s (%s:%s) fetched %d postings vs previous %d — implausible drop; "
                "treating as a partial fetch and skipping snapshot (time series unchanged)",
                slug, ats, ats_slug, len(postings), prev_count,
            )
            return _untrusted_fetch_result(slug, ats, len(postings))
        new_ids, removed_ids = compute_diff(slug, ats, current_ids, conn)
        upsert_postings(postings, conn)
        snapshot_id = write_snapshot(slug, postings, new_ids, removed_ids, conn)
        update_watermark(slug, ats, conn)
        conn.commit()

    # GitHub signals (independent — failure doesn't affect ATS result)
    gh_releases = gh_repos = 0
    if github_org:
        try:
            releases, snapshots = await fetch_github_signals(slug, github_org, client)
            with get_connection() as conn:
                gh_releases, gh_repos = write_github_signals(releases, snapshots, conn)
                conn.commit()
        except Exception:
            log.exception("%s: GitHub signal fetch failed — skipping", slug)

    # Blog RSS (independent)
    blog_posts_new = 0
    if blog_rss_url:
        try:
            posts = await fetch_blog_posts(slug, blog_rss_url, client)
            with get_connection() as conn:
                blog_posts_new = write_blog_posts(posts, conn)
                conn.commit()
        except Exception:
            log.exception("%s: Blog RSS fetch failed — skipping", slug)

    # Package downloads (npm/PyPI — independent)
    package_rows = 0
    if packages:
        try:
            downloads = await fetch_package_downloads(slug, packages, client)
            with get_connection() as conn:
                package_rows = write_package_downloads(downloads, conn)
                conn.commit()
        except Exception:
            log.exception("%s: Package download fetch failed — skipping", slug)

    return {
        "slug": slug,
        "ats": ats,
        "total": len(postings),
        "new": len(new_ids),
        "removed": len(removed_ids),
        "snapshot_id": snapshot_id,
        "gh_releases": gh_releases,
        "gh_repos": gh_repos,
        "blog_posts": blog_posts_new,
        "package_downloads": package_rows,
    }


async def main() -> None:
    with get_connection() as conn:
        companies = conn.execute(
            "SELECT slug, ats, ats_slug, github_org, blog_rss_url, packages "
            "FROM companies ORDER BY slug"
        ).fetchall()

    log.info("Starting ingestion run for %d companies", len(companies))

    # Bounded timeout so one hung board can't stall the whole run, plus
    # connection-level retries (httpx backs off between attempts) for transient
    # network blips. Per-board try/except below still isolates any hard failure.
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(15.0),
        transport=httpx.AsyncHTTPTransport(retries=2),
    ) as client:
        for row in companies:
            slug, ats, ats_slug, github_org, blog_rss_url = (
                row["slug"], row["ats"], row["ats_slug"], row["github_org"], row["blog_rss_url"]
            )
            # packages is a JSONB array of {"registry", "package"} objects.
            packages = [(p["registry"], p["package"]) for p in (row["packages"] or [])]
            try:
                result = await ingest_company(
                    slug, ats, ats_slug, github_org, blog_rss_url, packages, client
                )
                log.info(
                    "%s (%s:%s)  total=%d  new=%d  removed=%d  snapshot=%s  "
                    "gh_releases=%d  gh_repos=%d  blog_posts=%d  pkg_downloads=%d",
                    result["slug"], result["ats"], ats_slug,
                    result["total"], result["new"], result["removed"], result["snapshot_id"],
                    result["gh_releases"], result["gh_repos"], result["blog_posts"],
                    result["package_downloads"],
                )
            except Exception:
                log.exception("Failed to ingest %s (%s:%s) — skipping", slug, ats, ats_slug)

    log.info("Ingestion run complete")


if __name__ == "__main__":
    asyncio.run(main())
