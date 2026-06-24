"""Package download trends — npm + PyPI.

Fetches rolling download counts for the packages a company publishes:
  - PyPI  via pypistats.org  (one call returns last day/week/month)
  - npm   via api.npmjs.org  (one point call per period, gathered)

Each fetch is a point-in-time snapshot; the value is the trajectory across
runs (diff consecutive rows), mirroring github_repo_stats. Counts are stored
in package_downloads.

Pure deterministic ingestion; no LLM calls.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal
from urllib.parse import quote

import httpx
import psycopg
from psycopg.types.json import Jsonb
from pydantic import BaseModel

logger = logging.getLogger(__name__)

Registry = Literal["pypi", "npm"]

_PYPISTATS_API = "https://pypistats.org/api/packages"
_NPM_API = "https://api.npmjs.org/downloads/point"
_NPM_PERIODS: dict[str, str] = {"last_day": "last-day", "last_week": "last-week", "last_month": "last-month"}
_TIMEOUT = 10.0
_UA = "startup-intelligence/1.0 (package-trends)"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class PackageDownloads(BaseModel):
    company_slug: str
    registry: Registry
    package: str
    last_day: int | None
    last_week: int | None
    last_month: int | None
    raw: dict[str, Any]


# ---------------------------------------------------------------------------
# Per-registry fetchers
# ---------------------------------------------------------------------------

async def fetch_pypi_downloads(
    company_slug: str,
    package: str,
    client: httpx.AsyncClient,
) -> PackageDownloads | None:
    """Fetch rolling download counts for a PyPI *package* via pypistats."""
    url = f"{_PYPISTATS_API}/{quote(package, safe='')}/recent"
    try:
        resp = await client.get(url, headers={"User-Agent": _UA}, timeout=_TIMEOUT)
        resp.raise_for_status()
    except Exception:
        logger.warning("%s: PyPI download fetch failed for %s", company_slug, package, exc_info=True)
        return None

    payload = resp.json()
    data = payload.get("data", {})
    return PackageDownloads(
        company_slug=company_slug,
        registry="pypi",
        package=package,
        last_day=data.get("last_day"),
        last_week=data.get("last_week"),
        last_month=data.get("last_month"),
        raw=payload,
    )


async def _npm_point(package: str, period: str, client: httpx.AsyncClient) -> int | None:
    """One npm point query. Scoped packages (e.g. @scope/pkg) pass through as-is."""
    resp = await client.get(
        f"{_NPM_API}/{period}/{package}",
        headers={"User-Agent": _UA},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    downloads = resp.json().get("downloads")
    return int(downloads) if downloads is not None else None


async def fetch_npm_downloads(
    company_slug: str,
    package: str,
    client: httpx.AsyncClient,
) -> PackageDownloads | None:
    """Fetch rolling download counts for an npm *package* (day/week/month)."""
    try:
        results = await asyncio.gather(
            *(_npm_point(package, period, client) for period in _NPM_PERIODS.values())
        )
    except Exception:
        logger.warning("%s: npm download fetch failed for %s", company_slug, package, exc_info=True)
        return None

    counts = dict(zip(_NPM_PERIODS.keys(), results))
    return PackageDownloads(
        company_slug=company_slug,
        registry="npm",
        package=package,
        last_day=counts["last_day"],
        last_week=counts["last_week"],
        last_month=counts["last_month"],
        raw={"downloads": counts},
    )


_DISPATCH = {"pypi": fetch_pypi_downloads, "npm": fetch_npm_downloads}


# ---------------------------------------------------------------------------
# Fetch (all packages for a company)
# ---------------------------------------------------------------------------

async def fetch_package_downloads(
    company_slug: str,
    packages: list[tuple[str, str]],
    client: httpx.AsyncClient,
) -> list[PackageDownloads]:
    """Fetch every ``(registry, package)`` pair for a company, concurrently.

    Unknown registries are skipped; per-package failures resolve to None and
    are dropped (one bad package never sinks the rest).
    """
    async def _one(registry: str, package: str) -> PackageDownloads | None:
        fetch = _DISPATCH.get(registry)
        if fetch is None:
            logger.warning("%s: unknown registry %r for %s", company_slug, registry, package)
            return None
        return await fetch(company_slug, package, client)

    results = await asyncio.gather(*(_one(r, p) for r, p in packages))
    rows = [r for r in results if r is not None]
    logger.info("%s: %d/%d package download rows", company_slug, len(rows), len(packages))
    return rows


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------

def write_package_downloads(
    rows: list[PackageDownloads],
    conn: psycopg.Connection[dict[str, Any]],
) -> int:
    """Append download snapshots (time-series — never updated). Returns row count."""
    for d in rows:
        conn.execute(
            """
            INSERT INTO package_downloads
                (company_slug, registry, package, last_day, last_week, last_month, raw)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (d.company_slug, d.registry, d.package, d.last_day, d.last_week,
             d.last_month, Jsonb(d.raw)),
        )
    return len(rows)
