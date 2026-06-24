"""Unit tests for the npm/PyPI package download fetcher.

Network is mocked with respx; the DB write is exercised against a fake conn.
No live HTTP, no live Postgres.
"""
from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest
import respx

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion.signals.package_downloads import (  # noqa: E402
    PackageDownloads,
    fetch_npm_downloads,
    fetch_package_downloads,
    fetch_pypi_downloads,
    write_package_downloads,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class _FakeConn:
    """Records every INSERT so tests can assert what was written."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def execute(self, sql, params=None):  # noqa: ANN001
        self.calls.append((sql, params))
        return self


# ---------------------------------------------------------------------------
# PyPI
# ---------------------------------------------------------------------------

@respx.mock
@pytest.mark.asyncio
async def test_fetch_pypi_parses_recent():
    respx.get("https://pypistats.org/api/packages/anthropic/recent").mock(
        return_value=httpx.Response(
            200,
            json={"data": {"last_day": 10, "last_week": 70, "last_month": 300},
                  "package": "anthropic", "type": "recent_downloads"},
        )
    )
    async with httpx.AsyncClient() as client:
        row = await fetch_pypi_downloads("anthropic", "anthropic", client)

    assert row is not None
    assert (row.registry, row.package) == ("pypi", "anthropic")
    assert (row.last_day, row.last_week, row.last_month) == (10, 70, 300)
    assert row.raw["package"] == "anthropic"


@respx.mock
@pytest.mark.asyncio
async def test_fetch_pypi_404_returns_none():
    respx.get("https://pypistats.org/api/packages/nope/recent").mock(
        return_value=httpx.Response(404)
    )
    async with httpx.AsyncClient() as client:
        assert await fetch_pypi_downloads("acme", "nope", client) is None


# ---------------------------------------------------------------------------
# npm (incl. scoped packages)
# ---------------------------------------------------------------------------

@respx.mock
@pytest.mark.asyncio
async def test_fetch_npm_gathers_three_periods_scoped():
    base = "https://api.npmjs.org/downloads/point"
    pkg = "@anthropic-ai/sdk"
    respx.get(f"{base}/last-day/{pkg}").mock(return_value=httpx.Response(200, json={"downloads": 5}))
    respx.get(f"{base}/last-week/{pkg}").mock(return_value=httpx.Response(200, json={"downloads": 35}))
    respx.get(f"{base}/last-month/{pkg}").mock(return_value=httpx.Response(200, json={"downloads": 150}))

    async with httpx.AsyncClient() as client:
        row = await fetch_npm_downloads("anthropic", pkg, client)

    assert row is not None
    assert (row.registry, row.package) == ("npm", pkg)
    assert (row.last_day, row.last_week, row.last_month) == (5, 35, 150)


@respx.mock
@pytest.mark.asyncio
async def test_fetch_npm_one_bad_period_returns_none():
    base = "https://api.npmjs.org/downloads/point"
    respx.get(f"{base}/last-day/pkg").mock(return_value=httpx.Response(200, json={"downloads": 1}))
    respx.get(f"{base}/last-week/pkg").mock(return_value=httpx.Response(500))
    respx.get(f"{base}/last-month/pkg").mock(return_value=httpx.Response(200, json={"downloads": 9}))

    async with httpx.AsyncClient() as client:
        assert await fetch_npm_downloads("acme", "pkg", client) is None


# ---------------------------------------------------------------------------
# Dispatch over a company's package list
# ---------------------------------------------------------------------------

@respx.mock
@pytest.mark.asyncio
async def test_fetch_package_downloads_mixes_registries_and_drops_failures():
    respx.get("https://pypistats.org/api/packages/anthropic/recent").mock(
        return_value=httpx.Response(200, json={"data": {"last_day": 1, "last_week": 2, "last_month": 3}})
    )
    base = "https://api.npmjs.org/downloads/point"
    for period in ("last-day", "last-week", "last-month"):
        respx.get(f"{base}/{period}/@anthropic-ai/sdk").mock(
            return_value=httpx.Response(200, json={"downloads": 7})
        )

    packages = [
        ("pypi", "anthropic"),
        ("npm", "@anthropic-ai/sdk"),
        ("cargo", "ignored"),  # unknown registry → skipped
    ]
    async with httpx.AsyncClient() as client:
        rows = await fetch_package_downloads("anthropic", packages, client)

    assert len(rows) == 2
    assert {r.registry for r in rows} == {"pypi", "npm"}


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------

def test_write_package_downloads_inserts_each_row():
    rows = [
        PackageDownloads(company_slug="anthropic", registry="pypi", package="anthropic",
                         last_day=1, last_week=2, last_month=3, raw={}),
        PackageDownloads(company_slug="anthropic", registry="npm", package="@anthropic-ai/sdk",
                         last_day=4, last_week=5, last_month=6, raw={}),
    ]
    conn = _FakeConn()
    n = write_package_downloads(rows, conn)

    assert n == 2
    assert len(conn.calls) == 2
    # params carry the typed columns in order
    assert conn.calls[0][1][:3] == ("anthropic", "pypi", "anthropic")


def test_write_package_downloads_empty_is_noop():
    conn = _FakeConn()
    assert write_package_downloads([], conn) == 0
    assert conn.calls == []
