"""Watchlist — the universe of companies to monitor.

Holds the seed list and a deterministic ATS resolver that probes the three
public ATS endpoints to discover which board a company uses, then caches the
result in the `companies` table. Probing is a one-time cost per company; all
subsequent runs hit the cache.

Usage:
    async with httpx.AsyncClient() as client:
        with get_connection() as conn:
            results = await seed_companies(conn, client)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx
import psycopg

from ingestion.ats.models import ATSSource

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WatchlistEntry:
    name: str
    slug: str        # canonical key used throughout the system
    ats: ATSSource | None = None       # set if already known; skips probing
    ats_slug: str | None = None        # set when ATS slug differs from slug


# ---------------------------------------------------------------------------
# Seed list — AI / data / infra companies worth tracking
# ---------------------------------------------------------------------------

COMPANIES: list[WatchlistEntry] = [
    # Foundation models
    WatchlistEntry("Anthropic",  "anthropic",   "greenhouse", "anthropic"),
    WatchlistEntry("OpenAI",     "openai",       "ashby",      "openai"),
    WatchlistEntry("Cohere",     "cohere",       "ashby",      "cohere"),
    WatchlistEntry("Mistral AI", "mistral",      "lever",      "mistral"),
    # AI applications
    WatchlistEntry("Perplexity AI", "perplexityai", "ashby",  "perplexity"),
    WatchlistEntry("Character.AI",  "character",    "ashby",  "character"),
    WatchlistEntry("Harvey",        "harvey",       "ashby",  "harvey"),
    WatchlistEntry("Sierra",        "sierra",       "ashby",  "sierra"),
    WatchlistEntry("Cognition",     "cognition",    "greenhouse", "cognitionlabs"),
    WatchlistEntry("Glean",         "glean",        "greenhouse", "gleanwork"),
    # Data / infra
    WatchlistEntry("Scale AI",        "scaleai",    "greenhouse", "scaleai"),
    WatchlistEntry("Hugging Face",    "huggingface", "workable", "huggingface"),
    WatchlistEntry("Anyscale",        "anyscale",   "lever",      "anyscale"),
    WatchlistEntry("Modal",           "modal-labs", "ashby",      "modal"),
    WatchlistEntry("Runway",          "runway",     "ashby",      "runway"),
    # Vector / retrieval
    WatchlistEntry("Pinecone",  "pinecone", "ashby", "pinecone"),
    WatchlistEntry("Weaviate",  "weaviate", "ashby", "weaviate"),
    # Tooling / platforms
    WatchlistEntry("LangChain",   "langchain",  "ashby",      "langchain"),
    WatchlistEntry("Together AI", "togetherai", "greenhouse", "togetherai"),
    # Observability / voice
    WatchlistEntry("Arize AI",    "arize",      "greenhouse", "arizeai"),
    WatchlistEntry("Cartesia AI", "cartesia",   "ashby",      "cartesia"),
]

# ---------------------------------------------------------------------------
# ATS probe — tries all three endpoints concurrently
# ---------------------------------------------------------------------------

_GH = "https://api.greenhouse.io/v1/boards/{slug}/jobs"
_LV = "https://api.lever.co/v0/postings/{slug}?mode=json"
_AB = "https://api.ashbyhq.com/posting-api/job-board/{slug}"
_WK = "https://apply.workable.com/api/v2/accounts/{slug}/jobs"
_WK_BODY = {"query": "", "location": [], "department": [], "worktype": [], "remote": []}

_BOARD_URL: dict[ATSSource, str] = {
    "greenhouse": "https://boards.greenhouse.io/{slug}",
    "lever":      "https://jobs.lever.co/{slug}",
    "ashby":      "https://jobs.ashbyhq.com/{slug}",
    "workable":   "https://apply.workable.com/{slug}/",
}


async def _try_greenhouse(slug: str, client: httpx.AsyncClient) -> tuple[ATSSource, str] | None:
    try:
        r = await client.get(_GH.format(slug=slug), timeout=8.0)
        if r.status_code == 200 and "jobs" in r.json():
            return ("greenhouse", slug)
    except Exception:
        pass
    return None


async def _try_lever(slug: str, client: httpx.AsyncClient) -> tuple[ATSSource, str] | None:
    try:
        r = await client.get(_LV.format(slug=slug), timeout=8.0)
        if r.status_code == 200 and isinstance(r.json(), list):
            return ("lever", slug)
    except Exception:
        pass
    return None


async def _try_ashby(slug: str, client: httpx.AsyncClient) -> tuple[ATSSource, str] | None:
    try:
        r = await client.get(_AB.format(slug=slug), timeout=8.0)
        if r.status_code == 200 and "jobs" in r.json():
            return ("ashby", slug)
    except Exception:
        pass
    return None


async def _try_workable(slug: str, client: httpx.AsyncClient) -> tuple[ATSSource, str] | None:
    try:
        r = await client.post(_WK.format(slug=slug), json=_WK_BODY, timeout=8.0)
        if r.status_code == 200 and "results" in r.json():
            return ("workable", slug)
    except Exception:
        pass
    return None


async def probe_ats(slug: str, client: httpx.AsyncClient) -> tuple[ATSSource, str] | None:
    """Probe all four ATS endpoints concurrently for *slug*.

    Returns ``(ats, ats_slug)`` for the first that responds with a valid board,
    or ``None`` if none match.
    """
    results = await asyncio.gather(
        _try_greenhouse(slug, client),
        _try_lever(slug, client),
        _try_ashby(slug, client),
        _try_workable(slug, client),
    )
    for r in results:
        if r is not None:
            return r
    return None


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_cached(slug: str, conn: psycopg.Connection) -> dict | None:  # type: ignore[type-arg]
    return conn.execute(
        "SELECT ats, ats_slug FROM companies WHERE slug = %s", (slug,)
    ).fetchone()


def upsert_company(
    slug: str,
    name: str,
    ats: ATSSource,
    ats_slug: str,
    conn: psycopg.Connection,  # type: ignore[type-arg]
) -> None:
    board_url = _BOARD_URL[ats].format(slug=ats_slug)
    conn.execute(
        """
        INSERT INTO companies (slug, name, ats, ats_slug, board_url, updated_at)
        VALUES (%s, %s, %s, %s, %s, NOW())
        ON CONFLICT (slug) DO UPDATE
            SET ats        = EXCLUDED.ats,
                ats_slug   = EXCLUDED.ats_slug,
                board_url  = EXCLUDED.board_url,
                updated_at = NOW()
        """,
        (slug, name, ats, ats_slug, board_url),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def resolve_and_cache(
    entry: WatchlistEntry,
    conn: psycopg.Connection,  # type: ignore[type-arg]
    client: httpx.AsyncClient,
) -> tuple[ATSSource, str] | None:
    """Return ``(ats, ats_slug)`` for *entry*, probing and caching if needed."""
    cached = get_cached(entry.slug, conn)
    if cached:
        return (cached["ats"], cached["ats_slug"])

    ats_slug_to_probe = entry.ats_slug or entry.slug

    if entry.ats is not None:
        # ATS already known — skip network probe
        ats, ats_slug = entry.ats, ats_slug_to_probe
    else:
        result = await probe_ats(ats_slug_to_probe, client)
        if result is None:
            return None
        ats, ats_slug = result

    upsert_company(entry.slug, entry.name, ats, ats_slug, conn)
    return (ats, ats_slug)


async def seed_companies(
    conn: psycopg.Connection,  # type: ignore[type-arg]
    client: httpx.AsyncClient,
    companies: list[WatchlistEntry] | None = None,
) -> dict[str, tuple[ATSSource, str] | None]:
    """Resolve and cache every entry in *companies* (defaults to ``COMPANIES``).

    Returns a mapping of ``slug -> (ats, ats_slug) | None``.
    Not found entries are skipped from the DB but included in the return value
    so callers can log / alert on them.
    """
    entries = companies if companies is not None else COMPANIES
    results: dict[str, tuple[ATSSource, str] | None] = {}
    for entry in entries:
        resolved = await resolve_and_cache(entry, conn, client)
        if resolved is not None:
            conn.commit()  # commit per-company so a later failure doesn't roll back prior writes
        results[entry.slug] = resolved
    return results
