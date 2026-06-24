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
from typing import Any

import httpx
import psycopg
from psycopg.types.json import Jsonb

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
    github_org: str | None = None      # GitHub org slug, e.g. 'anthropics'
    blog_url: str | None = None        # blog/news page URL
    blog_rss_url: str | None = None    # RSS feed URL, null if not available
    # Published packages whose download trends proxy adoption.
    # (registry, package) pairs; registry is 'pypi' or 'npm'.
    packages: tuple[tuple[str, str], ...] = ()


# ---------------------------------------------------------------------------
# Seed list — AI / data / infra companies worth tracking
# ---------------------------------------------------------------------------

COMPANIES: list[WatchlistEntry] = [
    # Foundation models
    WatchlistEntry("Anthropic",  "anthropic",   "greenhouse", "anthropic",
                   github_org="anthropics",
                   blog_url="https://www.anthropic.com/blog",
                   packages=(("pypi", "anthropic"), ("npm", "@anthropic-ai/sdk"))),
    WatchlistEntry("OpenAI",     "openai",       "ashby",      "openai",
                   github_org="openai",
                   blog_rss_url="https://openai.com/blog/rss.xml",
                   packages=(("pypi", "openai"), ("npm", "openai"))),
    WatchlistEntry("Cohere",     "cohere",       "ashby",      "cohere",
                   github_org="cohere-ai",
                   blog_url="https://cohere.com/blog",
                   packages=(("pypi", "cohere"), ("npm", "cohere-ai"))),
    WatchlistEntry("Mistral AI", "mistral",      "lever",      "mistral",
                   github_org="mistralai",
                   blog_url="https://mistral.ai/news",
                   blog_rss_url="https://mistral.ai/rss.xml",
                   packages=(("pypi", "mistralai"), ("npm", "@mistralai/mistralai"))),
    # AI applications
    WatchlistEntry("Perplexity AI", "perplexityai", "ashby",  "perplexity",
                   github_org="perplexity-ai"),
    WatchlistEntry("Character.AI",  "character",    "ashby",  "character",
                   github_org="character-ai"),
    WatchlistEntry("Harvey",        "harvey",       "ashby",  "harvey",
                   blog_url="https://www.harvey.ai/blog"),
    WatchlistEntry("Sierra",        "sierra",       "ashby",  "sierra",
                   blog_url="https://sierra.ai/blog",
                   blog_rss_url="https://sierra.ai/rss.xml"),
    WatchlistEntry("Cognition",     "cognition",    "greenhouse", "cognitionlabs",
                   github_org="cognition-ai",
                   blog_url="https://www.cognition.ai/blog"),
    WatchlistEntry("Glean",         "glean",        "greenhouse", "gleanwork",
                   github_org="gleanwork",
                   blog_url="https://www.glean.com/blog"),
    # Data / infra
    WatchlistEntry("Scale AI",        "scaleai",    "greenhouse", "scaleai",
                   github_org="scaleapi",
                   blog_url="https://scale.com/blog",
                   packages=(("pypi", "scaleapi"),)),
    WatchlistEntry("Hugging Face",    "huggingface", "workable", "huggingface",
                   github_org="huggingface",
                   blog_url="https://huggingface.co/blog",
                   blog_rss_url="https://huggingface.co/blog/feed.xml",
                   packages=(("pypi", "transformers"), ("pypi", "huggingface-hub"),
                             ("pypi", "datasets"), ("npm", "@huggingface/inference"))),
    WatchlistEntry("Anyscale",        "anyscale",   "lever",      "anyscale",
                   github_org="ray-project",
                   blog_url="https://www.anyscale.com/blog",
                   blog_rss_url="https://www.anyscale.com/rss.xml",
                   packages=(("pypi", "ray"),)),
    WatchlistEntry("Modal",           "modal-labs", "ashby",      "modal",
                   github_org="modal-labs",
                   blog_url="https://modal.com/blog",
                   blog_rss_url="https://modal.com/blog/atom.xml",
                   packages=(("pypi", "modal"),)),
    WatchlistEntry("Runway",          "runway",     "ashby",      "runway",
                   github_org="runwayml",
                   blog_url="https://runwayml.com/blog"),
    # Vector / retrieval
    WatchlistEntry("Pinecone",  "pinecone", "ashby", "pinecone",
                   github_org="pinecone-io",
                   blog_url="https://www.pinecone.io/blog",
                   blog_rss_url="https://www.pinecone.io/rss",
                   packages=(("pypi", "pinecone-client"),
                             ("npm", "@pinecone-database/pinecone"))),
    WatchlistEntry("Weaviate",  "weaviate", "ashby", "weaviate",
                   github_org="weaviate",
                   blog_url="https://weaviate.io/blog",
                   blog_rss_url="https://weaviate.io/blog/rss.xml",
                   packages=(("pypi", "weaviate-client"), ("npm", "weaviate-ts-client"))),
    # Tooling / platforms
    WatchlistEntry("LangChain",   "langchain",  "ashby",      "langchain",
                   github_org="langchain-ai",
                   blog_url="https://www.langchain.com/blog",
                   blog_rss_url="https://www.langchain.com/blog/rss.xml",
                   packages=(("pypi", "langchain"), ("pypi", "langchain-core"),
                             ("npm", "langchain"))),
    WatchlistEntry("Together AI", "togetherai", "greenhouse", "togetherai",
                   github_org="togethercomputer",
                   blog_url="https://www.together.ai/blog",
                   blog_rss_url="https://www.together.ai/blog/rss.xml",
                   packages=(("pypi", "together"),)),
    # Observability / voice
    WatchlistEntry("Arize AI",    "arize",      "greenhouse", "arizeai",
                   github_org="Arize-ai",
                   blog_url="https://arize.com/blog",
                   blog_rss_url="https://arize.com/feed/",
                   packages=(("pypi", "arize"), ("pypi", "arize-phoenix"))),
    WatchlistEntry("Cartesia AI", "cartesia",   "ashby",      "cartesia",
                   github_org="cartesia-ai",
                   blog_url="https://cartesia.ai/blog",
                   packages=(("pypi", "cartesia"), ("npm", "@cartesia/cartesia-js"))),
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

def get_cached(slug: str, conn: psycopg.Connection[dict[str, Any]]) -> dict[str, Any] | None:
    return conn.execute(
        "SELECT ats, ats_slug FROM companies WHERE slug = %s", (slug,)
    ).fetchone()


def upsert_company(
    slug: str,
    name: str,
    ats: ATSSource,
    ats_slug: str,
    conn: psycopg.Connection[dict[str, Any]],
    github_org: str | None = None,
    blog_url: str | None = None,
    blog_rss_url: str | None = None,
    packages: tuple[tuple[str, str], ...] = (),
) -> None:
    board_url = _BOARD_URL[ats].format(slug=ats_slug)
    packages_json = [{"registry": r, "package": p} for r, p in packages]
    conn.execute(
        """
        INSERT INTO companies (slug, name, ats, ats_slug, board_url,
                               github_org, blog_url, blog_rss_url, packages, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (slug) DO UPDATE
            SET ats          = EXCLUDED.ats,
                ats_slug     = EXCLUDED.ats_slug,
                board_url    = EXCLUDED.board_url,
                github_org   = EXCLUDED.github_org,
                blog_url     = EXCLUDED.blog_url,
                blog_rss_url = EXCLUDED.blog_rss_url,
                packages     = EXCLUDED.packages,
                updated_at   = NOW()
        """,
        (slug, name, ats, ats_slug, board_url, github_org, blog_url, blog_rss_url,
         Jsonb(packages_json)),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def resolve_and_cache(
    entry: WatchlistEntry,
    conn: psycopg.Connection[dict[str, Any]],
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

    upsert_company(entry.slug, entry.name, ats, ats_slug, conn,
                   github_org=entry.github_org,
                   blog_url=entry.blog_url,
                   blog_rss_url=entry.blog_rss_url,
                   packages=entry.packages)
    return (ats, ats_slug)


async def seed_companies(
    conn: psycopg.Connection[dict[str, Any]],
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
