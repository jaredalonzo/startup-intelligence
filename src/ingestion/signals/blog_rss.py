"""Blog/changelog RSS feed ingestion.

Fetches RSS/Atom feeds for companies with blog_rss_url set, parses entries,
and inserts new posts into blog_posts (deduped by company_slug + url).

Pure deterministic ingestion; no LLM calls.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import feedparser
import httpx
import psycopg
from psycopg.types.json import Jsonb
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_MAX_POSTS = 50        # max entries read per feed
_SUMMARY_MAX = 500     # chars kept from feed summary/description field


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class BlogPost(BaseModel):
    company_slug: str
    url: str
    title: str
    published_at: datetime | None
    summary: str | None
    raw: dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_published(entry: Any) -> datetime | None:
    for attr in ("published_parsed", "updated_parsed"):
        t = getattr(entry, attr, None)
        if t is not None:
            try:
                return datetime(*t[:6], tzinfo=timezone.utc)
            except Exception:
                pass
    return None


def _entry_raw(entry: Any) -> dict[str, Any]:
    """Produce a JSON-serializable dict from a feedparser entry."""
    return {
        "id":        getattr(entry, "id", None),
        "title":     getattr(entry, "title", None),
        "link":      getattr(entry, "link", None),
        "summary":   getattr(entry, "summary", None),
        "published": getattr(entry, "published", None),
        "updated":   getattr(entry, "updated", None),
        "author":    getattr(entry, "author", None),
        "tags":      [t.get("term") for t in getattr(entry, "tags", []) if isinstance(t, dict)],
    }


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

async def fetch_blog_posts(
    company_slug: str,
    rss_url: str,
    client: httpx.AsyncClient,
    max_posts: int = _MAX_POSTS,
) -> list[BlogPost]:
    """Fetch and parse *rss_url*. Returns a list of BlogPost objects."""
    try:
        resp = await client.get(
            rss_url,
            headers={"User-Agent": "startup-intelligence/1.0 (feed reader)"},
            timeout=15.0,
        )
        resp.raise_for_status()
    except Exception:
        logger.warning("%s: RSS fetch failed for %s", company_slug, rss_url, exc_info=True)
        return []

    feed = feedparser.parse(resp.content)
    if feed.bozo and not feed.entries:
        logger.warning(
            "%s: RSS parse error for %s: %s",
            company_slug, rss_url, feed.bozo_exception,
        )
        return []

    posts: list[BlogPost] = []
    for entry in feed.entries[:max_posts]:
        url = getattr(entry, "link", None)
        title = getattr(entry, "title", None)
        if not url or not title:
            continue
        raw_summary = getattr(entry, "summary", None)
        summary = raw_summary[:_SUMMARY_MAX] if raw_summary else None
        posts.append(BlogPost(
            company_slug=company_slug,
            url=url,
            title=title,
            published_at=_parse_published(entry),
            summary=summary,
            raw=_entry_raw(entry),
        ))

    logger.info("%s: %d posts parsed from %s", company_slug, len(posts), rss_url)
    return posts


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------

def write_blog_posts(
    posts: list[BlogPost],
    conn: psycopg.Connection,  # type: ignore[type-arg]
) -> int:
    """Insert new posts, skipping duplicates. Returns the count of new rows."""
    new_count = 0
    for p in posts:
        result = conn.execute(
            """
            INSERT INTO blog_posts (company_slug, url, title, published_at, summary, raw)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (company_slug, url) DO NOTHING
            """,
            (p.company_slug, p.url, p.title, p.published_at, p.summary, Jsonb(p.raw)),
        )
        new_count += result.rowcount
    return new_count
