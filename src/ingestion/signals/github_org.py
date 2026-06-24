"""GitHub org signal fetcher — releases and repo stats.

Fetches the top public repos for a GitHub org by star count, then collects:
  - All releases per repo  → github_releases (deduped by tag)
  - Star/fork/issue counts → github_repo_stats (time-series snapshot)

Pure deterministic ingestion; no LLM calls.
Reads GITHUB_TOKEN from env if present (unauthenticated: 60 req/hr limit).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

import httpx
import psycopg
from psycopg.types.json import Jsonb
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_GITHUB_API = "https://api.github.com"
_FETCH_REPOS = 50   # repos fetched from GitHub (sorted client-side by stars)
_MAX_REPOS = 5      # top repos kept after sorting
_MAX_RELEASES = 20  # releases fetched per repo


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class GithubRelease(BaseModel):
    company_slug: str
    repo: str               # full name, e.g. 'openai/openai-python'
    release_tag: str
    release_name: str | None
    published_at: datetime | None
    body: str | None
    raw: dict[str, Any]


class GithubRepoSnapshot(BaseModel):
    company_slug: str
    repo: str
    star_count: int
    fork_count: int
    open_issues: int


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def fetch_github_signals(
    company_slug: str,
    github_org: str,
    client: httpx.AsyncClient,
    max_repos: int = _MAX_REPOS,
) -> tuple[list[GithubRelease], list[GithubRepoSnapshot]]:
    """Return (releases, repo_snapshots) for the top public repos of *github_org*."""
    resp = await client.get(
        f"{_GITHUB_API}/orgs/{github_org}/repos",
        headers=_headers(),
        params={"type": "public", "sort": "pushed", "per_page": _FETCH_REPOS},
        timeout=10.0,
    )
    if resp.status_code != 200:
        logger.warning(
            "GitHub repos fetch failed for %s (%s): HTTP %s",
            company_slug, github_org, resp.status_code,
        )
        return [], []

    # Sort client-side by stars and take the top N
    all_repos = sorted(resp.json(), key=lambda r: r.get("stargazers_count", 0), reverse=True)
    top_repos = all_repos[:max_repos]

    releases: list[GithubRelease] = []
    snapshots: list[GithubRepoSnapshot] = []

    for repo in top_repos:
        repo_full = repo["full_name"]

        snapshots.append(GithubRepoSnapshot(
            company_slug=company_slug,
            repo=repo_full,
            star_count=repo.get("stargazers_count", 0),
            fork_count=repo.get("forks_count", 0),
            open_issues=repo.get("open_issues_count", 0),
        ))

        rel_resp = await client.get(
            f"{_GITHUB_API}/repos/{repo_full}/releases",
            headers=_headers(),
            params={"per_page": _MAX_RELEASES},
            timeout=10.0,
        )
        if rel_resp.status_code != 200:
            continue

        for rel in rel_resp.json():
            published_at: datetime | None = None
            if rel.get("published_at"):
                published_at = datetime.fromisoformat(
                    rel["published_at"].replace("Z", "+00:00")
                )
            releases.append(GithubRelease(
                company_slug=company_slug,
                repo=repo_full,
                release_tag=rel["tag_name"],
                release_name=rel.get("name") or None,
                published_at=published_at,
                body=rel.get("body") or None,
                raw=rel,
            ))

    logger.info(
        "%s (%s): %d repos, %d releases",
        company_slug, github_org, len(snapshots), len(releases),
    )
    return releases, snapshots


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------

def write_github_signals(
    releases: list[GithubRelease],
    snapshots: list[GithubRepoSnapshot],
    conn: psycopg.Connection[dict[str, Any]],
) -> tuple[int, int]:
    """Insert releases (skip duplicates) and repo stat snapshots.

    Returns (new_release_count, snapshot_count).
    """
    new_releases = 0
    for r in releases:
        result = conn.execute(
            """
            INSERT INTO github_releases
                (company_slug, repo, release_tag, release_name, published_at, body, raw)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (repo, release_tag) DO NOTHING
            """,
            (
                r.company_slug, r.repo, r.release_tag, r.release_name,
                r.published_at, r.body, Jsonb(r.raw),
            ),
        )
        new_releases += result.rowcount

    for s in snapshots:
        conn.execute(
            """
            INSERT INTO github_repo_stats (company_slug, repo, star_count, fork_count, open_issues)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (s.company_slug, s.repo, s.star_count, s.fork_count, s.open_issues),
        )

    return new_releases, len(snapshots)
