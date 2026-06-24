"""Notion output writers for the skills agent and the startup tracker."""
from __future__ import annotations

import os
import re
from datetime import date, datetime, timezone
from typing import Any, Iterator

import httpx

_NOTION_API = "https://api.notion.com/v1"
_NOTION_VERSION = "2022-06-28"


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {os.environ['NOTION_TOKEN']}",
        "Notion-Version": _NOTION_VERSION,
        "Content-Type": "application/json",
    }


def _rich_text(content: str) -> list[dict]:  # type: ignore[type-arg]
    """Strip **bold** markers and return a Notion rich_text array."""
    plain = re.sub(r"\*\*(.+?)\*\*", r"\1", content)
    return [{"type": "text", "text": {"content": plain}}]


def _markdown_to_blocks(text: str) -> list[dict]:  # type: ignore[type-arg]
    """Convert the digest markdown (## headings + bullet lists) to Notion blocks."""
    blocks: list[dict] = []  # type: ignore[type-arg]
    for line in text.splitlines():
        line = line.rstrip()
        if line.startswith("## "):
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": _rich_text(line[3:])},
            })
        elif line.startswith(("- ", "* ")):
            blocks.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": _rich_text(line[2:])},
            })
        elif line:
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": _rich_text(line)},
            })
    return blocks


def write_skills_digest(digest: str, run_date: date | None = None) -> str:
    """Create a dated Skills Radar child page in Notion.

    Reads NOTION_TOKEN and NOTION_SKILLS_RADAR_PAGE_ID from the environment.
    Returns the URL of the created page.
    """
    parent_id = os.environ["NOTION_SKILLS_RADAR_PAGE_ID"]
    if run_date is None:
        run_date = datetime.now(timezone.utc).date()

    title = f"Skills Radar — {run_date.isoformat()}"

    payload = {
        "parent": {"type": "page_id", "page_id": parent_id},
        "properties": {
            "title": {"title": [{"type": "text", "text": {"content": title}}]},
        },
        "children": _markdown_to_blocks(digest),
    }

    with httpx.Client() as client:
        resp = client.post(f"{_NOTION_API}/pages", headers=_headers(), json=payload)
        resp.raise_for_status()

    return str(resp.json()["url"])


# ---------------------------------------------------------------------------
# Tracker dossiers — one living page per company, updated in place each run.
# ---------------------------------------------------------------------------

def _iter_child_blocks(client: httpx.Client, parent_id: str) -> Iterator[dict[str, Any]]:
    """Yield every child block of a Notion page, following pagination."""
    cursor: str | None = None
    while True:
        params: dict[str, Any] = {"page_size": 100}
        if cursor:
            params["start_cursor"] = cursor
        resp = client.get(
            f"{_NOTION_API}/blocks/{parent_id}/children",
            headers=_headers(),
            params=params,
        )
        resp.raise_for_status()
        body = resp.json()
        yield from body.get("results", [])
        if not body.get("has_more"):
            return
        cursor = body.get("next_cursor")


def _find_dossier_page(client: httpx.Client, parent_id: str, title: str) -> str | None:
    """Return the page id of an existing child page titled *title*, else None.

    Dossier pages are titled by company name, which is stable across runs, so
    matching on title is what makes the write an upsert rather than an append.
    """
    for block in _iter_child_blocks(client, parent_id):
        if block.get("type") != "child_page":
            continue
        if block.get("child_page", {}).get("title") == title:
            return str(block["id"])
    return None


def _clear_page(client: httpx.Client, page_id: str) -> None:
    """Archive all child blocks of a page so it can be rewritten from scratch.

    Notion has no 'replace content' call; archiving the existing blocks before
    appending the fresh dossier keeps the page id (and its URL) stable while
    leaving no stale content behind.
    """
    for block in list(_iter_child_blocks(client, page_id)):
        resp = client.patch(
            f"{_NOTION_API}/blocks/{block['id']}",
            headers=_headers(),
            json={"archived": True},
        )
        resp.raise_for_status()


def _page_url(client: httpx.Client, page_id: str) -> str:
    resp = client.get(f"{_NOTION_API}/pages/{page_id}", headers=_headers())
    resp.raise_for_status()
    return str(resp.json()["url"])


def upsert_company_dossier(
    dossier: str,
    company_name: str,
    *,
    run_date: date | None = None,
    client: httpx.Client | None = None,
) -> str:
    """Create or update this company's dossier page in Notion, returning its URL.

    One page per company (titled by company name) lives under the dossiers parent
    page named by NOTION_TRACKER_DOSSIERS_PAGE_ID. On a repeat run the existing
    page's content is cleared and rewritten in place, so the URL is stable and
    the page always reflects the latest state. `client` is injectable for tests.
    """
    parent_id = os.environ["NOTION_TRACKER_DOSSIERS_PAGE_ID"]
    if run_date is None:
        run_date = datetime.now(timezone.utc).date()

    blocks = (
        [{
            "object": "block",
            "type": "paragraph",
            "paragraph": {"rich_text": _rich_text(f"Updated {run_date.isoformat()}")},
        }]
        + _markdown_to_blocks(dossier)
    )

    owns_client = client is None
    client = client or httpx.Client()
    try:
        page_id = _find_dossier_page(client, parent_id, company_name)
        if page_id is None:
            resp = client.post(
                f"{_NOTION_API}/pages",
                headers=_headers(),
                json={
                    "parent": {"type": "page_id", "page_id": parent_id},
                    "properties": {
                        "title": {"title": [{"type": "text",
                                             "text": {"content": company_name}}]},
                    },
                    "children": blocks,
                },
            )
            resp.raise_for_status()
            return str(resp.json()["url"])

        # Existing page: clear it, then append the freshly rendered dossier.
        _clear_page(client, page_id)
        resp = client.patch(
            f"{_NOTION_API}/blocks/{page_id}/children",
            headers=_headers(),
            json={"children": blocks},
        )
        resp.raise_for_status()
        return _page_url(client, page_id)
    finally:
        if owns_client:
            client.close()
