"""Notion output writers for the skills agent."""
from __future__ import annotations

import os
import re
from datetime import date, datetime, timezone

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
