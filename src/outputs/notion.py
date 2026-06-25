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


_TEXT_LIMIT = 2000  # Notion rejects a single rich_text content longer than this.

# Inline spans, tried in this order so `***x***` beats `**x**` beats `*x*`.
# Only `*` is treated as emphasis — `_` is left alone so snake_case identifiers
# (updated_at, score_trending, ...) survive untouched.
_INLINE_RE = re.compile(
    r"\*\*\*(?P<bi>.+?)\*\*\*"
    r"|\*\*(?P<b>.+?)\*\*"
    r"|(?<!\*)\*(?!\*)(?P<i>[^*]+?)\*(?!\*)"
    r"|`(?P<c>[^`]+?)`"
    r"|\[(?P<lt>[^\]]+?)\]\((?P<lh>[^)]+?)\)"
)


def _segment(content: str, *, bold: bool = False, italic: bool = False,
             code: bool = False, href: str | None = None) -> Iterator[dict]:  # type: ignore[type-arg]
    """Yield Notion text segment(s), splitting on the 2000-char content limit."""
    annotations: dict[str, bool] = {}
    if bold:
        annotations["bold"] = True
    if italic:
        annotations["italic"] = True
    if code:
        annotations["code"] = True
    for i in range(0, len(content), _TEXT_LIMIT):
        chunk = content[i:i + _TEXT_LIMIT]
        text: dict[str, Any] = {"content": chunk}
        if href:
            text["link"] = {"url": href}
        seg: dict[str, Any] = {"type": "text", "text": text}
        if annotations:
            seg["annotations"] = dict(annotations)
        yield seg


def _rich_text(content: str) -> list[dict]:  # type: ignore[type-arg]
    """Parse inline markdown (bold/italic/code/links) into a Notion rich_text array."""
    segments: list[dict] = []  # type: ignore[type-arg]
    pos = 0
    for m in _INLINE_RE.finditer(content):
        if m.start() > pos:
            segments.extend(_segment(content[pos:m.start()]))
        if m.group("bi") is not None:
            segments.extend(_segment(m.group("bi"), bold=True, italic=True))
        elif m.group("b") is not None:
            segments.extend(_segment(m.group("b"), bold=True))
        elif m.group("i") is not None:
            segments.extend(_segment(m.group("i"), italic=True))
        elif m.group("c") is not None:
            segments.extend(_segment(m.group("c"), code=True))
        else:  # link
            segments.extend(_segment(m.group("lt"), href=m.group("lh")))
        pos = m.end()
    if pos < len(content):
        segments.extend(_segment(content[pos:]))
    return segments or list(_segment(content))


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_NUMBER_RE = re.compile(r"^(\s*)\d+[.)]\s+(.*)$")
_QUOTE_RE = re.compile(r"^>\s?(.*)$")
_DIVIDER_RE = re.compile(r"^(-{3,}|\*{3,}|_{3,})$")


def _leaf(block_type: str, text: str) -> dict:  # type: ignore[type-arg]
    return {"object": "block", "type": block_type,
            block_type: {"rich_text": _rich_text(text)}}


def _markdown_to_blocks(text: str) -> list[dict]:  # type: ignore[type-arg]
    """Convert digest/dossier markdown to Notion blocks.

    Handles headings (#–######, clamped to Notion's three levels), ordered and
    unordered lists with indentation-based nesting, block quotes, horizontal
    rules, and paragraphs. Inline bold/italic/code/links are rendered via
    _rich_text rather than left as literal markup.
    """
    root: list[dict] = []  # type: ignore[type-arg]
    # Stack of (indent, list_item_block) for nesting child list items.
    stack: list[tuple[int, dict]] = []  # type: ignore[type-arg]

    def append_list_item(indent: int, block: dict) -> None:  # type: ignore[type-arg]
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if stack:
            parent = stack[-1][1]
            parent[parent["type"]].setdefault("children", []).append(block)
        else:
            root.append(block)
        stack.append((indent, block))

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue

        stripped = line.strip()
        if _DIVIDER_RE.match(stripped):
            root.append({"object": "block", "type": "divider", "divider": {}})
            stack.clear()
            continue

        # Some models wrap the whole heading line in bold (`**## Heating Up**`);
        # unwrap a full-line bold span before testing so it still becomes a heading.
        heading_line = line
        wrapped = re.match(r"^\*\*(.+)\*\*$", stripped)
        if wrapped and _HEADING_RE.match(wrapped.group(1)):
            heading_line = wrapped.group(1)
        heading = _HEADING_RE.match(heading_line)
        if heading:
            level = min(len(heading.group(1)), 3)
            root.append(_leaf(f"heading_{level}", heading.group(2)))
            stack.clear()
            continue

        bullet = _BULLET_RE.match(line)
        number = None if bullet else _NUMBER_RE.match(line)
        if bullet or number:
            match = bullet or number
            assert match is not None
            indent = len(match.group(1).replace("\t", "  "))
            block_type = "bulleted_list_item" if bullet else "numbered_list_item"
            append_list_item(indent, _leaf(block_type, match.group(2)))
            continue

        quote = _QUOTE_RE.match(line)
        if quote:
            root.append(_leaf("quote", quote.group(1)))
            stack.clear()
            continue

        root.append(_leaf("paragraph", stripped))
        stack.clear()
    return root


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
