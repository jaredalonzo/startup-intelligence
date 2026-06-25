"""Unit tests for the Notion company-dossier upsert writer.

Exercise the create-vs-update branch of upsert_company_dossier with a fake
httpx-like client — no real Notion API call is made.
"""
from __future__ import annotations

import pytest

from outputs.notion import _markdown_to_blocks, _rich_text, upsert_company_dossier

_PARENT = "PARENT"


def _texts(block):
    t = block["type"]
    return "".join(seg["text"]["content"] for seg in block[t]["rich_text"])


def test_inline_markdown_becomes_annotations_not_literal_markup():
    segs = _rich_text("Hiring **up 40%** on `Rust` and *fast*, see [board](https://x.io)")
    # No literal markup leaks into the rendered content.
    assert "**" not in "".join(s["text"]["content"] for s in segs)
    by_text = {s["text"]["content"]: s for s in segs}
    assert by_text["up 40%"]["annotations"] == {"bold": True}
    assert by_text["Rust"]["annotations"] == {"code": True}
    assert by_text["fast"]["annotations"] == {"italic": True}
    assert by_text["board"]["text"]["link"] == {"url": "https://x.io"}


def test_snake_case_identifiers_are_not_italicized():
    [seg] = _rich_text("updated_at drives score_trending")
    assert seg["text"]["content"] == "updated_at drives score_trending"
    assert "annotations" not in seg


def test_bold_wrapped_heading_is_recognized():
    [block] = _markdown_to_blocks("**## Heating Up**")
    assert block["type"] == "heading_2"
    assert _texts(block) == "Heating Up"


def test_headings_lists_and_rules_map_to_block_types():
    md = "# T\n### Detail\n- top\n  - nested\n1. one\n> quote\n---\nplain"
    blocks = _markdown_to_blocks(md)
    types = [b["type"] for b in blocks]
    assert types == [
        "heading_1", "heading_3", "bulleted_list_item",
        "numbered_list_item", "quote", "divider", "paragraph",
    ]
    # The indented bullet nests under its parent rather than leaking as text.
    parent = blocks[2]
    [child] = parent["bulleted_list_item"]["children"]
    assert child["type"] == "bulleted_list_item"
    assert _texts(child) == "nested"


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeNotion:
    """Routes GET/POST/PATCH by URL; toggles whether the dossier page exists."""

    def __init__(self, *, existing: bool) -> None:
        self.existing = existing
        self.posted: list[dict] = []
        self.appended: list[dict] = []
        self.archived: list[str] = []

    def get(self, url, headers=None, params=None):
        if f"blocks/{_PARENT}/children" in url:
            results = (
                [{"type": "child_page", "id": "PAGE1",
                  "child_page": {"title": "Anthropic"}}]
                if self.existing else []
            )
            return _Resp({"results": results, "has_more": False})
        if "blocks/PAGE1/children" in url:
            return _Resp({"results": [{"id": "b1"}, {"id": "b2"}], "has_more": False})
        if "pages/PAGE1" in url:
            return _Resp({"url": "http://notion/existing"})
        return _Resp({"results": [], "has_more": False})

    def post(self, url, headers=None, json=None):
        self.posted.append(json)
        return _Resp({"url": "http://notion/new"})

    def patch(self, url, headers=None, json=None):
        if json == {"archived": True}:
            self.archived.append(url)
        else:
            self.appended.append(json)
        return _Resp({})


@pytest.fixture(autouse=True)
def _notion_env(monkeypatch):
    monkeypatch.setenv("NOTION_TOKEN", "secret")
    monkeypatch.setenv("NOTION_TRACKER_DOSSIERS_PAGE_ID", _PARENT)


def test_creates_page_when_none_exists():
    client = _FakeNotion(existing=False)
    url = upsert_company_dossier("## Summary\n- point", "Anthropic", client=client)

    assert url == "http://notion/new"
    [payload] = client.posted
    title = payload["properties"]["title"]["title"][0]["text"]["content"]
    assert title == "Anthropic"
    assert payload["parent"]["page_id"] == _PARENT
    assert client.archived == []          # nothing to clear on a fresh page


def test_updates_existing_page_in_place():
    client = _FakeNotion(existing=True)
    url = upsert_company_dossier("## Summary\n- point", "Anthropic", client=client)

    assert url == "http://notion/existing"        # URL stays stable across runs
    assert client.posted == []                     # no new page created
    # existing content archived, fresh dossier appended
    assert client.archived == [
        "https://api.notion.com/v1/blocks/b1",
        "https://api.notion.com/v1/blocks/b2",
    ]
    assert len(client.appended) == 1
