"""Unit tests for the Notion company-dossier upsert writer.

Exercise the create-vs-update branch of upsert_company_dossier with a fake
httpx-like client — no real Notion API call is made.
"""
from __future__ import annotations

import pytest

from outputs.notion import upsert_company_dossier

_PARENT = "PARENT"


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
