"""Unit tests for the Linear output writers.

Exercise the create-vs-update (idempotency) branch of both writers with a fake
httpx-like GraphQL client — no real Linear API call is made.
"""
from __future__ import annotations

import pytest

from outputs.linear import create_gap_tasks, create_top_mover_task


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeLinear:
    """Routes Linear GraphQL ops by operation name; toggles whether a matching
    open issue already exists. Records the create/update inputs for assertions."""

    def __init__(self, *, existing: bool) -> None:
        self.existing = existing
        self.created: list[dict] = []
        self.updated: list[dict] = []
        self._next_id = 70

    def post(self, url, headers=None, json=None):
        query = json["query"]
        variables = json["variables"]
        if "FindIssue" in query:
            nodes = (
                [{"id": "issue-uuid", "identifier": "JAR-1", "url": "http://linear/JAR-1"}]
                if self.existing else []
            )
            return _Resp({"data": {"issues": {"nodes": nodes}}})
        if "CreateIssue" in query:
            self.created.append(variables["input"])
            self._next_id += 1
            ident = f"JAR-{self._next_id}"
            return _Resp({"data": {"issueCreate": {
                "success": True,
                "issue": {"id": "new-uuid", "identifier": ident, "url": f"http://linear/{ident}"},
            }}})
        if "UpdateIssue" in query:
            self.updated.append(variables)
            return _Resp({"data": {"issueUpdate": {
                "success": True,
                "issue": {"id": variables["id"], "identifier": "JAR-1",
                          "url": "http://linear/JAR-1"},
            }}})
        raise AssertionError(f"unexpected query: {query}")


@pytest.fixture(autouse=True)
def _linear_env(monkeypatch):
    monkeypatch.setenv("LINEAR_API_KEY", "lin_api_test")


def test_top_mover_creates_issue_when_none_open():
    client = _FakeLinear(existing=False)
    ident = create_top_mover_task(
        company_name="Anyscale", company_slug="anyscale",
        composite=25.0, classification="accelerating",
        rationale="Lots of releases.", dossier_url="http://notion/anyscale",
        client=client,
    )

    assert ident == "JAR-71"
    assert client.updated == []
    [inp] = client.created
    assert inp["title"] == "[Top mover] Anyscale"
    assert "accelerating" in inp["description"]
    assert "http://notion/anyscale" in inp["description"]
    assert inp["teamId"] and inp["projectId"]


def test_top_mover_updates_existing_issue_in_place():
    client = _FakeLinear(existing=True)
    ident = create_top_mover_task(
        company_name="Anyscale", company_slug="anyscale",
        composite=25.0, classification="accelerating", rationale="Still hot.",
        client=client,
    )

    assert ident == "JAR-1"               # stable identifier across runs
    assert client.created == []           # no duplicate issue
    [upd] = client.updated
    assert upd["id"] == "issue-uuid"
    assert "Still hot." in upd["input"]["description"]


def test_create_gap_tasks_one_per_skill():
    client = _FakeLinear(existing=False)
    idents = create_gap_tasks([("Kubernetes", 0.42), ("Rust", 0.18)], client=client)

    assert len(idents) == 2
    titles = [c["title"] for c in client.created]
    assert titles == ["[Skill gap] Kubernetes", "[Skill gap] Rust"]
    assert "42%" in client.created[0]["description"]
