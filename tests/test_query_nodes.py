"""Unit tests for the query head's nodes — fake conn, stubbed LLMs, no live backends."""
from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langchain_core.messages import AIMessage  # noqa: E402

from agents.query import nodes  # noqa: E402
from agents.query.nodes import (  # noqa: E402
    ABSTAIN_MARKDOWN,
    answer,
    describe_filters,
    format_context,
    parse_query,
    retrieve,
)
from agents.query.state import QueryPlan, QueryState, RetrievedDoc  # noqa: E402
from config import EMBEDDING_QUERY_PREFIX  # noqa: E402


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

def _state(**over) -> QueryState:
    base: QueryState = {
        "question": "who is hiring rust engineers?",
        "plan": None,
        "parse_fallback": False,
        "postings": None,
        "dossiers": None,
        "answer_markdown": None,
    }
    base.update(over)  # type: ignore[typeddict-item]
    return base


class _FakeBound:
    """bind_tools(...) stand-in: returns a canned AIMessage or raises."""

    def __init__(self, message: AIMessage | None = None, exc: Exception | None = None) -> None:
        self._message = message
        self._exc = exc
        self.received: list = []

    async def ainvoke(self, messages):  # noqa: ANN001 - test double
        self.received.append(messages)
        if self._exc:
            raise self._exc
        return self._message


class _Cursor:
    def __init__(self, rows: list) -> None:
        self._rows = rows

    def fetchall(self) -> list:
        return self._rows


class _DispatchConn:
    """psycopg-like connection returning canned rows keyed by the table queried."""

    def __init__(self, **tables) -> None:
        self.tables = tables  # postings=, dossiers=
        self.read_only = False
        self.executed: list[tuple[str, dict]] = []

    def execute(self, sql, params=None):  # noqa: ANN001 - test double
        self.executed.append((sql, params))
        s = sql.lower()
        assert s.lstrip().startswith(("select", "with")), "query head must only SELECT"
        if "from postings" in s:
            return _Cursor(self.tables.get("postings", []))
        if "from latest_dossiers" in s:
            return _Cursor(self.tables.get("dossiers", []))
        return _Cursor([])


def _patch_retrieval(monkeypatch, conn: _DispatchConn) -> None:
    @contextmanager
    def _cm():
        yield conn

    monkeypatch.setattr(nodes, "get_connection", _cm)
    monkeypatch.setattr(nodes, "register_vector", lambda _conn: None)


class _FakeEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.1, 0.2, 0.3]


class _RecordingLLM:
    def __init__(self, content: str = "Grounded answer [1].") -> None:
        self.calls: list = []
        self._content = content

    async def ainvoke(self, messages):  # noqa: ANN001 - test double
        self.calls.append(messages)
        return AIMessage(content=self._content)


class _ExplodingLLM:
    async def ainvoke(self, messages):  # noqa: ANN001 - test double
        raise AssertionError("LLM must not be invoked on this path")


def _posting_row(title: str = "Rust Engineer") -> dict:
    return {
        "kind": "posting", "company_slug": "acme", "company_name": "Acme",
        "title": title, "url": "https://a.example/j/1", "snippet": "Build systems.",
        "distance": 0.21,
        "posted_at": datetime(2026, 6, 1, tzinfo=timezone.utc),
        "first_seen_at": datetime(2026, 6, 2, tzinfo=timezone.utc),
        "last_seen_at": datetime(2026, 7, 1, tzinfo=timezone.utc),
    }


def _dossier_row() -> dict:
    return {
        "kind": "dossier", "company_slug": "acme", "company_name": "Acme",
        "title": "Dossier — Acme (accelerating)", "url": "https://notion.so/acme",
        "snippet": "# Acme\nSteady growth.", "distance": 0.3,
        "generated_at": datetime(2026, 6, 30, tzinfo=timezone.utc),
    }


# ---------------------------------------------------------------------------
# parse_query
# ---------------------------------------------------------------------------

async def test_parse_query_happy_path(monkeypatch):
    plan_args = {"semantic_terms": "rust engineer", "seniorities": ["senior"]}
    bound = _FakeBound(AIMessage(content="", tool_calls=[
        {"name": "QueryPlan", "args": plan_args, "id": "call_0", "type": "tool_call"}
    ]))
    monkeypatch.setattr(nodes, "bind_tools", lambda _llm, _tools: bound)

    out = await parse_query(_state())

    assert out["parse_fallback"] is False
    assert out["plan"] == QueryPlan(semantic_terms="rust engineer", seniorities=["senior"])


async def test_parse_query_falls_back_on_no_tool_call(monkeypatch):
    bound = _FakeBound(AIMessage(content="I think you want rust jobs."))
    monkeypatch.setattr(nodes, "bind_tools", lambda _llm, _tools: bound)

    out = await parse_query(_state())

    assert out["parse_fallback"] is True
    assert out["plan"] == QueryPlan(semantic_terms="who is hiring rust engineers?")


async def test_parse_query_falls_back_on_invalid_args(monkeypatch):
    bound = _FakeBound(AIMessage(content="", tool_calls=[
        {"name": "QueryPlan", "args": {"corpus": "everything"}, "id": "c0", "type": "tool_call"}
    ]))
    monkeypatch.setattr(nodes, "bind_tools", lambda _llm, _tools: bound)

    out = await parse_query(_state())
    assert out["parse_fallback"] is True


async def test_parse_query_falls_back_on_backend_error(monkeypatch):
    bound = _FakeBound(exc=RuntimeError("ollama down"))
    monkeypatch.setattr(nodes, "bind_tools", lambda _llm, _tools: bound)

    out = await parse_query(_state())
    assert out["parse_fallback"] is True
    assert out["plan"].semantic_terms == "who is hiring rust engineers?"


async def test_parse_query_backfills_empty_semantic_terms(monkeypatch):
    bound = _FakeBound(AIMessage(content="", tool_calls=[
        {"name": "QueryPlan", "args": {"semantic_terms": "  "}, "id": "c0", "type": "tool_call"}
    ]))
    monkeypatch.setattr(nodes, "bind_tools", lambda _llm, _tools: bound)

    out = await parse_query(_state())
    assert out["parse_fallback"] is False
    assert out["plan"].semantic_terms == "who is hiring rust engineers?"


# ---------------------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------------------

async def test_retrieve_both_corpora(monkeypatch):
    conn = _DispatchConn(postings=[_posting_row()], dossiers=[_dossier_row()])
    _patch_retrieval(monkeypatch, conn)
    embedder = _FakeEmbedder()
    monkeypatch.setattr(nodes, "EMBEDDING_LLM", embedder)

    out = await retrieve(_state(plan=QueryPlan(semantic_terms="rust engineer")))

    # the question is embedded with the nomic query prefix
    assert embedder.queries == [f"{EMBEDDING_QUERY_PREFIX}rust engineer"]
    assert conn.read_only is True
    [p] = out["postings"]
    assert isinstance(p, RetrievedDoc) and p.kind == "posting" and p.title == "Rust Engineer"
    [d] = out["dossiers"]
    assert d.kind == "dossier" and d.generated_at is not None
    # both queries carried the bound vector param
    assert all("qvec" in params for _sql, params in conn.executed)


async def test_retrieve_respects_corpus_narrowing(monkeypatch):
    conn = _DispatchConn(postings=[_posting_row()], dossiers=[_dossier_row()])
    _patch_retrieval(monkeypatch, conn)
    monkeypatch.setattr(nodes, "EMBEDDING_LLM", _FakeEmbedder())

    out = await retrieve(
        _state(plan=QueryPlan(semantic_terms="trajectory", corpus="dossiers"))
    )

    assert out["postings"] == [] and len(out["dossiers"]) == 1
    assert len(conn.executed) == 1  # only the dossier query ran


async def test_retrieve_without_plan_uses_raw_question(monkeypatch):
    conn = _DispatchConn()
    _patch_retrieval(monkeypatch, conn)
    embedder = _FakeEmbedder()
    monkeypatch.setattr(nodes, "EMBEDDING_LLM", embedder)

    out = await retrieve(_state(plan=None))

    assert embedder.queries == [f"{EMBEDDING_QUERY_PREFIX}who is hiring rust engineers?"]
    assert out["postings"] == [] and out["dossiers"] == []


# ---------------------------------------------------------------------------
# format_context / answer
# ---------------------------------------------------------------------------

def test_format_context_numbers_continuously_across_sections():
    postings = [RetrievedDoc.model_validate(_posting_row("A")),
                RetrievedDoc.model_validate(_posting_row("B"))]
    dossiers = [RetrievedDoc.model_validate(_dossier_row())]

    ctx = format_context(postings, dossiers)

    assert "### Job postings" in ctx and "### Company dossiers" in ctx
    assert "[1] Acme — A" in ctx and "[2] Acme — B" in ctx
    assert "[3] Dossier — Acme (accelerating)" in ctx  # numbering continues
    assert "URL: https://a.example/j/1" in ctx
    assert "Posted: 2026-06-01 | First seen: 2026-06-02 | Last confirmed live: 2026-07-01" in ctx
    assert "Generated: 2026-06-30" in ctx


def test_format_context_tolerates_missing_url_and_dates():
    doc = RetrievedDoc(kind="posting", company_slug="a", company_name="A",
                       title="T", url=None, snippet="s", distance=0.5)
    ctx = format_context([doc], [])
    assert "URL: none" in ctx and "Posted: unknown" in ctx


def test_describe_filters_empty_for_semantic_only_plan():
    assert describe_filters(None) == ""
    assert describe_filters(QueryPlan(semantic_terms="rust")) == ""
    # corpus narrowing alone is not a document-level fact worth asserting
    assert describe_filters(QueryPlan(semantic_terms="x", corpus="postings")) == ""


def test_describe_filters_renders_enforced_constraints():
    note = describe_filters(QueryPlan(
        semantic_terms="x", skills=["Kubernetes"], growing_eng=True, companies=["acme"],
    ))
    assert note.startswith("Retrieval filters applied")
    assert "extracted skills include: Kubernetes" in note
    assert "growing engineering headcount" in note
    assert "restricted to companies: acme" in note


async def test_answer_context_includes_filters_note(monkeypatch):
    llm = _RecordingLLM()
    monkeypatch.setattr(nodes, "QUERY_ANSWER_LLM", llm)
    postings = [RetrievedDoc.model_validate(_posting_row())]
    plan = QueryPlan(semantic_terms="k8s", skills=["Kubernetes"], growing_eng=True)

    await answer(_state(plan=plan, postings=postings, dossiers=[]))

    [messages] = llm.calls
    human = messages[1].content
    assert "Retrieval filters applied" in human
    assert human.index("Retrieval filters applied") < human.index("[1] Acme")


async def test_answer_abstains_without_llm_on_empty_retrieval(monkeypatch):
    monkeypatch.setattr(nodes, "QUERY_ANSWER_LLM", _ExplodingLLM())

    out = await answer(_state(postings=[], dossiers=[]))

    assert out["answer_markdown"] == ABSTAIN_MARKDOWN


async def test_answer_grounds_on_context(monkeypatch):
    llm = _RecordingLLM()
    monkeypatch.setattr(nodes, "QUERY_ANSWER_LLM", llm)
    postings = [RetrievedDoc.model_validate(_posting_row())]

    out = await answer(_state(postings=postings, dossiers=[]))

    assert out["answer_markdown"] == "Grounded answer [1]."
    [messages] = llm.calls
    human = messages[1].content
    assert "who is hiring rust engineers?" in human
    assert "[1] Acme — Rust Engineer" in human      # the context reached the model
    system = messages[0].content
    assert "cite" in system.lower() and "## Sources" in system
