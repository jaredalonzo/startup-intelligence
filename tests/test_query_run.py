"""Tests for the shared one-shot runner (agents/query/run.py).

Proves the extraction out of scripts/query.py preserved behavior: the graph
path matches the graph e2e result, and the corpus override pins the corpus
between parse and retrieve. Same stubbing pattern as test_query_graph.py.
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langchain_core.messages import AIMessage  # noqa: E402

from agents.query import nodes  # noqa: E402
from agents.query.run import initial_state, run_query  # noqa: E402


class _Bound:
    async def ainvoke(self, _messages):
        return AIMessage(content="", tool_calls=[{
            "name": "QueryPlan", "args": {"semantic_terms": "rust", "corpus": "both"},
            "id": "c0", "type": "tool_call",
        }])


class _Embedder:
    def embed_query(self, _text):
        return [0.0, 1.0]


class _Cursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _RecordingConn:
    """Records executed SQL so tests can assert which corpora were queried."""

    read_only = False

    def __init__(self, log):
        self._log = log

    def execute(self, sql, _params=None):
        self._log.append(sql.lower())
        if "from postings" in sql.lower():
            return _Cursor([{
                "kind": "posting", "company_slug": "acme", "company_name": "Acme",
                "title": "Rust Engineer", "url": "https://a/1", "snippet": "s",
                "distance": 0.2, "posted_at": None, "first_seen_at": None,
                "last_seen_at": None,
            }])
        return _Cursor([])


class _AnswerLLM:
    async def ainvoke(self, _messages):
        return AIMessage(content="Acme is hiring Rust engineers [1].")


def _stub_externals(monkeypatch, sql_log):
    @contextmanager
    def _cm():
        yield _RecordingConn(sql_log)

    monkeypatch.setattr(nodes, "bind_tools", lambda _llm, _tools: _Bound())
    monkeypatch.setattr(nodes, "EMBEDDING_LLM", _Embedder())
    monkeypatch.setattr(nodes, "get_connection", _cm)
    monkeypatch.setattr(nodes, "register_vector", lambda _conn: None)
    monkeypatch.setattr(nodes, "QUERY_ANSWER_LLM", _AnswerLLM())


def test_initial_state_shape():
    state = initial_state("q?")
    assert state == {
        "question": "q?", "plan": None, "parse_fallback": False,
        "postings": None, "dossiers": None, "answer_markdown": None,
    }


async def test_graph_path_matches_e2e_result(monkeypatch):
    """run_query(corpus=None) produces the same result the graph e2e test asserts."""
    _stub_externals(monkeypatch, [])

    result = await run_query("who hires rust?")

    assert result["answer_markdown"] == "Acme is hiring Rust engineers [1]."
    assert result["parse_fallback"] is False
    assert result["plan"].semantic_terms == "rust"
    assert len(result["postings"]) == 1 and result["dossiers"] == []


async def test_callbacks_attach_on_graph_path(monkeypatch):
    """The guard the CLI/server passes must see the LLM calls."""
    from langchain_core.callbacks import BaseCallbackHandler

    class _Counter(BaseCallbackHandler):
        started = 0

        def on_chain_start(self, *_args, **_kwargs):
            type(self).started += 1

    _stub_externals(monkeypatch, [])
    await run_query("who hires rust?", callbacks=[_Counter()])
    assert _Counter.started > 0


async def test_corpus_override_pins_corpus(monkeypatch):
    """corpus='postings' must skip the dossier query even when the plan said 'both'."""
    sql_log: list[str] = []
    _stub_externals(monkeypatch, sql_log)

    result = await run_query("who hires rust?", corpus="postings")

    assert result["plan"].corpus == "postings"
    assert any("from postings" in sql for sql in sql_log)
    assert not any("from dossiers" in sql for sql in sql_log)
    assert result["answer_markdown"] == "Acme is hiring Rust engineers [1]."
