"""Graph-level tests for the query head: wiring + a stubbed end-to-end run."""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langchain_core.messages import AIMessage  # noqa: E402

from agents.query import nodes  # noqa: E402
from agents.query.graph import build_graph, compile_graph  # noqa: E402


def test_graph_wiring_is_the_contracted_linear_chain():
    g = build_graph()
    assert set(g.nodes) == {"parse_query", "retrieve", "answer"}
    edges = {(e[0], e[1]) for e in g.edges}
    assert ("__start__", "parse_query") in edges
    assert ("parse_query", "retrieve") in edges
    assert ("retrieve", "answer") in edges
    assert ("answer", "__end__") in edges


def test_compiles_without_checkpointer():
    assert compile_graph() is not None


async def test_end_to_end_with_stubbed_externals(monkeypatch):
    """question in → parsed plan → retrieval → grounded answer out."""

    class _Bound:
        async def ainvoke(self, _messages):
            return AIMessage(content="", tool_calls=[{
                "name": "QueryPlan", "args": {"semantic_terms": "rust"},
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

    class _Conn:
        read_only = False

        def execute(self, sql, _params=None):
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

    @contextmanager
    def _cm():
        yield _Conn()

    monkeypatch.setattr(nodes, "bind_tools", lambda _llm, _tools: _Bound())
    monkeypatch.setattr(nodes, "EMBEDDING_LLM", _Embedder())
    monkeypatch.setattr(nodes, "get_connection", _cm)
    monkeypatch.setattr(nodes, "register_vector", lambda _conn: None)
    monkeypatch.setattr(nodes, "QUERY_ANSWER_LLM", _AnswerLLM())

    result = await compile_graph().ainvoke({
        "question": "who hires rust?", "plan": None, "parse_fallback": False,
        "postings": None, "dossiers": None, "answer_markdown": None,
    })

    assert result["answer_markdown"] == "Acme is hiring Rust engineers [1]."
    assert result["parse_fallback"] is False
    assert result["plan"].semantic_terms == "rust"
    assert len(result["postings"]) == 1 and result["dossiers"] == []


async def test_end_to_end_parse_fallback_still_answers(monkeypatch):
    """A broken parse model must not break the pipeline."""

    class _BrokenBound:
        async def ainvoke(self, _messages):
            raise RuntimeError("no tools today")

    class _Embedder:
        def embed_query(self, _text):
            return [0.0]

    class _Conn:
        read_only = False

        def execute(self, _sql, _params=None):
            class _C:
                @staticmethod
                def fetchall():
                    return []
            return _C()

    @contextmanager
    def _cm():
        yield _Conn()

    monkeypatch.setattr(nodes, "bind_tools", lambda _llm, _tools: _BrokenBound())
    monkeypatch.setattr(nodes, "EMBEDDING_LLM", _Embedder())
    monkeypatch.setattr(nodes, "get_connection", _cm)
    monkeypatch.setattr(nodes, "register_vector", lambda _conn: None)

    result = await compile_graph().ainvoke({
        "question": "obscure question", "plan": None, "parse_fallback": False,
        "postings": None, "dossiers": None, "answer_markdown": None,
    })

    assert result["parse_fallback"] is True
    # empty retrieval → deterministic abstain (no answer LLM configured — proves no call)
    assert result["answer_markdown"] == nodes.ABSTAIN_MARKDOWN
