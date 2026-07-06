"""Query head (RAG) — LangGraph graph definition.

Graph contract (CLAUDE.md):
  parse_query (LLM tool-use: question -> {semantic_terms, structured_filters})
    → retrieve (det: hybrid pgvector cosine + typed-column/snapshot-delta SQL)
    → answer (LLM: grounded synthesis, cite-or-abstain)

Linear — the abstain path (empty retrieval) and the parse-failure fallback are
handled inside their nodes, so no conditional edges are needed. Compiled
WITHOUT a checkpointer by default: a query is a one-shot, read-only request
with no accumulating channels and no resume semantics (tracker/graph.py keeps
the make_checkpointer pattern if that ever changes). parse_query and answer
are async, so callers must use `ainvoke` / `astream`.
"""
from __future__ import annotations

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.query import nodes
from agents.query.state import QueryState


def build_graph() -> StateGraph[QueryState]:
    g = StateGraph(QueryState)

    g.add_node("parse_query", nodes.parse_query)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("answer", nodes.answer)

    g.add_edge(START, "parse_query")
    g.add_edge("parse_query", "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", END)

    return g


def compile_graph(checkpointer: PostgresSaver | None = None) -> CompiledStateGraph[QueryState]:
    """Compile the query graph; checkpointer kept for signature parity, unused by the CLI."""
    return build_graph().compile(checkpointer=checkpointer)
