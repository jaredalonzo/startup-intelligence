"""Startup tracker agent — LangGraph graph definition.

Graph contract (from architecture doc), mapped per company:
  resolve_board (LLM + tool-use, cache result)
    → branch: resolved? → END if not
    → fetch_signals (det, parallel)   [JAR-55]
    → snapshot (det)                  [JAR-55]
    → diff (det)                      [JAR-55]
    → branch: meaningful change?      [JAR-56]
    → synthesize_dossier (LLM)        [JAR-56]
    → score_trending                  [JAR-56]

Only resolve_board and its conditional skip are wired so far; the remaining
nodes land in JAR-55 / JAR-56. The node is async (it probes ATS endpoints and
runs a tool-use loop), so callers must use `ainvoke` / `astream`.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from psycopg import Connection
from psycopg.rows import dict_row

from agents.tracker.state import BoardResolution, TrackerState
from agents.tracker import nodes


def build_graph() -> StateGraph:
    g = StateGraph(TrackerState)

    g.add_node("resolve_board", nodes.resolve_board)

    g.add_edge(START, "resolve_board")
    g.add_conditional_edges(
        "resolve_board",
        nodes.route_after_resolve,
        {
            # TODO(JAR-55): replace END with the "fetch_signals" node once it exists.
            "fetch_signals": END,
            "__end__": END,
        },
    )

    return g


def compile_graph(checkpointer: PostgresSaver | None = None):
    """Compile the tracker graph, optionally with a Postgres checkpointer."""
    return build_graph().compile(checkpointer=checkpointer)


@contextmanager
def make_checkpointer() -> Iterator[PostgresSaver]:
    """Yield a PostgresSaver with registered state types to suppress msgpack warnings."""
    serde = JsonPlusSerializer(allowed_msgpack_modules=[BoardResolution])
    db_url = os.environ["DATABASE_URL"]
    with Connection.connect(
        db_url, autocommit=True, prepare_threshold=0, row_factory=dict_row
    ) as conn:
        yield PostgresSaver(conn, serde=serde)
