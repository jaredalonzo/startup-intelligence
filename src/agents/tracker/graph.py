"""Startup tracker agent — LangGraph graph definition.

Graph contract (from architecture doc), mapped per company:
  resolve_board (LLM + tool-use, cache result)
    → branch: resolved? → END if not
    → fetch_signals (det, parallel)   [JAR-55, not yet wired]
    → snapshot (det)                  [JAR-55, not yet wired]
    → diff (det)                      [JAR-55, not yet wired]
    → load_signals (det, reads store) [JAR-56]
    → branch: meaningful change?      [JAR-56]
    → synthesize_dossier (LLM)        [JAR-56]
    → score_trending (det + LLM flag) [JAR-56]
    → write_dossier (det)             [JAR-56]

resolve_board's conditional currently routes straight into load_signals (which
reads the persisted snapshots/signals); JAR-55's fetch_signals/snapshot/diff
nodes slot in ahead of it once built. resolve_board is async (it probes ATS
endpoints and runs a tool-use loop), so callers must use `ainvoke` / `astream`.
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

from agents.tracker.state import BoardResolution, DossierInputs, TrackerState, TrendScore
from agents.tracker import dossier, nodes


def build_graph() -> StateGraph:
    g = StateGraph(TrackerState)

    g.add_node("resolve_board", nodes.resolve_board)
    g.add_node("load_signals", dossier.load_signals)
    g.add_node("synthesize_dossier", dossier.synthesize_dossier)
    g.add_node("score_trending", dossier.score_trending)
    g.add_node("write_dossier", dossier.write_dossier)

    g.add_edge(START, "resolve_board")
    g.add_conditional_edges(
        "resolve_board",
        nodes.route_after_resolve,
        {
            # JAR-55's fetch_signals/snapshot/diff slot in ahead of load_signals
            # later; for now the post-resolve path enters the dossier tail here.
            "fetch_signals": "load_signals",
            "__end__": END,
        },
    )
    g.add_conditional_edges(
        "load_signals",
        dossier.route_after_signals,
        {
            "synthesize_dossier": "synthesize_dossier",
            "__end__": END,
        },
    )
    g.add_edge("synthesize_dossier", "score_trending")
    g.add_edge("score_trending", "write_dossier")
    g.add_edge("write_dossier", END)

    return g


def compile_graph(checkpointer: PostgresSaver | None = None):
    """Compile the tracker graph, optionally with a Postgres checkpointer."""
    return build_graph().compile(checkpointer=checkpointer)


@contextmanager
def make_checkpointer() -> Iterator[PostgresSaver]:
    """Yield a PostgresSaver with registered state types to suppress msgpack warnings."""
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[BoardResolution, DossierInputs, TrendScore]
    )
    db_url = os.environ["DATABASE_URL"]
    with Connection.connect(
        db_url, autocommit=True, prepare_threshold=0, row_factory=dict_row
    ) as conn:
        yield PostgresSaver(conn, serde=serde)
