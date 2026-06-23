"""Unit tests for the tracker graph wiring — that the dossier tail (JAR-56) is
connected behind resolve_board and that the graph compiles.
"""
from __future__ import annotations

from agents.tracker.graph import build_graph, compile_graph


def test_graph_has_resolve_and_dossier_nodes():
    nodes = set(build_graph().nodes)
    assert {"resolve_board", "load_signals", "synthesize_dossier",
            "score_trending", "write_dossier"} <= nodes


def test_graph_compiles_without_checkpointer():
    compile_graph()   # raises if the edges/branches are misconfigured
