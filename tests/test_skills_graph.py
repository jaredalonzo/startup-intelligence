"""Unit tests for the skills agent graph wiring — the routing functions that
control the new-postings short-circuit and the extract_skills fan-out.
"""
from __future__ import annotations

from langgraph.types import Send

from agents.skills.graph import _fan_out_extractions, _route_after_load, build_graph, compile_graph


# ---------------------------------------------------------------------------
# _route_after_load — cost-control short-circuit
# ---------------------------------------------------------------------------

def test_route_ends_when_no_new_postings():
    assert _route_after_load({"new_postings": []}) == "__end__"


def test_route_ends_when_key_missing():
    assert _route_after_load({}) == "__end__"


def test_route_proceeds_when_postings_present():
    assert _route_after_load({"new_postings": [{"id": "a"}]}) == "extract_skills"


# ---------------------------------------------------------------------------
# _fan_out_extractions — one Send per posting into extract_one
# ---------------------------------------------------------------------------

def test_fan_out_emits_one_send_per_posting():
    postings = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    sends = _fan_out_extractions({"new_postings": postings})

    assert len(sends) == 3
    assert all(isinstance(s, Send) for s in sends)
    assert all(s.node == "extract_one" for s in sends)
    assert [s.arg["posting"] for s in sends] == postings


def test_fan_out_empty_is_empty():
    assert _fan_out_extractions({"new_postings": []}) == []


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def test_graph_compiles_with_expected_nodes():
    nodes = set(build_graph().nodes)
    assert {"load_deltas", "extract_skills", "extract_one", "normalize_taxonomy",
            "aggregate_trends", "synthesize_radar", "route_outputs"} <= nodes
    compile_graph()   # compiles without a checkpointer
