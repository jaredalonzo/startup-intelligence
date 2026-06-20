"""Skills trend agent — LangGraph graph definition.

Graph contract (from architecture doc):
  load_deltas (det)
    → branch: new postings? → END if none
    → extract_skills map (LLM, Send fan-out, operator.add reducer)
    → normalize_taxonomy (det)
    → aggregate_trends (det)
    → synthesize_radar (LLM)
    → route_outputs (det)

Compiled with a Postgres checkpointer for resumability and run history.
"""
from __future__ import annotations

import os
from typing import Literal

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents.state import SkillExtraction, SkillsState
from agents.skills import nodes

# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def _route_after_load(state: SkillsState) -> Literal["extract_skills", "__end__"]:
    """Short-circuit to END when there are no new postings (cost control)."""
    if not state.get("new_postings"):
        return "__end__"
    return "extract_skills"


def _fan_out_extractions(state: SkillsState) -> list[Send]:
    """Emit one Send per posting so extract_skills runs in parallel."""
    return [
        Send("extract_one", {"posting": p})
        for p in state["new_postings"]
    ]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    g = StateGraph(SkillsState)

    g.add_node("load_deltas",        nodes.load_deltas)
    g.add_node("extract_skills",     nodes.extract_skills)   # fan-out coordinator
    g.add_node("extract_one",        nodes.extract_one)      # per-posting LLM node
    g.add_node("normalize_taxonomy", nodes.normalize_taxonomy)
    g.add_node("aggregate_trends",   nodes.aggregate_trends)
    g.add_node("synthesize_radar",   nodes.synthesize_radar)
    g.add_node("route_outputs",      nodes.route_outputs)

    g.add_edge(START, "load_deltas")
    g.add_conditional_edges("load_deltas", _route_after_load)

    # extract_skills fans out via Send; extract_one results accumulate via operator.add
    g.add_conditional_edges("extract_skills", _fan_out_extractions, ["extract_one"])
    g.add_edge("extract_one",        "normalize_taxonomy")
    g.add_edge("normalize_taxonomy", "aggregate_trends")
    g.add_edge("aggregate_trends",   "synthesize_radar")
    g.add_edge("synthesize_radar",   "route_outputs")
    g.add_edge("route_outputs",      END)

    return g


def compile_graph(checkpointer: PostgresSaver | None = None):
    """Compile the skills graph, optionally with a Postgres checkpointer."""
    g = build_graph()
    return g.compile(checkpointer=checkpointer)


def make_checkpointer() -> PostgresSaver:
    """Return a PostgresSaver backed by DATABASE_URL."""
    db_url = os.environ["DATABASE_URL"]
    # psycopg3 expects the postgresql:// scheme; PostgresSaver accepts it directly.
    return PostgresSaver.from_conn_string(db_url)
