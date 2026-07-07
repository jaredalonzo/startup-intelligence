"""One-shot query runner shared by the CLI and the serving surface.

The single place that turns a natural-language question into a final
QueryState: build the initial state, run parse_query → retrieve → answer,
return the result. `scripts/query.py` (CLI) and `serving/app.py` (HTTP)
both call `run_query` so the two surfaces cannot drift.

Read-only by construction — the graph writes nothing to the store.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from agents.query import nodes
from agents.query.graph import compile_graph
from agents.query.state import QueryState

# Compiled once per process. Safe to cache: the graph is stateless (no
# checkpointer) and its nodes resolve their LLM/DB globals at call time,
# so tests that monkeypatch `agents.query.nodes` attributes still work.
_GRAPH: CompiledStateGraph[QueryState] | None = None


def _graph() -> CompiledStateGraph[QueryState]:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = compile_graph()
    return _GRAPH


def initial_state(question: str) -> QueryState:
    return {
        "question": question,
        "plan": None,
        "parse_fallback": False,
        "postings": None,
        "dossiers": None,
        "answer_markdown": None,
    }


async def run_query(
    question: str,
    *,
    corpus: Literal["postings", "dossiers", "both"] | None = None,
    callbacks: Sequence[BaseCallbackHandler] = (),
    run_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> QueryState:
    """Run one question end-to-end and return the final state.

    `run_name` / `metadata` flow into the LangChain run config, so LangSmith
    groups the spans per request (e.g. metadata={"request_id": ...}).

    `corpus` pins the plan's corpus between parse and retrieve — a CLI debug
    affordance. That path runs the node functions directly (same functions
    the graph wires), so `callbacks` / `run_name` / `metadata` do NOT attach
    on it; the serving surface always leaves corpus=None.
    """
    state = initial_state(question)

    if corpus is None:
        config: RunnableConfig = {"callbacks": list(callbacks)}
        if run_name is not None:
            config["run_name"] = run_name
        if metadata is not None:
            config["metadata"] = metadata
        return cast(QueryState, await _graph().ainvoke(state, config=config))

    state.update(await nodes.parse_query(state))  # type: ignore[typeddict-item]
    if state["plan"] is not None:
        state["plan"] = state["plan"].model_copy(update={"corpus": corpus})
    state.update(await nodes.retrieve(state))  # type: ignore[typeddict-item]
    state.update(await nodes.answer(state))  # type: ignore[typeddict-item]
    return state
