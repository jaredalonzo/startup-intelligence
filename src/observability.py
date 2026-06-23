"""Observability — LangSmith tracing for the two LangGraph agents.

CLAUDE.md requires every LLM node to emit a trace span. The agents are built on
LangChain/LangGraph, so LangSmith does this automatically once tracing env vars
are set before the graphs run: each node becomes a run and each LLM call (incl.
the local Ollama models, which are still LangChain chat models) a child span.

`init_tracing()` is called by the run entrypoints. It is a no-op unless
LANGSMITH_API_KEY is present, so local runs without a key just skip tracing
rather than erroring.
"""
from __future__ import annotations

import logging
import os

from config import LANGSMITH_PROJECT

logger = logging.getLogger(__name__)


def init_tracing(project: str | None = None) -> bool:
    """Enable LangSmith tracing if LANGSMITH_API_KEY is set. Returns whether it
    was enabled. Safe and idempotent to call once at process startup."""
    if not os.environ.get("LANGSMITH_API_KEY"):
        logger.info("Observability: LANGSMITH_API_KEY not set — tracing disabled.")
        return False

    project = project or LANGSMITH_PROJECT
    # Set both the legacy (LANGCHAIN_*) and current (LANGSMITH_*) flags so tracing
    # is picked up regardless of the installed langchain/langsmith versions.
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ["LANGCHAIN_PROJECT"] = project
    os.environ["LANGSMITH_PROJECT"] = project
    logger.info("Observability: LangSmith tracing enabled (project=%s).", project)
    return True
