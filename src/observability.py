"""Observability — LangSmith tracing and a per-run LLM cost guard.

CLAUDE.md requires every LLM node to emit a trace span. The agents are built on
LangChain/LangGraph, so LangSmith does this automatically once tracing env vars
are set before the graphs run: each node becomes a run and each LLM call (incl.
the local Ollama models, which are still LangChain chat models) a child span.

`init_tracing()` is called by the run entrypoints. It is a no-op unless
LANGSMITH_API_KEY is present, so local runs without a key just skip tracing
rather than erroring.

`CostGuard` is a LangChain callback the entrypoints attach to graph runs. It
tracks model invocations and (when the provider reports them) tokens per run,
and logs a one-time WARNING if either soft budget is crossed — the run still
finishes (alerting, not blocking), per the JAR-58 guardrail decision.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from config import LANGSMITH_PROJECT

logger = logging.getLogger(__name__)


def _extract_tokens(response: Any) -> int:
    """Best-effort total-token count from an LLMResult, across provider shapes.

    Prefers llm_output.token_usage (Anthropic-style); falls back to summing
    usage_metadata on chat generations (Ollama / newer LangChain). Returns 0 when
    the provider reports nothing — the call-count budget still applies."""
    out = getattr(response, "llm_output", None) or {}
    usage = out.get("token_usage") or out.get("usage") or {}
    total = usage.get("total_tokens")
    if total:
        return int(total)
    counted = 0
    for gen_list in getattr(response, "generations", []) or []:
        for gen in gen_list:
            um = getattr(getattr(gen, "message", None), "usage_metadata", None)
            if um:
                counted += int(um.get("total_tokens", 0) or 0)
    return counted


class CostGuard(BaseCallbackHandler):
    """Per-run LLM budget tracker (soft / warn-and-continue).

    Counts model invocations and tokens across a whole run; when either budget is
    first exceeded it logs one WARNING and keeps going. Attach via
    config={"callbacks": [guard]} on the graph invoke(s); call log_summary() at
    the end. Catches runaway loops/fan-outs without dropping a legitimate run.
    """

    def __init__(self, call_budget: int, token_budget: int) -> None:
        self.call_budget = call_budget
        self.token_budget = token_budget
        self.calls = 0
        self.tokens = 0
        self._warned = False

    # Chat models fire on_chat_model_start; text-completion models fire on_llm_start.
    def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
        self._count_call()

    def on_llm_start(self, serialized: Any, prompts: Any, **kwargs: Any) -> None:
        self._count_call()

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        self.tokens += _extract_tokens(response)
        self._check()

    def _count_call(self) -> None:
        self.calls += 1
        self._check()

    def _check(self) -> None:
        if not self._warned and (self.calls > self.call_budget or self.tokens > self.token_budget):
            logger.warning(
                "LLM budget exceeded this run: %d calls (budget %d), %d tokens (budget %d) — "
                "continuing, but investigate runaway cost.",
                self.calls, self.call_budget, self.tokens, self.token_budget,
            )
            self._warned = True

    def log_summary(self) -> None:
        logger.info(
            "LLM usage this run: %d calls, %d tokens (budgets: %d calls, %d tokens)",
            self.calls, self.tokens, self.call_budget, self.token_budget,
        )


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
