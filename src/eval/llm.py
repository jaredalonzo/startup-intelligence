"""Build a chat model by name for eval (judge or candidate target).

Centralizes the claude-* → Anthropic / else → Ollama dispatch so the offline
bake-off and the online evaluator construct models the same way.
"""
from __future__ import annotations

from typing import Any


def build_llm(model: str) -> Any:
    """Construct a chat model by name. claude-* → Anthropic (lazy import); else Ollama.

    temperature=0 for determinism so a comparison reflects the model, not sampling.
    """
    if model.startswith("claude"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=0, max_tokens=1024)  # type: ignore[call-arg]
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model, temperature=0)
