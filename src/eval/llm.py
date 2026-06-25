"""Build a chat model by name for eval (judge or candidate target).

Centralizes the claude-* → Anthropic / else → Ollama dispatch so the offline
bake-off and the online evaluator construct models the same way.
"""
from __future__ import annotations

from typing import Any


def build_llm(model: str, *, cloud: bool | None = None) -> Any:
    """Construct a chat model by name. claude-* → Anthropic (lazy import); else Ollama.

    ``cloud`` selects the Ollama backend per model (None = global default,
    True = Ollama Cloud, False = local daemon), so a single run can mix backends
    — e.g. local candidate models judged by a cloud model. Ignored for claude-*.
    temperature=0 for determinism so a comparison reflects the model, not sampling.
    """
    if model.startswith("claude"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=0, max_tokens=1024)  # type: ignore[call-arg]
    # Ollama (local or cloud) — reuse the configured backend so the judge and
    # bake-off targets pick up Ollama Cloud auth the same way the graphs do.
    from config import build_ollama
    return build_ollama(model, cloud=cloud)
