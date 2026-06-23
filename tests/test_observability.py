"""Unit tests for the per-run LLM CostGuard and token extraction."""
from __future__ import annotations

import logging

from observability import CostGuard, _extract_tokens


class _Msg:
    def __init__(self, total: int) -> None:
        self.usage_metadata = {"total_tokens": total}


class _Gen:
    def __init__(self, total: int) -> None:
        self.message = _Msg(total)


class _Resp:
    """Minimal stand-in for a LangChain LLMResult."""
    def __init__(self, total: int | None = None, llm_output: dict | None = None) -> None:
        self.llm_output = llm_output or {}
        self.generations = [[_Gen(total)]] if total is not None else []


def test_extract_tokens_prefers_llm_output():
    resp = _Resp(llm_output={"token_usage": {"total_tokens": 42}})
    assert _extract_tokens(resp) == 42


def test_extract_tokens_falls_back_to_usage_metadata():
    assert _extract_tokens(_Resp(total=17)) == 17


def test_extract_tokens_zero_when_nothing_reported():
    assert _extract_tokens(_Resp()) == 0


def test_cost_guard_counts_and_is_silent_under_budget(caplog):
    guard = CostGuard(call_budget=10, token_budget=1000)
    with caplog.at_level(logging.WARNING, logger="observability"):
        for _ in range(3):
            guard.on_chat_model_start({}, [])
            guard.on_llm_end(_Resp(total=100))

    assert guard.calls == 3
    assert guard.tokens == 300
    assert not any("budget exceeded" in r.getMessage().lower() for r in caplog.records)


def test_cost_guard_warns_once_when_call_budget_exceeded(caplog):
    guard = CostGuard(call_budget=2, token_budget=10_000)
    with caplog.at_level(logging.WARNING, logger="observability"):
        for _ in range(5):
            guard.on_chat_model_start({}, [])

    assert guard.calls == 5
    breaches = [r for r in caplog.records if "budget exceeded" in r.getMessage().lower()]
    assert len(breaches) == 1     # alerts once, not on every subsequent call


def test_cost_guard_warns_on_token_budget(caplog):
    guard = CostGuard(call_budget=10_000, token_budget=150)
    with caplog.at_level(logging.WARNING, logger="observability"):
        guard.on_chat_model_start({}, [])
        guard.on_llm_end(_Resp(total=200))     # 200 > 150 token budget

    assert any("budget exceeded" in r.getMessage().lower() for r in caplog.records)
