"""Serving settings — the per-process knobs the app factory consumes.

`QUERY_API_KEY` is a secret, so it is read here from the environment (like
LINEAR_API_KEY in outputs/linear.py) rather than exported from config.py.
The numeric knobs default from config.py's serving block and exist as
dataclass fields so tests can construct tight limits directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from config import (
    QUERY_MAX_CONCURRENCY,
    QUERY_RATE_BURST,
    QUERY_RATE_PER_MINUTE,
    QUERY_REQUEST_CALL_BUDGET,
    QUERY_REQUEST_TIMEOUT_S,
    QUERY_REQUEST_TOKEN_BUDGET,
)


@dataclass(frozen=True)
class ServingSettings:
    api_key: str
    call_budget: int = QUERY_REQUEST_CALL_BUDGET
    token_budget: int = QUERY_REQUEST_TOKEN_BUDGET
    timeout_s: float = QUERY_REQUEST_TIMEOUT_S
    rate_per_minute: float = QUERY_RATE_PER_MINUTE
    rate_burst: int = QUERY_RATE_BURST
    max_concurrency: int = QUERY_MAX_CONCURRENCY

    @classmethod
    def from_env(cls) -> ServingSettings:
        """Build settings from the environment; refuse to start without a key.

        No anonymous/localhost-only mode: that would couple auth policy to
        the bind address. Generate a key with
        `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
        """
        api_key = os.getenv("QUERY_API_KEY")
        if not api_key:
            raise RuntimeError(
                "QUERY_API_KEY is not set — the serving surface refuses to start "
                "without one. Generate: python -c "
                '"import secrets; print(secrets.token_urlsafe(32))"'
            )
        return cls(api_key=api_key)
