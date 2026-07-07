"""In-process rate limiting for the serving surface.

One global token bucket — single-user phase, so no per-key map and no locks:
`try_acquire` is synchronous and runs on the event loop thread. Correct only
with a single uvicorn worker (enforced in scripts/serve.py); a multi-worker
deployment needs an external limiter.
"""
from __future__ import annotations

import time
from collections.abc import Callable


class TokenBucket:
    """Classic token bucket on a monotonic clock.

    Refills at `rate_per_minute / 60` tokens per second, capped at `burst`.
    The clock is injectable for tests.
    """

    def __init__(
        self,
        rate_per_minute: float,
        burst: int,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._rate_per_s = rate_per_minute / 60.0
        self._burst = float(burst)
        self._now = now
        self._tokens = float(burst)
        self._last = now()

    def try_acquire(self) -> tuple[bool, float]:
        """Take one token. Returns (allowed, retry_after_seconds)."""
        t = self._now()
        self._tokens = min(self._burst, self._tokens + (t - self._last) * self._rate_per_s)
        self._last = t
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True, 0.0
        deficit = 1.0 - self._tokens
        return False, deficit / self._rate_per_s if self._rate_per_s > 0 else float("inf")
