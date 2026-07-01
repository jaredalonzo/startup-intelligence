"""Unit tests for the partial-fetch guard (JAR-94).

is_suspect_drop is pure and exercised directly. latest_posting_count is exercised
against a fake conn. No live Postgres.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion.snapshot import is_suspect_drop, latest_posting_count  # noqa: E402


# ---------------------------------------------------------------------------
# is_suspect_drop (pure)
# ---------------------------------------------------------------------------

_GUARD = {"min_prev": 5, "max_drop_fraction": 0.6}   # suspect below 40% of prev


def test_first_run_is_never_suspect():
    # No prior snapshot ⇒ nothing to compare against ⇒ always trusted.
    assert is_suspect_drop(None, 0, **_GUARD) is False
    assert is_suspect_drop(None, 42, **_GUARD) is False


def test_small_prior_board_is_never_suspect():
    # Below the floor a ratio test is meaningless: 4 -> 1 is ordinary churn, not a
    # truncated fetch, so it must be trusted through.
    assert is_suspect_drop(4, 1, **_GUARD) is False
    assert is_suspect_drop(4, 0, **_GUARD) is False


def test_implausible_collapse_is_suspect():
    # A healthy board that suddenly reports a fraction of its postings is the
    # signature of a partial fetch — flag it so the caller skips the snapshot.
    assert is_suspect_drop(100, 10, **_GUARD) is True
    assert is_suspect_drop(50, 0, **_GUARD) is True     # collapse to zero at/above floor


def test_modest_drop_is_trusted():
    # A drop that stays above the (1 - fraction) threshold is plausible attrition
    # (roles filled/closed) and must be recorded, not suppressed.
    assert is_suspect_drop(100, 41, **_GUARD) is False  # 41% survives; above 40% floor
    assert is_suspect_drop(100, 100, **_GUARD) is False
    assert is_suspect_drop(100, 130, **_GUARD) is False  # growth is never a drop


def test_threshold_boundary_is_exclusive():
    # Exactly (1 - fraction) * prev survives; only strictly below it is suspect.
    assert is_suspect_drop(100, 40, **_GUARD) is False   # == threshold, trusted
    assert is_suspect_drop(100, 39, **_GUARD) is True     # just under, suspect


def test_prev_at_floor_engages_the_guard():
    # min_prev is inclusive: a prior count equal to the floor is guarded.
    assert is_suspect_drop(5, 0, **_GUARD) is True
    assert is_suspect_drop(4, 0, **_GUARD) is False       # one below the floor: trusted


# ---------------------------------------------------------------------------
# latest_posting_count (fake conn)
# ---------------------------------------------------------------------------

class _FakeCursor:
    def __init__(self, one):
        self._one = one

    def fetchone(self):
        return self._one


class _FakeConn:
    def __init__(self, one):
        self._one = one
        self.calls: list[tuple] = []

    def execute(self, sql, params=None):  # noqa: ANN001 - test double
        self.calls.append((sql, params))
        return _FakeCursor(self._one)


def test_latest_posting_count_returns_value():
    conn = _FakeConn({"posting_count": 37})
    assert latest_posting_count("acme", conn) == 37
    # scoped to the requested company
    assert conn.calls[0][1] == ("acme",)


def test_latest_posting_count_none_when_no_snapshots():
    assert latest_posting_count("newco", _FakeConn(None)) is None
