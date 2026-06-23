"""Unit tests for the tracker dossier tail (JAR-56).

Cover the deterministic logic (store read + delta assembly, the meaningful-change
gate, the composite score, output routing) and the field-assembly in the LLM
nodes, using fake connections / models / HTTP clients — no live DB, LLM, or
network is touched.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone

from agents.tracker import dossier
from agents.tracker.dossier import (
    _classify,
    _composite,
    _growth,
    load_signals,
    route_after_signals,
    score_trending,
    synthesize_dossier,
    write_dossier,
    _TrendRationale,
)
from agents.tracker.state import DossierInputs, TrendScore


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class _Cursor:
    def __init__(self, rows: list) -> None:
        self._rows = rows

    def fetchall(self) -> list:
        return self._rows


class _DispatchConn:
    """psycopg-like connection that returns canned rows keyed by the table queried."""

    def __init__(self, **tables) -> None:
        self.tables = tables  # snapshots=, postings=, blog=, releases=, stars=

    def execute(self, sql, params=None):  # noqa: ANN001 - test double
        s = sql.lower()
        if "from snapshots" in s:
            rows = self.tables.get("snapshots", [])
        elif "from postings" in s:
            rows = self.tables.get("postings", [])
        elif "from blog_posts" in s:
            rows = self.tables.get("blog", [])
        elif "from github_releases" in s:
            rows = self.tables.get("releases", [])
        elif "from github_repo_stats" in s:
            rows = self.tables.get("stars", [])
        else:
            rows = []
        return _Cursor(rows)


def _patch_conn(monkeypatch, conn: _DispatchConn) -> None:
    @contextmanager
    def _cm():
        yield conn

    monkeypatch.setattr(dossier, "get_connection", _cm)


def _dt(day: int) -> datetime:
    return datetime(2026, 6, day, tzinfo=timezone.utc)


def _company() -> dict:
    return {"slug": "anthropic", "name": "Anthropic"}


def _signals(**over) -> DossierInputs:
    base = dict(
        company_slug="anthropic", company_name="Anthropic", snapshots_available=8,
        posting_count=200, posting_count_delta=6, posting_count_window_delta=20,
        eng_count=130, eng_count_delta=4, eng_count_window_delta=30,
        new_postings=2, removed_postings=0, new_release_count=4, new_blog_count=3,
        star_delta_by_repo=[("anthropics/sdk", 500)],
    )
    base.update(over)
    return DossierInputs(**base)


# ---------------------------------------------------------------------------
# load_signals
# ---------------------------------------------------------------------------

def test_load_signals_assembles_deltas_and_flags_change(monkeypatch):
    conn = _DispatchConn(
        snapshots=[
            {"snapshot_at": _dt(23), "posting_count": 10, "eng_count": 6,
             "new_ids": ["a", "b"], "removed_ids": []},
            {"snapshot_at": _dt(20), "posting_count": 7, "eng_count": 4,
             "new_ids": ["x"], "removed_ids": ["y"]},
        ],
        postings=[
            {"title": "Staff SRE", "department": "Eng", "seniority": "staff"},
            {"title": "Backend Eng", "department": "Eng", "seniority": "senior"},
            {"title": "Recruiter", "department": "People", "seniority": None},
        ],
        blog=[{"title": "Launch", "url": "http://b/1", "published_at": _dt(22),
               "first_seen_at": _dt(22)}],
        releases=[{"repo": "anthropics/sdk", "release_tag": "v2", "release_name": "Two",
                   "published_at": _dt(21), "first_seen_at": _dt(21)}],
        stars=[{"repo": "anthropics/sdk", "measured_at": _dt(23), "star_count": 1300},
               {"repo": "anthropics/sdk", "measured_at": _dt(20), "star_count": 1000}],
    )
    _patch_conn(monkeypatch, conn)

    out = load_signals({"company": _company()})
    s = out["signals"]

    assert out["meaningful_change"] is True
    assert (s.posting_count, s.posting_count_delta) == (10, 3)
    assert (s.eng_count, s.eng_count_delta) == (6, 2)
    # only two snapshots loaded, so oldest == prev ⇒ window delta equals run delta
    assert (s.posting_count_window_delta, s.eng_count_window_delta) == (3, 2)
    assert (s.new_postings, s.removed_postings) == (2, 0)
    assert s.open_by_department == [("Eng", 2), ("People", 1)]
    assert s.star_delta_by_repo == [("anthropics/sdk", 300)]
    # first_seen_at on the blog/release rows is at/after the previous snapshot ⇒ new
    assert s.new_blog_count == 1 and s.new_release_count == 1
    assert s.snapshots_available == 2


def test_load_signals_no_change_skips(monkeypatch):
    conn = _DispatchConn(
        snapshots=[
            {"snapshot_at": _dt(23), "posting_count": 7, "eng_count": 4,
             "new_ids": [], "removed_ids": []},
            {"snapshot_at": _dt(20), "posting_count": 7, "eng_count": 4,
             "new_ids": [], "removed_ids": []},
        ],
        postings=[{"title": "Backend Eng", "department": "Eng", "seniority": "senior"}],
    )
    _patch_conn(monkeypatch, conn)

    out = load_signals({"company": _company()})
    assert out["meaningful_change"] is False
    assert out["signals"].posting_count_delta == 0


def test_load_signals_first_run_is_meaningful(monkeypatch):
    conn = _DispatchConn(
        snapshots=[{"snapshot_at": _dt(23), "posting_count": 5, "eng_count": 3,
                    "new_ids": [], "removed_ids": []}],
        postings=[{"title": "Eng", "department": "Eng", "seniority": None}],
    )
    _patch_conn(monkeypatch, conn)

    out = load_signals({"company": _company()})
    # only one snapshot ⇒ no deltas, but a first dossier is still worth writing
    assert out["meaningful_change"] is True
    assert out["signals"].snapshots_available == 1
    assert out["signals"].posting_count_delta is None


def test_load_signals_empty_store_is_not_meaningful(monkeypatch):
    _patch_conn(monkeypatch, _DispatchConn())
    out = load_signals({"company": _company()})
    assert out["meaningful_change"] is False
    assert out["signals"].posting_count is None
    assert out["signals"].snapshots_available == 0


# ---------------------------------------------------------------------------
# route_after_signals
# ---------------------------------------------------------------------------

def test_route_synthesizes_on_change():
    assert route_after_signals({"meaningful_change": True}) == "synthesize_dossier"


def test_route_ends_when_no_change():
    assert route_after_signals({"meaningful_change": False}) == "__end__"
    assert route_after_signals({}) == "__end__"


# ---------------------------------------------------------------------------
# synthesize_dossier
# ---------------------------------------------------------------------------

class _FakeMsg:
    def __init__(self, content):
        self.content = content


class _FakeSynthLLM:
    def __init__(self, content):
        self._content = content
        self.received = None

    def invoke(self, messages):
        self.received = messages
        return _FakeMsg(self._content)


def test_synthesize_dossier_returns_none_without_signals():
    assert synthesize_dossier({"signals": None}) == {"dossier_markdown": None}


def test_synthesize_dossier_builds_prompt_from_signals(monkeypatch):
    fake = _FakeSynthLLM("## Summary\nGrowing fast")
    monkeypatch.setattr(dossier, "SYNTHESIS_LLM", fake)

    out = synthesize_dossier({"signals": _signals()})

    assert out["dossier_markdown"] == "## Summary\nGrowing fast"
    prompt = fake.received[1].content
    assert "Anthropic" in prompt
    assert "run-over-run: 6" in prompt             # posting_count_delta surfaced
    assert "over tracked window: 20" in prompt     # window delta surfaced
    assert "pending" in prompt.lower()             # customers metric noted, not invented


# ---------------------------------------------------------------------------
# score_trending — deterministic composite + LLM judgment flag
# ---------------------------------------------------------------------------

def test_growth_rate_and_edges():
    # 30 added on a window-start base of (130-30)=100 ⇒ 0.30 growth
    assert _growth(130, 30) == 0.30
    assert _growth(None, 5) == 0.0          # no current value
    assert _growth(100, None) == 0.0        # no prior window
    assert _growth(5, 10) == 0.0            # base <= 0 ⇒ no spurious spike


def test_composite_is_bounded_hiring_led_sum():
    # eng 0.45*(30/100)=0.135; post 0.20*(20/180)=0.022222; release 0.15*(4/8)=0.075;
    # blog 0.10*(3/6)=0.05; star 0.10*(500/1000)=0.05 → 0.332222 *100 = 33.22
    assert _composite(_signals()) == 33.22


def test_classify_bands():
    assert _classify(25.0) == ("accelerating", True)
    assert _classify(12.0) == ("accelerating", True)    # boundary is inclusive
    assert _classify(5.0) == ("steady", False)
    assert _classify(-3.0) == ("cooling", False)        # boundary is inclusive
    assert _classify(-10.0) == ("cooling", False)


class _FakeChain:
    def __init__(self, result):
        self._result = result

    def invoke(self, messages):
        return self._result


class _FakeRationaleLLM:
    def __init__(self, result):
        self._result = result

    def with_structured_output(self, schema):
        assert schema is _TrendRationale
        return _FakeChain(self._result)


def test_score_trending_classifies_deterministically_with_llm_rationale(monkeypatch):
    # The LLM only supplies the rationale; classification/top_mover are deterministic,
    # so the label can never contradict the score (the live-run bug this fixes).
    monkeypatch.setattr(dossier, "SYNTHESIS_LLM",
                        _FakeRationaleLLM(_TrendRationale(rationale="hiring + release surge")))

    score: TrendScore = score_trending({"signals": _signals()})["trend_score"]

    assert score.composite == 33.22
    assert score.classification == "accelerating"        # 33.22 >= ACCEL band (12.0)
    assert score.is_top_mover is True
    assert score.rationale == "hiring + release surge"


def test_score_trending_quiet_company_is_steady(monkeypatch):
    monkeypatch.setattr(dossier, "SYNTHESIS_LLM",
                        _FakeRationaleLLM(_TrendRationale(rationale="quiet")))

    quiet = _signals(eng_count_window_delta=0, posting_count_window_delta=1,
                     new_release_count=0, new_blog_count=0, star_delta_by_repo=[])
    score = score_trending({"signals": quiet})["trend_score"]
    assert score.classification == "steady" and score.is_top_mover is False


# ---------------------------------------------------------------------------
# write_dossier — Notion upsert + Linear top-mover flag
# ---------------------------------------------------------------------------

def test_write_dossier_upserts_and_flags_top_mover(monkeypatch, caplog):
    calls: list[tuple] = []
    monkeypatch.setattr(dossier, "upsert_company_dossier",
                        lambda md, name, **k: calls.append((md, name)) or "http://notion/d")

    score = TrendScore(composite=15.0, classification="accelerating",
                       rationale="hiring surge", is_top_mover=True)
    import logging
    with caplog.at_level(logging.INFO, logger="agents.tracker.dossier"):
        out = write_dossier({
            "dossier_markdown": "## Summary\nx",
            "signals": _signals(),
            "trend_score": score,
        })

    assert out == {"dossier_url": "http://notion/d"}
    assert calls == [("## Summary\nx", "Anthropic")]
    assert "top mover" in caplog.text.lower()


def test_write_dossier_noop_without_dossier(monkeypatch):
    called = False

    def _boom(*a, **k):
        nonlocal called
        called = True

    monkeypatch.setattr(dossier, "upsert_company_dossier", _boom)
    assert write_dossier({"dossier_markdown": None, "signals": _signals()}) == {}
    assert called is False
