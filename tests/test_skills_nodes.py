"""Unit tests for the skills agent node implementations.

These cover the deterministic logic (taxonomy normalization, trend aggregation,
output routing) and the field-assembly in the LLM nodes, using a fake connection
and fake models so no live DB / LLM / network is required.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone

from agents.skills import nodes
from agents.skills.nodes import (
    _PostingExtraction,
    _normalize_list,
    aggregate_trends,
    extract_one,
    extract_skills,
    load_deltas,
    normalize_taxonomy,
    route_outputs,
    synthesize_radar,
)
from agents.skills.state import SkillExtraction, SkillTrend, TrendReport


# ---------------------------------------------------------------------------
# Test doubles / helpers
# ---------------------------------------------------------------------------

class _FakeCursor:
    def __init__(self, rows: list) -> None:
        self._rows = rows

    def fetchall(self) -> list:
        return self._rows


class _FakeConn:
    """Minimal psycopg-like connection. Records calls; returns canned rows."""

    def __init__(self, rows: list | None = None) -> None:
        self.rows = rows or []
        self.calls: list[tuple] = []
        self.commits = 0

    def execute(self, sql, params=None):  # noqa: ANN001 - test double
        self.calls.append((sql, params))
        return _FakeCursor(self.rows)

    def commit(self) -> None:
        self.commits += 1


def _patch_conn(monkeypatch, conn: _FakeConn) -> None:
    @contextmanager
    def _cm():
        yield conn

    monkeypatch.setattr(nodes, "get_connection", _cm)


def _ex(skills, platforms, **over) -> SkillExtraction:
    base = dict(
        posting_id="p", ats="greenhouse", company_slug="anthropic",
        skills=skills, platforms=platforms, seniority=None, years_experience=None,
        comp_min=None, comp_max=None, comp_currency=None, comp_interval=None,
    )
    base.update(over)
    return SkillExtraction(**base)


# ---------------------------------------------------------------------------
# _normalize_list (pure)
# ---------------------------------------------------------------------------

_ALIASES = {"k8s": "Kubernetes", "postgres": "PostgreSQL", "ts": "TypeScript"}
_KNOWN = frozenset(_ALIASES.values())


def test_normalize_collapses_aliases_case_insensitively():
    norm, unknown = _normalize_list(["k8s", "K8S", "Postgres"], _ALIASES, _KNOWN)
    assert norm == ["Kubernetes", "PostgreSQL"]   # deduped, canonicalized
    assert unknown == []


def test_normalize_canonical_passes_through_not_flagged_unknown():
    norm, unknown = _normalize_list(["Kubernetes"], _ALIASES, _KNOWN)
    assert norm == ["Kubernetes"]
    assert unknown == []


def test_normalize_flags_genuinely_unknown_skill():
    norm, unknown = _normalize_list(["Rust"], _ALIASES, _KNOWN)
    assert norm == ["Rust"]          # kept verbatim
    assert unknown == ["Rust"]       # but flagged for review


def test_normalize_dedupes_alias_and_canonical_to_one():
    norm, _ = _normalize_list(["k8s", "Kubernetes"], _ALIASES, _KNOWN)
    assert norm == ["Kubernetes"]


# ---------------------------------------------------------------------------
# normalize_taxonomy (node) — alias table injected for determinism
# ---------------------------------------------------------------------------

def test_normalize_taxonomy_applies_aliases_and_collects_unknowns(monkeypatch):
    monkeypatch.setattr(nodes, "_load_aliases", lambda: ({"k8s": "Kubernetes"},
                                                         frozenset({"Kubernetes"})))
    state = {"extractions": [_ex(["k8s", "Rust"], ["k8s"], seniority="senior", comp_min=100)]}

    out = normalize_taxonomy(state)

    result = out["normalized_extractions"][0]
    assert result.skills == ["Kubernetes", "Rust"]
    assert result.platforms == ["Kubernetes"]
    assert out["unknown_skills"] == ["Rust"]          # sorted, unique
    # untouched fields survive the model_copy
    assert result.seniority == "senior"
    assert result.comp_min == 100


def test_normalize_taxonomy_empty_is_noop(monkeypatch):
    monkeypatch.setattr(nodes, "_load_aliases", lambda: ({}, frozenset()))
    out = normalize_taxonomy({"extractions": []})
    assert out == {"normalized_extractions": [], "unknown_skills": []}


def test_normalize_taxonomy_accepts_dict_extractions_from_checkpointer(monkeypatch):
    # Regression: the Postgres checkpointer round-trips the extractions channel
    # and can hand back plain dicts instead of SkillExtraction. The node must
    # coerce them, not crash on `extraction.skills`.
    monkeypatch.setattr(nodes, "_load_aliases", lambda: ({"k8s": "Kubernetes"},
                                                         frozenset({"Kubernetes"})))
    dict_ex = _ex(["k8s", "Rust"], ["k8s"]).model_dump()   # checkpointer → dict
    assert isinstance(dict_ex, dict)

    out = normalize_taxonomy({"extractions": [dict_ex]})

    result = out["normalized_extractions"][0]
    assert result.skills == ["Kubernetes", "Rust"]
    assert result.platforms == ["Kubernetes"]
    assert out["unknown_skills"] == ["Rust"]


# ---------------------------------------------------------------------------
# aggregate_trends — fake empty DB ⇒ no previous window
# ---------------------------------------------------------------------------

def test_aggregate_trends_counts_thresholds_and_co_occurrence(monkeypatch):
    conn = _FakeConn(rows=[])   # empty extractions table ⇒ prev counts all 0
    _patch_conn(monkeypatch, conn)

    # Kubernetes×3, Rust×3 qualify (>= SKILLS_MIN_POSTING_COUNT); Go×1 does not.
    extractions = [
        _ex(["Kubernetes", "Rust", "Go"], ["AWS"]),
        _ex(["Kubernetes", "Rust"], ["AWS"]),
        _ex(["Kubernetes", "Rust"], ["AWS"]),
    ]
    out = aggregate_trends({"normalized_extractions": extractions})
    report = out["trend_report"]

    assert report.total_postings == 3
    assert {t.skill for t in report.rising} == {"Kubernetes", "Rust"}
    assert all(t.delta == 3 for t in report.rising)        # prev window empty
    assert set(report.new) == {"Kubernetes", "Rust"}
    assert all("Go" != t.skill for t in report.rising)     # below threshold, excluded
    assert "Go" not in report.new

    k = next(t for t in report.rising if t.skill == "Kubernetes")
    assert k.pct_of_postings == 1.0

    assert report.top_platforms[0].skill == "AWS"
    assert report.top_platforms[0].count_current == 3

    assert ("Kubernetes", "Rust", 3) in report.co_occurrences
    assert conn.commits == 1     # extractions were persisted before diffing


def test_aggregate_trends_empty_window(monkeypatch):
    conn = _FakeConn(rows=[])
    _patch_conn(monkeypatch, conn)
    report = aggregate_trends({"normalized_extractions": []})["trend_report"]
    assert report.total_postings == 0
    assert report.rising == [] and report.falling == [] and report.new == []


def test_aggregate_trends_accepts_dict_extractions_from_checkpointer(monkeypatch):
    # Regression: normalized_extractions also round-trips the checkpointer, so
    # aggregate_trends must coerce dicts back to models before reading ex.skills.
    conn = _FakeConn(rows=[])
    _patch_conn(monkeypatch, conn)
    dict_exs = [_ex(["Kubernetes", "Rust"], ["AWS"]).model_dump() for _ in range(3)]
    assert all(isinstance(d, dict) for d in dict_exs)

    report = aggregate_trends({"normalized_extractions": dict_exs})["trend_report"]

    assert report.total_postings == 3
    assert {t.skill for t in report.rising} == {"Kubernetes", "Rust"}
    assert conn.commits == 1     # persisted despite dict inputs


# ---------------------------------------------------------------------------
# route_outputs — Notion write + gap-task logging
# ---------------------------------------------------------------------------

def test_route_outputs_writes_digest_and_creates_gap_tasks(monkeypatch):
    written: list[str] = []
    monkeypatch.setattr(nodes, "write_skills_digest",
                        lambda digest, *a, **k: written.append(digest) or "http://notion/x")
    gap_calls: list[list] = []
    monkeypatch.setattr(nodes, "create_gap_tasks",
                        lambda skills, *a, **k: gap_calls.append(skills) or ["JAR-1"])

    trend = SkillTrend(skill="Kubernetes", count_current=10, count_previous=2,
                       delta=8, pct_of_postings=0.5)   # well above 15% gap threshold
    report = TrendReport(window_days=30, total_postings=20, rising=[trend],
                         falling=[], new=[], top_platforms=[], co_occurrences=[])

    out = route_outputs({"radar_digest": "## Heading\n- point", "trend_report": report})

    assert out == {}
    assert written == ["## Heading\n- point"]
    # the gap skill is passed through to the Linear writer as (skill, pct)
    assert gap_calls == [[("Kubernetes", 0.5)]]


def test_route_outputs_skips_notion_when_no_digest(monkeypatch):
    written: list[str] = []
    monkeypatch.setattr(nodes, "write_skills_digest",
                        lambda digest, *a, **k: written.append(digest))
    route_outputs({"radar_digest": "", "trend_report": None})
    assert written == []


# ---------------------------------------------------------------------------
# extract_one (LLM node) — field assembly from posting + model output
# ---------------------------------------------------------------------------

class _FakeChain:
    def __init__(self, result):
        self._result = result
        self.invoked_with = None

    def invoke(self, messages):
        self.invoked_with = messages
        return self._result


class _FakeStructuredLLM:
    def __init__(self, result):
        self._result = result

    def with_structured_output(self, schema):
        self.schema = schema
        return _FakeChain(self._result)


def test_extract_one_degrades_to_empty_on_persistent_failure(monkeypatch):
    # A posting whose extraction keeps failing must not abort the run — it yields
    # an empty extraction so trends just exclude it.
    monkeypatch.setattr(nodes.time, "sleep", lambda *_: None)  # no real backoff in tests
    calls = {"n": 0}

    def _boom(posting, llm=None):
        calls["n"] += 1
        raise RuntimeError("cloud 429")

    monkeypatch.setattr(nodes, "extract_posting_fields", _boom)

    posting = {"id": "p9", "ats": "ashby", "company_slug": "openai", "title": "SRE"}
    out = extract_one({"posting": posting})

    [ex] = out["extractions"]
    assert (ex.posting_id, ex.ats, ex.company_slug) == ("p9", "ashby", "openai")
    assert ex.skills == [] and ex.platforms == [] and ex.seniority is None
    assert calls["n"] == 2  # retried before giving up


def test_extract_one_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(nodes.time, "sleep", lambda *_: None)
    result = _PostingExtraction(skills=["Go"], platforms=["AWS"], seniority="staff",
                                years_experience=4)
    calls = {"n": 0}

    def _flaky(posting, llm=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return result

    monkeypatch.setattr(nodes, "extract_posting_fields", _flaky)

    out = extract_one({"posting": {"id": "p1", "ats": "ashby", "company_slug": "openai"}})
    [ex] = out["extractions"]
    assert ex.skills == ["Go"] and ex.seniority == "staff"
    assert calls["n"] == 2


def test_extract_one_maps_model_output_and_posting_comp(monkeypatch):
    result = _PostingExtraction(skills=["Kubernetes"], platforms=["AWS"],
                                seniority="senior", years_experience=5)
    monkeypatch.setattr(nodes, "EXTRACTION_LLM", _FakeStructuredLLM(result))

    posting = {
        "id": "p1", "ats": "greenhouse", "company_slug": "anthropic",
        "title": "SRE", "department": "Eng", "description_text": "run things",
        "compensation_min": 100, "compensation_max": 200,
        "compensation_currency": "USD", "compensation_interval": "annual",
    }
    out = extract_one({"posting": posting})

    [ex] = out["extractions"]
    assert (ex.posting_id, ex.ats, ex.company_slug) == ("p1", "greenhouse", "anthropic")
    assert ex.skills == ["Kubernetes"] and ex.platforms == ["AWS"]
    assert ex.seniority == "senior" and ex.years_experience == 5
    # comp comes from the posting row, not the LLM
    assert (ex.comp_min, ex.comp_max, ex.comp_currency, ex.comp_interval) == \
        (100, 200, "USD", "annual")


def test_extract_one_handles_missing_comp_and_text(monkeypatch):
    result = _PostingExtraction(skills=[], platforms=[], seniority=None, years_experience=None)
    monkeypatch.setattr(nodes, "EXTRACTION_LLM", _FakeStructuredLLM(result))
    posting = {"id": "p2", "ats": "lever", "company_slug": "mistral", "title": "Eng"}

    [ex] = extract_one({"posting": posting})["extractions"]
    assert ex.comp_min is None and ex.comp_currency is None
    assert ex.skills == []


# ---------------------------------------------------------------------------
# synthesize_radar (LLM node)
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


def test_synthesize_radar_returns_none_without_report():
    assert synthesize_radar({"trend_report": None}) == {"radar_digest": None}


def test_synthesize_radar_passes_through_model_content(monkeypatch):
    fake = _FakeSynthLLM("## Heating Up\n- Kubernetes")
    monkeypatch.setattr(nodes, "SYNTHESIS_LLM", fake)
    report = TrendReport(window_days=30, total_postings=42, rising=[], falling=[],
                         new=[], top_platforms=[], co_occurrences=[])

    out = synthesize_radar({"trend_report": report})

    assert out["radar_digest"] == "## Heating Up\n- Kubernetes"
    # the prompt is built from the report
    assert "42" in fake.received[1].content


# ---------------------------------------------------------------------------
# load_deltas (DB node) — watermark handling
# ---------------------------------------------------------------------------

def test_load_deltas_defaults_watermark_and_returns_rows(monkeypatch):
    row = {"id": "x", "title": "Software Engineer", "department": "Engineering"}
    conn = _FakeConn(rows=[row])
    _patch_conn(monkeypatch, conn)

    out = load_deltas({})

    assert out["new_postings"] == [row]
    wm = datetime.fromisoformat(out["watermark"])
    assert (datetime.now(timezone.utc) - wm).total_seconds() < 5   # advanced to ~now


def test_load_deltas_filters_non_technical_by_default(monkeypatch):
    eng = {"id": "e", "title": "Forward Deployed Engineer", "department": "Eng"}
    sales = {"id": "s", "title": "Account Executive", "department": "Sales"}
    _patch_conn(monkeypatch, _FakeConn(rows=[eng, sales]))

    out = load_deltas({})

    assert out["new_postings"] == [eng]   # the GTM role is dropped


def test_load_deltas_all_roles_keeps_everything(monkeypatch):
    eng = {"id": "e", "title": "ML Engineer", "department": "Eng"}
    sales = {"id": "s", "title": "Account Executive", "department": "Sales"}
    _patch_conn(monkeypatch, _FakeConn(rows=[eng, sales]))

    out = load_deltas({}, {"configurable": {"all_roles": True}})

    assert out["new_postings"] == [eng, sales]


def test_load_deltas_uses_provided_watermark_in_query(monkeypatch):
    conn = _FakeConn(rows=[])
    _patch_conn(monkeypatch, conn)
    provided = "2026-06-01T00:00:00+00:00"

    load_deltas({"watermark": provided})

    _, params = conn.calls[0]
    assert params["watermark"] == datetime.fromisoformat(provided)


# ---------------------------------------------------------------------------
# extract_skills coordinator (no-op)
# ---------------------------------------------------------------------------

def test_extract_skills_is_noop():
    assert extract_skills({"new_postings": [{"id": "a"}]}) == {}
