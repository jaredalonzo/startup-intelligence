"""Unit tests for the shared extraction-quality judge.

The judge model is faked (no live LLM, no LangSmith), so these exercise the pure
core and both adapters: offline (run, example) and online (production Run).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval.extraction_quality import (  # noqa: E402
    ExtractionJudgment,
    extract_run_payload,
    judge_extraction,
    make_offline_evaluator,
    make_online_evaluator,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class _FakeChain:
    def __init__(self, result: ExtractionJudgment) -> None:
        self._result = result
        self.invoked_with = None

    def invoke(self, messages):
        self.invoked_with = messages
        return self._result


class _FakeJudgeLLM:
    """Stand-in for a chat model: with_structured_output(...).invoke(...)."""

    def __init__(self, result: ExtractionJudgment) -> None:
        self._result = result
        self.chain = _FakeChain(result)

    def with_structured_output(self, schema):
        self.schema = schema
        return self.chain


_VERDICT = ExtractionJudgment(score=0.8, rationale="skills match the JD")


# ---------------------------------------------------------------------------
# Pure core
# ---------------------------------------------------------------------------

def test_judge_extraction_returns_verdict_and_caps_jd():
    llm = _FakeJudgeLLM(_VERDICT)
    long_jd = "x" * 10_000
    out = judge_extraction(long_jd, {"skills": ["Kubernetes"]}, llm)

    assert out == _VERDICT
    assert llm.schema is ExtractionJudgment
    # JD is truncated to the cap before reaching the model
    user_msg = llm.chain.invoked_with[1]["content"]
    assert "x" * 6000 in user_msg
    assert "x" * 6001 not in user_msg


# ---------------------------------------------------------------------------
# Offline adapter
# ---------------------------------------------------------------------------

def test_offline_evaluator_reads_example_and_run():
    evaluator = make_offline_evaluator(_FakeJudgeLLM(_VERDICT))
    run = SimpleNamespace(outputs={"skills": ["Go"], "platforms": ["AWS"]})
    example = SimpleNamespace(inputs={"description_text": "We need a Go engineer on AWS"})

    fb = evaluator(run, example)
    assert fb == {"key": "extraction_quality", "score": 0.8, "comment": "skills match the JD"}


def test_offline_evaluator_handles_empty_outputs():
    evaluator = make_offline_evaluator(_FakeJudgeLLM(_VERDICT))
    run = SimpleNamespace(outputs=None)
    example = SimpleNamespace(inputs=None)
    # Should not raise; judge still invoked with empty extraction/JD.
    assert evaluator(run, example)["score"] == 0.8


# ---------------------------------------------------------------------------
# Run payload extraction
# ---------------------------------------------------------------------------

def test_extract_run_payload_parses_production_shape():
    run = SimpleNamespace(
        inputs={"posting": {"description_text": "JD body", "title": "SRE"}},
        outputs={"extractions": [{"skills": ["Kubernetes"], "platforms": ["GCP"],
                                  "seniority": "senior", "years_experience": 5}]},
    )
    payload = extract_run_payload(run)
    assert payload is not None
    jd, extraction = payload
    assert jd == "JD body"
    assert extraction["skills"] == ["Kubernetes"]


def test_extract_run_payload_none_when_incomplete():
    no_jd = SimpleNamespace(inputs={"posting": {}}, outputs={"extractions": [{"skills": []}]})
    no_extraction = SimpleNamespace(inputs={"posting": {"description_text": "x"}}, outputs={"extractions": []})
    empty = SimpleNamespace(inputs=None, outputs=None)

    assert extract_run_payload(no_jd) is None
    assert extract_run_payload(no_extraction) is None
    assert extract_run_payload(empty) is None


# ---------------------------------------------------------------------------
# Online adapter
# ---------------------------------------------------------------------------

def test_online_evaluator_scores_production_run():
    evaluator = make_online_evaluator(_FakeJudgeLLM(_VERDICT))
    run = SimpleNamespace(
        inputs={"posting": {"description_text": "JD body"}},
        outputs={"extractions": [{"skills": ["Kubernetes"]}]},
    )
    fb = evaluator(run)
    assert fb == {"key": "extraction_quality", "score": 0.8, "comment": "skills match the JD"}


def test_online_evaluator_skips_unscorable_run():
    evaluator = make_online_evaluator(_FakeJudgeLLM(_VERDICT))
    run = SimpleNamespace(inputs={"posting": {}}, outputs={"extractions": []})
    assert evaluator(run) is None
