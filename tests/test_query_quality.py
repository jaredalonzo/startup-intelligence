"""Unit tests for the query-head evaluators — fake judge, no LangSmith/LLM."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval.query_dataset_seed import QUERY_EVAL_SEED  # noqa: E402
from eval.query_quality import (  # noqa: E402
    FaithfulnessJudgment,
    GoldSpec,
    make_faithfulness_evaluator,
    make_retrieval_evaluator,
    retrieval_recall,
)


def _doc(slug: str, title: str) -> dict:
    return {"company_slug": slug, "title": title, "kind": "posting"}


# ---------------------------------------------------------------------------
# retrieval_recall — deterministic arithmetic
# ---------------------------------------------------------------------------

def test_recall_full_hit():
    gold = [GoldSpec(company_slug="acme", title_contains="Rust")]
    assert retrieval_recall([_doc("acme", "Senior Rust Engineer")], gold) == 1.0


def test_recall_partial():
    gold = [GoldSpec(company_slug="acme", title_contains="Rust"),
            GoldSpec(company_slug="globex", title_contains="Go")]
    assert retrieval_recall([_doc("acme", "Rust Engineer")], gold) == 0.5


def test_recall_miss():
    gold = [GoldSpec(company_slug="acme", title_contains="Rust")]
    assert retrieval_recall([_doc("acme", "Sales Lead"), _doc("globex", "Rust Dev")], gold) == 0.0


def test_recall_empty_gold_is_trivially_satisfied():
    assert retrieval_recall([_doc("acme", "Anything")], []) == 1.0
    assert retrieval_recall([], []) == 1.0


def test_recall_matching_is_case_insensitive_and_slug_exact():
    gold = [GoldSpec(company_slug="acme", title_contains="rust engineer")]
    assert retrieval_recall([_doc("acme", "Senior RUST Engineer II")], gold) == 1.0
    # slug must match exactly — a similar slug does not count
    assert retrieval_recall([_doc("acme-corp", "Rust Engineer")], gold) == 0.0


def test_recall_empty_title_contains_matches_any_company_doc():
    gold = [GoldSpec(company_slug="acme", title_contains="")]
    assert retrieval_recall([_doc("acme", "Whatever Role")], gold) == 1.0


# ---------------------------------------------------------------------------
# evaluators — feedback-dict plumbing
# ---------------------------------------------------------------------------

def _run(outputs: dict) -> SimpleNamespace:
    return SimpleNamespace(outputs=outputs)


def _example(inputs: dict, outputs: dict) -> SimpleNamespace:
    return SimpleNamespace(inputs=inputs, outputs=outputs)


def test_retrieval_evaluator_feedback_shape():
    evaluator = make_retrieval_evaluator("query_retrieval")
    fb = evaluator(
        _run({"retrieved": [_doc("acme", "Rust Engineer")]}),
        _example({"question": "q"}, {"gold": [{"company_slug": "acme", "title_contains": "Rust"}]}),
    )
    assert fb == {"key": "query_retrieval", "score": 1.0, "comment": "all gold retrieved"}


def test_retrieval_evaluator_reports_misses():
    evaluator = make_retrieval_evaluator()
    fb = evaluator(
        _run({"retrieved": []}),
        _example({}, {"gold": [{"company_slug": "acme", "title_contains": "Rust"}]}),
    )
    assert fb["score"] == 0.0 and "missed: acme:Rust" in fb["comment"]


def test_retrieval_evaluator_abstain_question():
    evaluator = make_retrieval_evaluator()
    fb = evaluator(_run({"retrieved": []}), _example({}, {"gold": []}))
    assert fb["score"] == 1.0 and "abstain" in fb["comment"]


class _FakeJudgeChain:
    def __init__(self, verdict: FaithfulnessJudgment) -> None:
        self._verdict = verdict
        self.messages: list = []

    def invoke(self, messages):  # noqa: ANN001 - test double
        self.messages.append(messages)
        return self._verdict


def test_faithfulness_evaluator_plumbs_judgment(monkeypatch):
    from eval import query_quality

    chain = _FakeJudgeChain(FaithfulnessJudgment(score=0.75, rationale="one uncited claim"))
    monkeypatch.setattr(query_quality, "structured", lambda _llm, _schema: chain)

    evaluator = make_faithfulness_evaluator(judge_llm=object(), key="query_faithfulness")
    fb = evaluator(
        _run({"answer_markdown": "Acme hires [1].", "context": "[1] Acme — Rust"}),
        _example({"question": "who hires rust?"}, {}),
    )

    assert fb == {"key": "query_faithfulness", "score": 0.75, "comment": "one uncited claim"}
    [messages] = chain.messages
    user = messages[1]["content"]
    assert "who hires rust?" in user and "[1] Acme — Rust" in user and "Acme hires [1]." in user


# ---------------------------------------------------------------------------
# dataset seed — structural sanity
# ---------------------------------------------------------------------------

def test_seed_entries_are_well_formed():
    assert len(QUERY_EVAL_SEED) >= 15
    abstain = 0
    for entry in QUERY_EVAL_SEED:
        assert entry["question"].strip()
        specs = [GoldSpec.model_validate(g) for g in entry["gold"]]  # validates shape
        if not specs:
            abstain += 1
    assert abstain >= 3  # the abstain cases are present
