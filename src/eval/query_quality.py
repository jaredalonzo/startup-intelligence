"""Evaluators for the query head: deterministic retrieval recall + judged faithfulness.

Two evaluators over one full-graph run per example (scripts/eval_query.py):

  - retrieval recall — deterministic, no judge. Gold docs are identified by
    (company_slug, title substring) — ATS posting IDs churn as boards repost,
    slug + substring is stable and human-auditable. Scores the fraction of
    gold specs matched by any retrieved doc.
  - answer faithfulness — LLM-as-judge, mirroring extraction_quality.py: is
    every claim in the answer supported by the provided context, with [n]
    citations? An honest abstention on unsupportive context scores HIGH —
    faithfulness is not helpfulness.

The judge model is injected so this module stays unit-testable with a fake.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from pydantic import BaseModel, Field

from llm_structured import structured

# Cap the context slice sent to the judge — same budget discipline as the
# extraction judge (the answer must cite [n] entries near the top anyway).
_MAX_CONTEXT_CHARS = 12000


# ---------------------------------------------------------------------------
# Retrieval recall — deterministic
# ---------------------------------------------------------------------------

class GoldSpec(BaseModel):
    """One known-relevant document a good retrieval should surface.

    Matched when a retrieved doc has this company_slug AND its title contains
    title_contains (case-insensitive). An empty title_contains matches any
    title from the company.
    """

    company_slug: str
    title_contains: str = ""


def _matches(spec: GoldSpec, doc: Mapping[str, Any]) -> bool:
    if doc.get("company_slug") != spec.company_slug:
        return False
    title = str(doc.get("title") or "")
    return spec.title_contains.lower() in title.lower()


def retrieval_recall(
    retrieved: Sequence[Mapping[str, Any]], gold: Sequence[GoldSpec]
) -> float:
    """Fraction of gold specs matched by any retrieved doc. Empty gold → 1.0.

    Empty gold marks an abstain question: there is nothing to retrieve, so
    retrieval trivially succeeds (the faithfulness judge scores the abstention).
    """
    if not gold:
        return 1.0
    hits = sum(1 for spec in gold if any(_matches(spec, doc) for doc in retrieved))
    return hits / len(gold)


def make_retrieval_evaluator(
    key: str = "query_retrieval",
) -> Callable[[Any, Any], dict[str, Any]]:
    """LangSmith evaluator: gold from the example outputs, docs from the run outputs."""

    def evaluator(run: Any, example: Any) -> dict[str, Any]:
        gold_raw = (example.outputs or {}).get("gold", [])
        gold = [GoldSpec.model_validate(g) for g in gold_raw]
        retrieved = (run.outputs or {}).get("retrieved", [])
        score = retrieval_recall(retrieved, gold)
        missed = [
            f"{g.company_slug}:{g.title_contains}"
            for g in gold
            if not any(_matches(g, doc) for doc in retrieved)
        ]
        comment = "all gold retrieved" if not missed else f"missed: {', '.join(missed)}"
        if not gold:
            comment = "abstain question (no gold) — recall trivially 1.0"
        return {"key": key, "score": score, "comment": comment}

    return evaluator


# ---------------------------------------------------------------------------
# Answer faithfulness — LLM-as-judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = (
    "You grade the faithfulness of an answer produced from numbered context "
    "documents. Score 0-1 on grounding alone, not helpfulness: every factual "
    "claim must be supported by a provided document and carry its [n] citation; "
    "trend language ('rising', 'growing') is unsupported unless a provided field "
    "states it; a Sources section must list the cited documents. An explicit "
    "abstention ('the context does not support an answer') is PERFECTLY faithful "
    "when the context is empty or off-topic — score it 1.0. Unsupported claims, "
    "missing citations, or invented details reduce the score."
)


class FaithfulnessJudgment(BaseModel):
    """Structured judge verdict (never free prose downstream code must parse)."""

    score: float  # 0.0 (ungrounded) .. 1.0 (every claim cited & supported, or honest abstain)
    unsupported_claims: list[str] = Field(default_factory=list)
    citations_present: bool = True
    rationale: str = ""


def judge_faithfulness(
    question: str, context: str, answer: str, judge_llm: Any
) -> FaithfulnessJudgment:
    """Score one answer against the context it was given. `judge_llm` is injected."""
    chain = structured(judge_llm, FaithfulnessJudgment)
    verdict: FaithfulnessJudgment = chain.invoke(
        [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"QUESTION:\n{question}\n\n"
                    f"CONTEXT DOCUMENTS:\n{context[:_MAX_CONTEXT_CHARS]}\n\n"
                    f"ANSWER:\n{answer}"
                ),
            },
        ]
    )
    return verdict


def make_faithfulness_evaluator(
    judge_llm: Any, key: str = "query_faithfulness"
) -> Callable[[Any, Any], dict[str, Any]]:
    """LangSmith evaluator: question from example inputs, answer+context from run outputs."""

    def evaluator(run: Any, example: Any) -> dict[str, Any]:
        question = (example.inputs or {}).get("question", "")
        outputs = run.outputs or {}
        answer = outputs.get("answer_markdown", "")
        context = outputs.get("context", "")
        verdict = judge_faithfulness(question, context, answer, judge_llm)
        return {"key": key, "score": verdict.score, "comment": verdict.rationale}

    return evaluator
