"""LLM-as-judge for skills-extraction quality.

Reference-free: the judge scores a candidate extraction against the JD text
itself (precision, recall, canonical naming, seniority), not against a gold
label. One pure core (`judge_extraction`) with two thin adapters:

  - `make_offline_evaluator` → a LangSmith `evaluate()` evaluator, signature
    (run, example), used by the model bake-off over a fixed dataset.
  - `make_online_evaluator` → scores a single production trace Run, used by the
    online evaluator that writes feedback back onto live skills-agent runs.

The judge model is injected (a LangChain chat model) so this module has no LLM
or LangSmith import-time dependency and is unit-testable with a fake.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

from pydantic import BaseModel

# Cap the JD slice sent to the judge — long postings cost tokens without adding
# signal, and keeps the judge prompt within a small, predictable budget.
_MAX_JD_CHARS = 6000

JUDGE_SYSTEM_PROMPT = (
    "You grade skill-extraction quality from job postings. Given the JD text and a "
    "candidate extraction, score 0-1: are the listed skills/platforms actually "
    "required by the JD (precision), are obvious ones missing (recall), are names "
    "canonical, and is seniority right? Penalize hallucinated or generic entries."
)


class ExtractionJudgment(BaseModel):
    """Structured judge verdict (never free prose downstream code must parse)."""
    score: float        # 0.0 (wrong/empty) .. 1.0 (accurate and complete)
    rationale: str


def _build_user_prompt(jd_text: str, extraction: Mapping[str, Any]) -> str:
    return (
        f"JD:\n{jd_text[:_MAX_JD_CHARS]}\n\nEXTRACTION:\n"
        f"skills={extraction.get('skills')}\nplatforms={extraction.get('platforms')}\n"
        f"seniority={extraction.get('seniority')}\n"
        f"years_experience={extraction.get('years_experience')}"
    )


def judge_extraction(
    jd_text: str,
    extraction: Mapping[str, Any],
    judge_llm: Any,
) -> ExtractionJudgment:
    """Score one extraction against its JD. `judge_llm` is a LangChain chat model."""
    chain = judge_llm.with_structured_output(ExtractionJudgment)
    verdict: ExtractionJudgment = chain.invoke([
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(jd_text, extraction)},
    ])
    return verdict


def _feedback(verdict: ExtractionJudgment, key: str) -> dict[str, Any]:
    return {"key": key, "score": verdict.score, "comment": verdict.rationale}


# ---------------------------------------------------------------------------
# Offline — LangSmith evaluate() evaluator over a fixed dataset
# ---------------------------------------------------------------------------

def make_offline_evaluator(
    judge_llm: Any, key: str = "extraction_quality",
) -> Callable[[Any, Any], dict[str, Any]]:
    """Return an `evaluate()` evaluator: (run, example) -> feedback dict.

    The JD comes from the dataset example's inputs; the extraction from the
    target run's outputs (the production extraction under the candidate model).
    """
    def evaluator(run: Any, example: Any) -> dict[str, Any]:
        jd = (example.inputs or {}).get("description_text", "")
        extraction = run.outputs or {}
        verdict = judge_extraction(jd, extraction, judge_llm)
        return _feedback(verdict, key)

    return evaluator


# ---------------------------------------------------------------------------
# Online — score a single production trace Run
# ---------------------------------------------------------------------------

def extract_run_payload(run: Any) -> tuple[str, Mapping[str, Any]] | None:
    """Pull (jd_text, extraction) from a production `extract_one` trace Run.

    extract_one's inputs are {"posting": {...}} and outputs
    {"extractions": [<SkillExtraction dict>]}. Returns None when the run does not
    carry a usable JD + extraction (so the caller can skip it).
    """
    inputs = run.inputs or {}
    outputs = run.outputs or {}

    posting = inputs.get("posting") or {}
    jd = posting.get("description_text")

    extractions = outputs.get("extractions") or []
    if not jd or not extractions:
        return None

    extraction = extractions[0]
    if not isinstance(extraction, Mapping):
        return None
    return jd, extraction


def make_online_evaluator(
    judge_llm: Any, key: str = "extraction_quality",
) -> Callable[[Any], dict[str, Any] | None]:
    """Return an evaluator over a single production Run: run -> feedback dict | None.

    Returns None when the run isn't a scorable extraction (no JD/extraction),
    so the online loop can skip it without writing feedback.
    """
    def evaluator(run: Any) -> dict[str, Any] | None:
        payload = extract_run_payload(run)
        if payload is None:
            return None
        jd, extraction = payload
        verdict = judge_extraction(jd, extraction, judge_llm)
        return _feedback(verdict, key)

    return evaluator
