"""Model bake-off for the skills-extraction task (observability "Mode B").

Runs a *fixed* sample of real postings through the **exact production extraction**
(agents.skills.nodes.extract_posting_fields) under each candidate model, scores
each with an LLM-as-judge, and records everything as LangSmith Experiments —
where latency and token usage are captured automatically per example. The result
is a side-by-side quality x tokens x latency comparison across models.

Prerequisites: LANGSMITH_API_KEY set (results live in LangSmith), DATABASE_URL,
and the candidate models available (local Ollama models pulled, or ANTHROPIC_API_KEY
for claude-* candidates).

Usage:
    # 1. build the fixed evaluation dataset once (samples postings from the store)
    python scripts/eval_models.py --build-dataset --sample 30

    # 2. run the bake-off across models (one LangSmith Experiment each)
    python scripts/eval_models.py --models qwen2.5:14b llama3.1:8b
    python scripts/eval_models.py --models qwen2.5:14b claude-haiku-4-5-20251001 \
        --judge-model claude-sonnet-4-6
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import ChatOllama
from langsmith import Client, evaluate
from pydantic import BaseModel

from agents.skills.nodes import extract_posting_fields
from store.db import get_connection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)

_DATASET = "skills-extraction-eval"


def _build_llm(model: str) -> Any:
    """Construct a chat model by name. claude-* → Anthropic (lazy import); else Ollama.

    temperature=0 for determinism so the comparison reflects the model, not sampling."""
    if model.startswith("claude"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=0, max_tokens=1024)  # type: ignore[call-arg]
    return ChatOllama(model=model, temperature=0)


# ---------------------------------------------------------------------------
# Dataset — a fixed sample of real postings (reference-free; judged on the JD text)
# ---------------------------------------------------------------------------

def build_dataset(sample: int) -> None:
    client = Client()
    if client.has_dataset(dataset_name=_DATASET):
        log.warning("Dataset %r already exists — leaving it as-is (delete it in the UI to "
                    "rebuild). Skipping.", _DATASET)
        return

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT title, department, team, description_text
            FROM postings
            WHERE description_text IS NOT NULL AND description_text <> ''
            ORDER BY first_seen_at DESC
            LIMIT %s
            """,
            (sample,),
        ).fetchall()

    if not rows:
        log.error("No postings with description_text found — run ingestion first.")
        return

    ds = client.create_dataset(_DATASET, description="Fixed posting sample for model bake-off")
    client.create_examples(
        dataset_id=ds.id,
        examples=[
            {"inputs": {"title": r["title"], "department": r["department"],
                        "team": r["team"], "description_text": r["description_text"]}}
            for r in rows
        ],
    )
    log.info("Built dataset %r with %d examples.", _DATASET, len(rows))


# ---------------------------------------------------------------------------
# Target — the production extraction, parametrized by model
# ---------------------------------------------------------------------------

def make_target(model: str):
    """Return a LangSmith target fn that runs the real extraction with `model`."""
    llm = _build_llm(model)

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        return extract_posting_fields(inputs, llm).model_dump()

    return target


# ---------------------------------------------------------------------------
# Evaluator — LLM-as-judge for extraction quality (latency/tokens are auto-captured)
# ---------------------------------------------------------------------------

class _Judgment(BaseModel):
    score: float        # 0.0 (wrong/empty) .. 1.0 (accurate and complete)
    rationale: str


def make_quality_judge(judge_model: str):
    judge = _build_llm(judge_model).with_structured_output(_Judgment)

    def quality_judge(run: Any, example: Any) -> dict[str, Any]:
        """Score how well the extraction reflects the posting. Reference-free:
        judges the extracted skills/platforms against the JD text itself."""
        out = run.outputs or {}
        jd = example.inputs.get("description_text", "")[:6000]
        verdict: _Judgment = judge.invoke([
            {"role": "system", "content": (
                "You grade skill-extraction quality from job postings. Given the JD text and a "
                "candidate extraction, score 0-1: are the listed skills/platforms actually "
                "required by the JD (precision), are obvious ones missing (recall), are names "
                "canonical, and is seniority right? Penalize hallucinated or generic entries."
            )},
            {"role": "user", "content": (
                f"JD:\n{jd}\n\nEXTRACTION:\n"
                f"skills={out.get('skills')}\nplatforms={out.get('platforms')}\n"
                f"seniority={out.get('seniority')}\nyears_experience={out.get('years_experience')}"
            )},
        ])
        return {"key": "extraction_quality", "score": verdict.score, "comment": verdict.rationale}

    return quality_judge


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Bake off LLM models on skills extraction.")
    parser.add_argument("--build-dataset", action="store_true",
                        help="Sample postings from the store into the LangSmith dataset, then exit.")
    parser.add_argument("--sample", type=int, default=30, help="Dataset size when building.")
    parser.add_argument("--models", nargs="+", default=["qwen2.5:14b"],
                        help="Candidate models to compare (one Experiment each).")
    parser.add_argument("--judge-model", default="qwen2.5:14b",
                        help="Model used as the quality judge (a stronger model is recommended).")
    args = parser.parse_args()

    if not os.environ.get("LANGSMITH_API_KEY"):
        log.error("LANGSMITH_API_KEY is required — eval results are recorded in LangSmith.")
        sys.exit(1)

    if args.build_dataset:
        build_dataset(args.sample)
        return

    judge = make_quality_judge(args.judge_model)
    for model in args.models:
        log.info("Evaluating model %r (judge=%r)…", model, args.judge_model)
        # max_concurrency=1: serialize calls so a queued local model (OLLAMA_NUM_PARALLEL=1)
        # doesn't inflate per-example latency — keeps the latency comparison fair.
        results = evaluate(
            make_target(model),
            data=_DATASET,
            evaluators=[judge],
            experiment_prefix=f"skills-extract-{model}",
            metadata={"model": model, "judge_model": args.judge_model},
            max_concurrency=1,
        )
        log.info("Done: %s", getattr(results, "experiment_name", model))


if __name__ == "__main__":
    main()
