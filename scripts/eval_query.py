"""Query-head eval: retrieval recall + answer faithfulness over a fixed dataset.

Runs the **exact production query graph** once per example and scores each run
with two evaluators in one LangSmith Experiment:

  - query_retrieval    — deterministic recall against hand-labeled gold docs
  - query_faithfulness — LLM-as-judge grounding check (cite-or-abstain)

The dataset seed is checked into the repo (eval/query_dataset_seed.py) and
uploaded once with --build-dataset — versioned in git, experiments in LangSmith,
mirroring the extraction bake-off (scripts/eval_models.py).

Prerequisites: LANGSMITH_API_KEY, DATABASE_URL, and the query head's backends
(Ollama serving nomic-embed-text; OLLAMA_API_KEY for the default gpt-oss:120b
answer model).

Usage:
    python scripts/eval_query.py --build-dataset
    python scripts/eval_query.py
    python scripts/eval_query.py --judge-model claude-sonnet-4-6
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from langsmith import Client, evaluate

from agents.query.graph import compile_graph
from agents.query.nodes import describe_filters, format_context
from config import (
    EVAL_JUDGE_MODEL,
    QUERY_EVAL_FAITHFULNESS_KEY,
    QUERY_EVAL_RETRIEVAL_KEY,
)
from eval.llm import build_llm
from eval.query_dataset_seed import QUERY_EVAL_DATASET_NAME, QUERY_EVAL_SEED
from eval.query_quality import make_faithfulness_evaluator, make_retrieval_evaluator
from observability import init_tracing

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)

init_tracing()

_BACKEND = {"auto": None, "local": False, "cloud": True}  # CLI choice → build_llm cloud flag


def build_dataset() -> None:
    """Upload the checked-in seed to LangSmith (skip if the dataset exists)."""
    client = Client()
    if client.has_dataset(dataset_name=QUERY_EVAL_DATASET_NAME):
        log.warning(
            "Dataset %r already exists — leaving it as-is (delete it in the UI to "
            "rebuild). Skipping.", QUERY_EVAL_DATASET_NAME,
        )
        return

    ds = client.create_dataset(
        QUERY_EVAL_DATASET_NAME,
        description=(
            "Hand-built query-head eval: questions with gold (company_slug, "
            "title substring) retrieval labels; gold=[] marks abstain questions. "
            "Seed lives in src/eval/query_dataset_seed.py."
        ),
    )
    client.create_examples(
        dataset_id=ds.id,
        examples=[
            {
                "inputs": {"question": entry["question"]},
                "outputs": {"gold": entry["gold"], "notes": entry.get("notes", "")},
            }
            for entry in QUERY_EVAL_SEED
        ],
    )
    log.info("Built dataset %r with %d examples.", QUERY_EVAL_DATASET_NAME, len(QUERY_EVAL_SEED))


# One persistent event loop on a daemon thread for every example. asyncio.run()
# per example would close its loop each time while the module-level LLM clients
# cache async HTTP clients bound to the first loop — the second example then
# dies with "Event loop is closed". A single long-lived loop keeps them valid.
_LOOP = asyncio.new_event_loop()
threading.Thread(target=_LOOP.run_forever, daemon=True).start()

_GRAPH = compile_graph()


def target(inputs: dict[str, Any]) -> dict[str, Any]:
    """Run the production query graph on one question; emit what the evaluators need."""
    future = asyncio.run_coroutine_threadsafe(
        _GRAPH.ainvoke(
            {
                "question": inputs["question"],
                "plan": None,
                "parse_fallback": False,
                "postings": None,
                "dossiers": None,
                "answer_markdown": None,
            }
        ),
        _LOOP,
    )
    result = future.result()
    postings = result["postings"] or []
    dossiers = result["dossiers"] or []
    # The judge must see exactly what the answer node saw — including the
    # deterministic filters note, or it would penalize filter-attributed claims.
    context = format_context(postings, dossiers)
    filters_note = describe_filters(result["plan"])
    if filters_note:
        context = f"{filters_note}\n\n{context}"
    return {
        "answer_markdown": result["answer_markdown"],
        "retrieved": [d.model_dump(mode="json") for d in postings + dossiers],
        "context": context,
        "parse_fallback": result["parse_fallback"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the query head (retrieval + faithfulness).")
    parser.add_argument("--build-dataset", action="store_true",
                        help="Upload the checked-in seed to LangSmith, then exit.")
    parser.add_argument("--judge-model", default=EVAL_JUDGE_MODEL,
                        help="Faithfulness judge model (a stronger model is recommended).")
    parser.add_argument("--judge-backend", choices=("auto", "local", "cloud"), default="auto",
                        help="Ollama backend for the judge model (default: auto = key presence).")
    parser.add_argument("--label", default=None,
                        help="Optional free-form tag appended to the experiment name.")
    args = parser.parse_args()

    if not os.environ.get("LANGSMITH_API_KEY"):
        log.error("LANGSMITH_API_KEY is required — eval results are recorded in LangSmith.")
        sys.exit(1)

    if args.build_dataset:
        build_dataset()
        return

    answer_model = os.getenv("QUERY_ANSWER_MODEL", "gpt-oss:120b")
    prefix = f"query-head=answer[{answer_model}]|judge[{args.judge_model}]"
    if args.label:
        prefix += f"|{args.label}"

    judge_llm = build_llm(args.judge_model, cloud=_BACKEND[args.judge_backend])
    log.info("Evaluating query head (answer=%r, judge=%r)…", answer_model, args.judge_model)
    # max_concurrency=1: the graph serializes local Ollama embedding calls anyway,
    # and serialized runs keep per-example latency comparable across experiments.
    results = evaluate(
        target,
        data=QUERY_EVAL_DATASET_NAME,
        evaluators=[
            make_retrieval_evaluator(QUERY_EVAL_RETRIEVAL_KEY),
            make_faithfulness_evaluator(judge_llm, QUERY_EVAL_FAITHFULNESS_KEY),
        ],
        experiment_prefix=prefix,
        metadata={"answer_model": answer_model, "judge_model": args.judge_model,
                  "judge_backend": args.judge_backend, "label": args.label},
        max_concurrency=1,
    )
    log.info("Done: %s", getattr(results, "experiment_name", prefix))


if __name__ == "__main__":
    main()
