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
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from langsmith import Client, evaluate

from agents.skills.nodes import extract_posting_fields
from config import EVAL_JUDGE_MODEL
from eval.extraction_quality import make_offline_evaluator
from eval.llm import build_llm
from store.db import get_connection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)

_DATASET = "skills-extraction-eval"
# Postings shorter than this are stubs/redirects (e.g. "We have moved our Careers
# Page to …"), not real JDs — they would poison the extraction eval. Real JDs run
# into the thousands of characters.
_MIN_DESC_CHARS = 300


# ---------------------------------------------------------------------------
# Dataset — a fixed sample of real postings (reference-free; judged on the JD text)
#
# The eval is only meaningful if its distribution matches what the skills agent
# actually extracts over: engineering + FDE/TAM/CSE/implementation roles. A naive
# "newest N postings" sample is dominated by sales/recruiting/marketing and by
# whichever company just opened the most reqs — junk in, junk out. So selection is:
#   1. a deterministic technical-role gate (title-driven; the customer-facing
#      technical roles live under non-Engineering departments, so department alone
#      is not enough — and an exclude pass kills GTM/ops noise),
#   2. dedup by (company, title) to drop reposts, and
#   3. round-robin across companies so no single company dominates the sample.
# All deterministic — dataset construction stays out of the LLM (repo principle).
# ---------------------------------------------------------------------------

# Title/department signals that a posting is an engineering or technical-field role.
_TECH_INCLUDE = re.compile(
    r"\b(engineer|engineering|developer|forward[\s-]?deployed|solutions?\s+architect|"
    r"solutions?\s+engineer|implementation|technical\s+(account|deployment|solutions)|"
    r"platform|infrastructure|data\s+(scientist|engineer)|machine\s+learning|\bml\b|"
    r"\bai\b|security|devops|\bsre\b|reliability|architect|deploy(ment)?|"
    r"research\s+(engineer|scientist|intern))\b",
    re.IGNORECASE,
)
# Non-technical signals — checked first, so "Developer Marketing" / "Technical
# Program Manager" are rejected even though they trip the include pass.
_NONTECH_EXCLUDE = re.compile(
    r"\b(sales|marketing|market\b|finance|financial|legal|counsel|recruit|recruiting|"
    r"people\b|talent|account\s+executive|\bsdr\b|\bbdr\b|communications|social\s+media|"
    r"events?\s+manager|field\s+marketer|deal\s+strategy|storytelling|"
    r"program\s+manager|product\s+manager|strategist)\b",
    re.IGNORECASE,
)


# Placeholder "postings" whose title is a notice, not a role (e.g. an ATS that left
# a "We have moved our Careers Page to <url>" stub with only company boilerplate as
# the body). These slip the length guard but have no real JD to extract from.
_STUB_TITLE = re.compile(r"we have moved|careers?\s+page|https?://|board\s+has\s+moved",
                         re.IGNORECASE)


def _is_technical(title: str | None, department: str | None) -> bool:
    """Deterministic gate: keep eng + technical-field roles, drop GTM/ops/HR/stubs."""
    title = title or ""
    if _STUB_TITLE.search(title):
        return False
    text = f"{title} {department or ''}"
    if _NONTECH_EXCLUDE.search(text):
        return False
    return bool(_TECH_INCLUDE.search(text))


def _select_postings(rows: list[Any], sample: int) -> list[Any]:
    """Filter to technical roles, dedup reposts, then round-robin across companies.

    `rows` must be ordered most-recent-first so each company contributes its newest
    postings; round-robin then keeps the sample broad across the watchlist.
    """
    seen: set[tuple[str, str]] = set()
    by_company: dict[str, list[Any]] = defaultdict(list)
    for r in rows:
        if not _is_technical(r["title"], r["department"]):
            continue
        key = (r["company_slug"], (r["title"] or "").strip().lower())
        if key in seen:
            continue
        seen.add(key)
        by_company[r["company_slug"]].append(r)

    picked: list[Any] = []
    depth = 0
    while len(picked) < sample:
        added = False
        for company in sorted(by_company):
            pool = by_company[company]
            if depth < len(pool):
                picked.append(pool[depth])
                added = True
                if len(picked) >= sample:
                    break
        if not added:
            break  # every company's pool exhausted
        depth += 1
    return picked


def build_dataset(sample: int) -> None:
    client = Client()
    if client.has_dataset(dataset_name=_DATASET):
        log.warning("Dataset %r already exists — leaving it as-is (delete it in the UI to "
                    "rebuild). Skipping.", _DATASET)
        return

    # Pull a light candidate pool (no description_text) to filter/stratify cheaply,
    # then fetch full descriptions only for the postings we actually keep.
    with get_connection() as conn:
        candidates = conn.execute(
            """
            SELECT id, title, department, team, company_slug
            FROM postings
            WHERE description_text IS NOT NULL
              AND length(description_text) >= %s
            ORDER BY first_seen_at DESC
            """,
            (_MIN_DESC_CHARS,),
        ).fetchall()

        if not candidates:
            log.error("No postings with description_text found — run ingestion first.")
            return

        chosen = _select_postings(candidates, sample)
        if not chosen:
            log.error("No technical postings matched the role filter — check the corpus.")
            return

        rows = conn.execute(
            """
            SELECT id, title, department, team, description_text
            FROM postings
            WHERE id = ANY(%s)
            """,
            ([r["id"] for r in chosen],),
        ).fetchall()

    by_id = {r["id"]: r for r in rows}
    ordered = [by_id[r["id"]] for r in chosen]  # preserve the stratified order

    log.info("Selected %d technical postings from %d candidates across %d companies:",
             len(ordered), len(candidates), len({r["company_slug"] for r in chosen}))
    for r, c in zip(ordered, chosen):
        log.info("  [%-14s] %s", c["company_slug"][:14], r["title"])

    ds = client.create_dataset(_DATASET, description="Fixed technical-posting sample for "
                               "model bake-off (role-filtered, deduped, company-stratified)")
    client.create_examples(
        dataset_id=ds.id,
        examples=[
            {"inputs": {"title": r["title"], "department": r["department"],
                        "team": r["team"], "description_text": r["description_text"]}}
            for r in ordered
        ],
    )
    log.info("Built dataset %r with %d examples.", _DATASET, len(ordered))


# ---------------------------------------------------------------------------
# Target — the production extraction, parametrized by model
# ---------------------------------------------------------------------------

def make_target(model: str):
    """Return a LangSmith target fn that runs the real extraction with `model`."""
    llm = build_llm(model)

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        return extract_posting_fields(inputs, llm).model_dump()

    return target


# The LLM-as-judge for extraction quality lives in eval.extraction_quality so it
# is shared with the online evaluator (scripts/online_eval.py). Latency and token
# usage are captured automatically by LangSmith per example.

# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Bake off LLM models on skills extraction.")
    parser.add_argument("--build-dataset", action="store_true",
                        help="Sample postings from the store into the LangSmith dataset, then exit.")
    parser.add_argument("--sample", type=int, default=30, help="Dataset size when building.")
    parser.add_argument("--models", nargs="+", default=["qwen2.5:14b"],
                        help="Candidate models to compare (one Experiment each).")
    parser.add_argument("--judge-model", default=EVAL_JUDGE_MODEL,
                        help="Model used as the quality judge (a stronger model is recommended).")
    args = parser.parse_args()

    if not os.environ.get("LANGSMITH_API_KEY"):
        log.error("LANGSMITH_API_KEY is required — eval results are recorded in LangSmith.")
        sys.exit(1)

    if args.build_dataset:
        build_dataset(args.sample)
        return

    judge = make_offline_evaluator(build_llm(args.judge_model))
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
