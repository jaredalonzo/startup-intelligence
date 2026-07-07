"""Query head entrypoint — ask a natural-language question over the corpus.

Read-only: parses the question into a plan (LLM tool-use), runs hybrid
retrieval (pgvector cosine + structured filters, deterministic), and
synthesizes a grounded, cited answer. Writes nothing to the store.

Usage:
    python scripts/query.py "which companies are hiring Rust and distributed systems engineers?"
    python scripts/query.py --show-plan --show-hits "senior remote FDE roles?"

Backends: query embedding needs a LOCAL Ollama daemon serving the configured
embedding model (`ollama serve` + `ollama pull <EMBEDDING_MODEL>`; Ollama Cloud
has no embedding models, so the embedding pin never uses it); answer synthesis
defaults to gpt-oss:120b on Ollama Cloud (OLLAMA_API_KEY) — override
QUERY_ANSWER_MODEL for a local-only setup.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import httpx
import psycopg
from dotenv import load_dotenv

load_dotenv()

from agents.query.run import run_query
from agents.query.state import QueryState
from config import EMBEDDING_MODEL_NAME, LLM_CALL_BUDGET_PER_RUN, LLM_TOKEN_BUDGET_PER_RUN
from observability import CostGuard, init_tracing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger(__name__)

init_tracing()

_BACKEND_HELP = (
    "The query head needs two Ollama backends: query embedding uses a LOCAL "
    f"daemon serving {EMBEDDING_MODEL_NAME} (`ollama serve` + `ollama pull "
    f"{EMBEDDING_MODEL_NAME}` — Ollama Cloud has no embedding models), and answer "
    "synthesis defaults to gpt-oss:120b on Ollama Cloud (needs OLLAMA_API_KEY; "
    "override QUERY_ANSWER_MODEL for a local model)."
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask a question over the corpus (read-only).")
    parser.add_argument("question", help="natural-language question")
    parser.add_argument(
        "--corpus",
        choices=["postings", "dossiers", "both"],
        default=None,
        help="force the corpus, overriding the parsed plan",
    )
    parser.add_argument(
        "--show-plan", action="store_true", help="print the parsed QueryPlan before the answer"
    )
    parser.add_argument(
        "--show-hits", action="store_true", help="print retrieved docs with distances"
    )
    return parser.parse_args()


def _print_plan(result: QueryState) -> None:
    plan = result["plan"]
    fallback = " (FALLBACK: parse failed, semantic-only)" if result["parse_fallback"] else ""
    print(f"--- plan{fallback} ---")
    if plan is not None:
        print(plan.model_dump_json(exclude_defaults=True, indent=2))
    print()


def _print_hits(result: QueryState) -> None:
    print("--- retrieved ---")
    for doc in (result["postings"] or []) + (result["dossiers"] or []):
        print(f"  {doc.distance:.4f}  [{doc.kind}] {doc.company_slug}: {doc.title}")
    print()


def main() -> None:
    args = _parse_args()
    guard = CostGuard(LLM_CALL_BUDGET_PER_RUN, LLM_TOKEN_BUDGET_PER_RUN)

    try:
        result = asyncio.run(run_query(args.question, corpus=args.corpus, callbacks=[guard]))
    except RuntimeError as exc:
        if "DATABASE_URL" in str(exc):
            log.error("DATABASE_URL is not set — point it at the shared Postgres (.env).")
            sys.exit(1)
        raise
    except psycopg.OperationalError as exc:
        log.error("could not reach the database: %s", exc)
        sys.exit(1)
    except (httpx.HTTPError, ConnectionError) as exc:
        # langchain_ollama raises builtin ConnectionError for a down daemon;
        # httpx errors surface from the raw-client (tool-use/structured) paths.
        log.error("Ollama backend unreachable (%s). %s", exc, _BACKEND_HELP)
        sys.exit(1)

    if args.show_plan:
        _print_plan(result)
    if args.show_hits:
        _print_hits(result)

    print(result["answer_markdown"] or "(no answer produced)")
    guard.log_summary()


if __name__ == "__main__":
    main()
