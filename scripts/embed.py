"""Embedding backfill entrypoint (RAG data plane).

Embeds new/edited postings and dossiers into their pgvector columns so the query
head can retrieve them. Deterministic and incremental (content-hash gated) — safe
to run every cycle; unchanged rows are skipped. Run after ingestion so the store
is populated and the schema (pgvector extension + embedding columns) is applied.

Usage:
    python scripts/embed.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

load_dotenv()

from ingestion.embed import run_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    counts = run_embeddings()
    log.info("Embedding run complete: %d postings, %d dossiers embedded",
             counts["postings"], counts["dossiers"])


if __name__ == "__main__":
    main()
