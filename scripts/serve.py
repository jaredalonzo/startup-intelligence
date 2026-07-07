"""Serve the query head over HTTP — the first always-on component (M-RAG.3).

Read-only per-request question answering over the corpus. Requires
QUERY_API_KEY in the environment (the server refuses to start without it).

Usage:
    python scripts/serve.py                # 127.0.0.1:8100
    python scripts/serve.py --port 9000
    python scripts/serve.py --reload       # dev autoreload

Backends: same as scripts/query.py — a LOCAL Ollama daemon for the query
embedding, and the pinned answer model (gpt-oss:120b on Ollama Cloud by
default; override QUERY_ANSWER_MODEL).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import uvicorn
from dotenv import load_dotenv

load_dotenv()

from serving.settings import ServingSettings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the query head over HTTP (read-only).")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default localhost)")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--reload", action="store_true", help="dev autoreload")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    # Fail fast on a missing QUERY_API_KEY before uvicorn forks anything.
    ServingSettings.from_env()
    # workers=1 is load-bearing: the rate limiter and the LLM concurrency
    # semaphore are in-process; more workers would silently multiply both
    # limits. A multi-worker deployment needs an external limiter first.
    uvicorn.run(
        "serving.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1,
    )


if __name__ == "__main__":
    main()
