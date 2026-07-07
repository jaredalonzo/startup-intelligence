"""FastAPI app factory for the query head serving surface.

Construction (settings, limiter, semaphore) happens in the factory so tests
can drive the app through httpx's ASGITransport without lifespan plumbing;
the lifespan does only side-effectful startup (tracing init). There is
deliberately no module-level `app`: that would read env at import time —
scripts/serve.py uses uvicorn's factory mode instead.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
import psycopg
from fastapi import Depends, FastAPI, HTTPException, Request

# Imported into this namespace (not called via the module) so tests can
# monkeypatch `serving.app.run_query` with a stub.
from agents.query.run import run_query
from observability import CostGuard, init_tracing
from serving.auth import require_api_key
from serving.limits import TokenBucket
from serving.schemas import Citation, QueryRequest, QueryResponse, QueryUsage
from serving.settings import ServingSettings

log = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_tracing()  # no-op without LANGSMITH_API_KEY; idempotent
    s: ServingSettings = app.state.settings
    log.info(
        "serving: query head up (timeout=%.0fs, rate=%s/min burst=%d, concurrency=%d)",
        s.timeout_s, s.rate_per_minute, s.rate_burst, s.max_concurrency,
    )
    yield  # nothing to close — retrieval opens per-request store connections


def create_app(settings: ServingSettings | None = None) -> FastAPI:
    settings = settings or ServingSettings.from_env()
    app = FastAPI(title="startup-intelligence query head", lifespan=_lifespan)
    app.state.settings = settings
    app.state.bucket = TokenBucket(settings.rate_per_minute, settings.rate_burst)
    # Caps concurrent graph runs toward the LLM backend (see config.py note).
    app.state.llm_semaphore = asyncio.Semaphore(settings.max_concurrency)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        # Liveness only — no DB/Ollama probing (a readiness probe that opens
        # connections on every poll is a cost, not a signal, at this scale).
        return {"status": "ok"}

    @app.post("/query", dependencies=[Depends(require_api_key)])
    async def post_query(body: QueryRequest, request: Request) -> QueryResponse:
        st = request.app.state
        allowed, retry_after = st.bucket.try_acquire()
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="rate limit exceeded",
                headers={"Retry-After": str(math.ceil(retry_after))},
            )

        request_id = uuid4().hex[:12]
        guard = CostGuard(st.settings.call_budget, st.settings.token_budget)
        started = time.monotonic()
        try:
            # Semaphore inside the timeout: a request stuck waiting for its
            # LLM slot still times out cleanly and releases nothing it holds.
            async with asyncio.timeout(st.settings.timeout_s):
                async with st.llm_semaphore:
                    result = await run_query(
                        body.question,
                        callbacks=[guard],
                        run_name="query_api",
                        metadata={"request_id": request_id},
                    )
        except TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"query timed out after {st.settings.timeout_s:.0f}s",
            ) from None
        except psycopg.OperationalError:
            raise HTTPException(status_code=503, detail="store unreachable") from None
        except (httpx.HTTPError, ConnectionError):
            # Mirrors the CLI's error taxonomy (scripts/query.py): the Ollama
            # backend is down or unreachable.
            raise HTTPException(status_code=502, detail="LLM backend unreachable") from None

        duration_ms = int((time.monotonic() - started) * 1000)
        postings = result["postings"] or []
        dossiers = result["dossiers"] or []
        log.info(
            "query %s: %dms, %d llm calls, %d tokens, fallback=%s, docs=%d",
            request_id, duration_ms, guard.calls, guard.tokens,
            result["parse_fallback"], len(postings) + len(dossiers),
        )
        return QueryResponse(
            answer_markdown=result["answer_markdown"] or "",
            abstained=not postings and not dossiers,
            parse_fallback=result["parse_fallback"],
            citations=[Citation.from_doc(d) for d in [*postings, *dossiers]],
            request_id=request_id,
            usage=QueryUsage(
                llm_calls=guard.calls, llm_tokens=guard.tokens, duration_ms=duration_ms
            ),
        )

    return app
