"""Tests for the HTTP serving surface (src/serving) — no LLM, no DB.

The graph run is stubbed by monkeypatching `serving.app.run_query` (imported
into the app module's namespace precisely so it is patchable there); requests
go through httpx's ASGITransport, so no real server or lifespan is needed.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import httpx  # noqa: E402
import psycopg  # noqa: E402

from agents.query.state import QueryState, RetrievedDoc  # noqa: E402
from serving import app as app_module  # noqa: E402
from serving.limits import TokenBucket  # noqa: E402
from serving.settings import ServingSettings  # noqa: E402

_KEY = "test-key"


def _settings(**over: Any) -> ServingSettings:
    defaults: dict[str, Any] = {
        "api_key": _KEY,
        "call_budget": 5,
        "token_budget": 1000,
        "timeout_s": 5.0,
        "rate_per_minute": 600.0,  # effectively unlimited unless a test tightens it
        "rate_burst": 100,
        "max_concurrency": 1,
    }
    defaults.update(over)
    return ServingSettings(**defaults)


def _client(settings: ServingSettings) -> httpx.AsyncClient:
    app = app_module.create_app(settings)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


def _doc(kind: str = "posting", title: str = "Rust Engineer") -> RetrievedDoc:
    return RetrievedDoc(
        kind=kind,  # type: ignore[arg-type]
        company_slug="acme",
        company_name="Acme",
        title=title,
        url="https://a/1",
        snippet="context the model saw",
        distance=0.2,
    )


def _state(
    postings: list[RetrievedDoc] | None = None,
    dossiers: list[RetrievedDoc] | None = None,
    answer: str = "Acme is hiring Rust engineers [1].",
    fallback: bool = False,
) -> QueryState:
    return {
        "question": "q?",
        "plan": None,
        "parse_fallback": fallback,
        "postings": postings if postings is not None else [],
        "dossiers": dossiers if dossiers is not None else [],
        "answer_markdown": answer,
    }


def _stub(monkeypatch: Any, result: QueryState) -> None:
    async def fake_run_query(question: str, **_kwargs: Any) -> QueryState:
        return result

    monkeypatch.setattr(app_module, "run_query", fake_run_query)


async def test_healthz_requires_no_auth() -> None:
    async with _client(_settings()) as client:
        resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_query_missing_key_is_401() -> None:
    async with _client(_settings()) as client:
        resp = await client.post("/query", json={"question": "q?"})
    assert resp.status_code == 401


async def test_query_wrong_key_is_401() -> None:
    async with _client(_settings()) as client:
        resp = await client.post(
            "/query", json={"question": "q?"}, headers={"X-API-Key": "nope"}
        )
    assert resp.status_code == 401


async def test_query_happy_path_shape(monkeypatch: Any) -> None:
    _stub(monkeypatch, _state(postings=[_doc()], dossiers=[_doc("dossier", "Acme dossier")]))
    async with _client(_settings()) as client:
        resp = await client.post(
            "/query", json={"question": "who hires rust?"}, headers={"X-API-Key": _KEY}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer_markdown"] == "Acme is hiring Rust engineers [1]."
    assert body["abstained"] is False
    assert body["parse_fallback"] is False
    # postings before dossiers — matches the answer's [n] numbering
    assert [c["kind"] for c in body["citations"]] == ["posting", "dossier"]
    assert body["citations"][0]["url"] == "https://a/1"
    assert "snippet" not in body["citations"][0]
    assert len(body["request_id"]) == 12
    assert body["usage"]["llm_calls"] == 0  # stub never fires callbacks
    assert body["usage"]["duration_ms"] >= 0


async def test_abstain_sets_abstained_true(monkeypatch: Any) -> None:
    _stub(monkeypatch, _state(answer="I can't answer that from the corpus."))
    async with _client(_settings()) as client:
        resp = await client.post(
            "/query", json={"question": "q?"}, headers={"X-API-Key": _KEY}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["abstained"] is True
    assert body["citations"] == []


async def test_empty_question_is_422() -> None:
    async with _client(_settings()) as client:
        resp = await client.post(
            "/query", json={"question": ""}, headers={"X-API-Key": _KEY}
        )
    assert resp.status_code == 422


async def test_rate_limit_trips(monkeypatch: Any) -> None:
    _stub(monkeypatch, _state())
    async with _client(_settings(rate_per_minute=1.0, rate_burst=1)) as client:
        first = await client.post(
            "/query", json={"question": "q?"}, headers={"X-API-Key": _KEY}
        )
        second = await client.post(
            "/query", json={"question": "q?"}, headers={"X-API-Key": _KEY}
        )
    assert first.status_code == 200
    assert second.status_code == 429
    assert int(second.headers["Retry-After"]) >= 1


async def test_timeout_maps_to_504(monkeypatch: Any) -> None:
    async def slow_run_query(question: str, **_kwargs: Any) -> QueryState:
        await asyncio.sleep(0.05)
        return _state()

    monkeypatch.setattr(app_module, "run_query", slow_run_query)
    async with _client(_settings(timeout_s=0.01)) as client:
        resp = await client.post(
            "/query", json={"question": "q?"}, headers={"X-API-Key": _KEY}
        )
    assert resp.status_code == 504


async def test_store_down_maps_to_503(monkeypatch: Any) -> None:
    async def broken_run_query(question: str, **_kwargs: Any) -> QueryState:
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(app_module, "run_query", broken_run_query)
    async with _client(_settings()) as client:
        resp = await client.post(
            "/query", json={"question": "q?"}, headers={"X-API-Key": _KEY}
        )
    assert resp.status_code == 503


async def test_llm_backend_down_maps_to_502(monkeypatch: Any) -> None:
    async def broken_run_query(question: str, **_kwargs: Any) -> QueryState:
        raise ConnectionError("ollama daemon down")

    monkeypatch.setattr(app_module, "run_query", broken_run_query)
    async with _client(_settings()) as client:
        resp = await client.post(
            "/query", json={"question": "q?"}, headers={"X-API-Key": _KEY}
        )
    assert resp.status_code == 502


def test_from_env_requires_api_key(monkeypatch: Any) -> None:
    import pytest

    monkeypatch.delenv("QUERY_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="QUERY_API_KEY"):
        ServingSettings.from_env()


# --------------------------------------------------------------------------
# TokenBucket unit tests (injected clock)
# --------------------------------------------------------------------------

def test_token_bucket_burst_then_refill() -> None:
    clock = [0.0]
    bucket = TokenBucket(rate_per_minute=60.0, burst=2, now=lambda: clock[0])  # 1 token/s

    assert bucket.try_acquire() == (True, 0.0)
    assert bucket.try_acquire() == (True, 0.0)
    allowed, retry_after = bucket.try_acquire()  # burst exhausted
    assert allowed is False
    assert 0.0 < retry_after <= 1.0

    clock[0] += 1.0  # one second refills exactly one token
    assert bucket.try_acquire() == (True, 0.0)
    assert bucket.try_acquire()[0] is False


def test_token_bucket_refill_caps_at_burst() -> None:
    clock = [0.0]
    bucket = TokenBucket(rate_per_minute=60.0, burst=2, now=lambda: clock[0])
    clock[0] += 100.0  # a long idle must not accumulate beyond burst
    assert bucket.try_acquire()[0] is True
    assert bucket.try_acquire()[0] is True
    assert bucket.try_acquire()[0] is False
