"""Unit tests for the tracker resolve_board node.

These exercise resolve_one — the pure core — with an injected probe and a fake
tool-calling model, plus the resolve_board node wrapper (cache short-circuit and
caching on success) against a fake connection. No network or live LLM is used.
"""
from __future__ import annotations

from contextlib import contextmanager

import pytest
from langchain_core.messages import AIMessage

from agents.tracker.state import BoardResolution
from agents.tracker import nodes
from agents.tracker.nodes import _seed_candidates, resolve_board, resolve_one, route_after_resolve


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

def make_probe(live: dict[str, tuple[str, str]]):
    """Build an async probe that returns a hit only for slugs in *live*.

    Also records, on the returned function, every slug it was asked about so
    tests can assert the fast-path ran (or didn't reach the LLM).
    """
    seen: list[str] = []

    async def probe(slug: str):
        seen.append(slug)
        return live.get(slug)

    probe.seen = seen  # type: ignore[attr-defined]
    return probe


class FakeBoundModel:
    """Stand-in for `llm.bind_tools([...])`. Replays a scripted list of slugs.

    Each entry in *slug_script* becomes one AIMessage with a single ProbeCandidate
    tool call. Once the script is exhausted it returns a plain message with no
    tool calls (the model "giving up").
    """

    def __init__(self, slug_script: list[str]) -> None:
        self._script = list(slug_script)
        self.invocations = 0

    async def ainvoke(self, messages):
        self.invocations += 1
        if not self._script:
            return AIMessage(content="I could not determine the board.")
        slug = self._script.pop(0)
        return AIMessage(
            content="",
            tool_calls=[{
                "name": "ProbeCandidate",
                "args": {"slug": slug},
                "id": f"call_{self.invocations}",
                "type": "tool_call",
            }],
        )


class FakeLLM:
    def __init__(self, slug_script: list[str]) -> None:
        self._bound = FakeBoundModel(slug_script)

    def bind_tools(self, tools):
        return self._bound


# ---------------------------------------------------------------------------
# Deterministic fast-path — no LLM needed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hint_slug_resolves_without_llm():
    company = {"name": "Cognition", "slug": "cognition", "ats_slug": "cognitionlabs"}
    probe = make_probe({"cognitionlabs": ("greenhouse", "cognitionlabs")})
    llm = FakeLLM([])  # must never be invoked

    res = await resolve_one(company, probe=probe, llm=llm)

    assert res.resolved is True
    assert (res.ats, res.ats_slug) == ("greenhouse", "cognitionlabs")
    assert res.method == "hint"
    assert res.board_url == "https://boards.greenhouse.io/cognitionlabs"
    assert llm._bound.invocations == 0


@pytest.mark.asyncio
async def test_name_derived_slug_resolves_without_llm():
    company = {"name": "Scale AI", "slug": "scaleai"}
    probe = make_probe({"scaleai": ("greenhouse", "scaleai")})
    llm = FakeLLM([])

    res = await resolve_one(company, probe=probe, llm=llm)

    assert res.resolved is True
    assert res.method == "deterministic"
    assert llm._bound.invocations == 0


# ---------------------------------------------------------------------------
# Agentic escalation — LLM proposes slugs, the probe verifies
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_escalation_finds_board():
    company = {"name": "Glean", "slug": "glean"}
    # Seed candidates ("glean") miss; the model's second guess hits.
    probe = make_probe({"gleanwork": ("greenhouse", "gleanwork")})
    llm = FakeLLM(["gleaninc", "gleanwork"])

    res = await resolve_one(company, probe=probe, llm=llm)

    assert res.resolved is True
    assert (res.ats, res.ats_slug) == ("greenhouse", "gleanwork")
    assert res.method == "agentic"
    assert llm._bound.invocations == 2
    # The verified hit short-circuits — every attempted slug is recorded for audit.
    assert res.attempted_slugs == ["glean", "gleaninc", "gleanwork"]


@pytest.mark.asyncio
async def test_unresolvable_company_returns_unresolved():
    company = {"name": "Stealth Co", "slug": "stealthco"}
    probe = make_probe({})  # nothing ever resolves
    llm = FakeLLM(["stealth", "stealthlabs", "stealthhq"])

    res = await resolve_one(company, probe=probe, llm=llm, max_probes=3)

    assert res.resolved is False
    assert res.ats is None
    assert res.method == "agentic"
    # Seed + three LLM guesses, all recorded, none duplicated.
    assert "stealthco" in res.attempted_slugs
    assert len(res.attempted_slugs) == len(set(res.attempted_slugs))


@pytest.mark.asyncio
async def test_max_probes_caps_llm_loop():
    company = {"name": "Stealth Co", "slug": "stealthco"}
    probe = make_probe({})
    # Script longer than max_probes; loop must stop early.
    llm = FakeLLM([f"guess{i}" for i in range(20)])

    res = await resolve_one(company, probe=probe, llm=llm, max_probes=2)

    assert res.resolved is False
    assert llm._bound.invocations == 2


@pytest.mark.asyncio
async def test_duplicate_slug_guesses_are_not_reprobed():
    company = {"name": "Repeat Co", "slug": "repeatco"}
    probe = make_probe({"realslug": ("lever", "realslug")})
    # Model repeats a dead guess before finding the live one.
    llm = FakeLLM(["dead", "dead", "realslug"])

    res = await resolve_one(company, probe=probe, llm=llm)

    assert res.resolved is True
    # "dead" probed once despite two guesses; "repeatco" seed + "dead" + "realslug".
    assert probe.seen.count("dead") == 1  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def test_route_continues_when_resolved():
    state = {"resolution": BoardResolution(company_slug="x", resolved=True,
                                           ats="lever", ats_slug="x")}
    assert route_after_resolve(state) == "fetch_signals"


def test_route_skips_when_unresolved():
    state = {"resolution": BoardResolution(company_slug="x", resolved=False)}
    assert route_after_resolve(state) == "__end__"


def test_route_skips_when_missing():
    assert route_after_resolve({}) == "__end__"


# ---------------------------------------------------------------------------
# _seed_candidates — deterministic fast-path ordering
# ---------------------------------------------------------------------------

def test_seed_candidates_hint_first_then_slug():
    # Hint differs from the internal key; name-derived slug equals the key (deduped).
    cands = _seed_candidates({"name": "Cognition", "slug": "cognition",
                              "ats_slug": "cognitionlabs"})
    assert cands == ["cognitionlabs", "cognition"]


def test_seed_candidates_collapse_when_all_equal():
    cands = _seed_candidates({"name": "Scale AI", "slug": "scaleai", "ats_slug": "scaleai"})
    assert cands == ["scaleai"]


def test_seed_candidates_no_hint_uses_slug_and_name():
    cands = _seed_candidates({"name": "Hugging Face", "slug": "huggingface"})
    assert cands == ["huggingface"]   # "Hugging Face" -> "huggingface" == slug, deduped


# ---------------------------------------------------------------------------
# resolve_board node wrapper — cache short-circuit and caching on success
# ---------------------------------------------------------------------------

class _FakeCursor:
    def __init__(self, one):
        self._one = one

    def fetchone(self):
        return self._one


class _FakeConn:
    def __init__(self, cached=None):
        self.cached = cached
        self.commits = 0

    def execute(self, sql, params=None):
        return _FakeCursor(self.cached)

    def commit(self):
        self.commits += 1


class _ExplodingLLM:
    """Fails loudly if the cache short-circuit ever reaches the LLM."""

    def bind_tools(self, tools):  # pragma: no cover - must never run
        raise AssertionError("LLM must not be invoked on a cache hit")


def _patch_conn(monkeypatch, conn):
    @contextmanager
    def _cm():
        yield conn

    monkeypatch.setattr(nodes, "get_connection", _cm)


@pytest.mark.asyncio
async def test_resolve_board_cache_hit_skips_probe_and_llm(monkeypatch):
    _patch_conn(monkeypatch, _FakeConn(cached={"ats": "lever", "ats_slug": "acme-co"}))
    monkeypatch.setattr(nodes, "RESOLVE_LLM", _ExplodingLLM())

    async def _boom(*a, **k):  # probing must not happen on a cache hit
        raise AssertionError("probe_ats must not be called on a cache hit")

    monkeypatch.setattr(nodes, "probe_ats", _boom)

    out = await resolve_board({"company": {"slug": "acme", "name": "Acme"}})

    res = out["resolution"]
    assert res.resolved is True
    assert res.method == "cache"
    assert (res.ats, res.ats_slug) == ("lever", "acme-co")
    assert res.board_url == "https://jobs.lever.co/acme-co"


@pytest.mark.asyncio
async def test_resolve_board_caches_on_success(monkeypatch):
    conn = _FakeConn(cached=None)   # not yet cached
    _patch_conn(monkeypatch, conn)

    async def fake_probe(slug, client):
        return ("greenhouse", "acme") if slug == "acme" else None

    monkeypatch.setattr(nodes, "probe_ats", fake_probe)

    upserts: list[tuple] = []
    monkeypatch.setattr(
        nodes, "upsert_company",
        lambda slug, name, ats, ats_slug, c, **kw: upserts.append((slug, ats, ats_slug)),
    )

    out = await resolve_board({"company": {"slug": "acme", "name": "Acme"}})

    res = out["resolution"]
    assert res.resolved is True
    assert res.method == "deterministic"   # resolved on the fast-path, no LLM
    assert upserts == [("acme", "greenhouse", "acme")]   # cached for next time
    assert conn.commits == 1


@pytest.mark.asyncio
async def test_resolve_board_unresolved_is_not_cached(monkeypatch):
    conn = _FakeConn(cached=None)
    _patch_conn(monkeypatch, conn)

    async def fake_probe(slug, client):
        return None   # nothing ever resolves

    monkeypatch.setattr(nodes, "probe_ats", fake_probe)
    # LLM gives up immediately (no tool calls).
    monkeypatch.setattr(nodes, "RESOLVE_LLM", FakeLLM([]))

    def _must_not_cache(*a, **k):
        raise AssertionError("an unresolved company must not be cached")

    monkeypatch.setattr(nodes, "upsert_company", _must_not_cache)

    out = await resolve_board({"company": {"slug": "ghost", "name": "Ghost Co"}})

    assert out["resolution"].resolved is False
    assert conn.commits == 0   # nothing written
