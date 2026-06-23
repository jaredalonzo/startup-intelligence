"""Unit tests for the tracker resolve_board node.

These exercise resolve_one — the pure core — with an injected probe and a fake
tool-calling model, so no network or live LLM is required. The node wrapper
(resolve_board) is exercised separately for its cache short-circuit.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from agents.state import BoardResolution
from agents.tracker.nodes import resolve_one, route_after_resolve


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
