"""Unit tests for the provider-robust structured/tool-calling helpers.

The ChatOllama raw-client paths are integration-verified live; here we cover the
pure pieces with no network: provider dispatch (non-Ollama → LangChain) and the
message normalization that feeds the raw client.
"""
from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_structured import _to_ollama_messages, bind_tools, structured  # noqa: E402


class _Schema(BaseModel):
    x: int


class _FakeLLM:
    """Non-ChatOllama model: records that the LangChain path was taken."""

    def with_structured_output(self, schema):
        return ("wso", schema)

    def bind_tools(self, tools):
        return ("bound", tools)


# ---------------------------------------------------------------------------
# Dispatch — non-Ollama models keep the LangChain path
# ---------------------------------------------------------------------------

def test_structured_uses_langchain_for_non_ollama():
    assert structured(_FakeLLM(), _Schema) == ("wso", _Schema)


def test_bind_tools_uses_langchain_for_non_ollama():
    assert bind_tools(_FakeLLM(), [_Schema]) == ("bound", [_Schema])


# ---------------------------------------------------------------------------
# Message normalization
# ---------------------------------------------------------------------------

def test_to_ollama_messages_maps_roles_and_dicts():
    msgs = _to_ollama_messages([
        {"role": "system", "content": "s"},
        SystemMessage(content="sys"),
        HumanMessage(content="hi"),
    ])
    assert [m["role"] for m in msgs] == ["system", "system", "user"]
    assert msgs[2]["content"] == "hi"


def test_to_ollama_messages_carries_tool_calls_and_results():
    ai = AIMessage(content="", tool_calls=[
        {"name": "ProbeCandidate", "args": {"slug": "acme"}, "id": "c1", "type": "tool_call"},
    ])
    tool = ToolMessage(content="No board for 'acme'.", tool_call_id="c1")
    msgs = _to_ollama_messages([ai, tool])

    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["tool_calls"][0]["function"] == {"name": "ProbeCandidate", "arguments": {"slug": "acme"}}
    assert msgs[1]["role"] == "tool"
    assert msgs[1]["content"] == "No board for 'acme'."
