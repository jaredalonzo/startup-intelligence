"""Provider-robust structured output and tool-calling.

langchain_ollama (1.1.0) does not reliably drive Ollama's cloud API for either
feature: reasoning models such as gpt-oss return prose instead of honoring
``format=<schema>``, and ``bind_tools`` fails to surface the tool calls the
server actually emits. The raw ``ollama`` client handles both correctly, so for
ChatOllama we call it directly; every other provider keeps LangChain's path
(Anthropic tool-calling, etc.).

  - `structured(llm, schema)` → object with ``.invoke(messages) -> schema``.
  - `bind_tools(llm, tools)`  → object with ``.ainvoke(messages) -> AIMessage``
    (with ``.tool_calls``), matching what ``llm.bind_tools(...)`` returns.

Call sites stay provider-agnostic.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

_ROLE_MAP = {"system": "system", "human": "user", "ai": "assistant",
             "user": "user", "assistant": "assistant", "tool": "tool"}

# Some models wrap the JSON object in a Markdown code fence (```json ... ```)
# despite a format/schema directive — gemma3 does this. Extract the fenced body.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    """Return JSON unwrapped from a Markdown code fence if present, else trimmed."""
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _describe_fields(schema: dict[str, Any]) -> str:
    """Render a schema's fields as a 'name (type), ...' hint for the prompt.

    Used instead of dumping the full JSON Schema — some models echo a verbatim
    schema back as their answer. Optional fields (anyOf with null) read as their
    non-null type; unknown shapes fall back to 'any'.
    """
    def _type_of(spec: dict[str, Any]) -> str:
        if "type" in spec:
            return str(spec["type"])
        for branch in spec.get("anyOf", []):
            if isinstance(branch, dict) and branch.get("type") not in (None, "null"):
                return str(branch["type"])
        return "any"

    props = schema.get("properties", {})
    fields = ", ".join(f"{name} ({_type_of(spec)})" for name, spec in props.items())
    return fields or "the required fields"


def _example_object(schema: dict[str, Any]) -> str:
    """Render a placeholder JSON object showing the required keys and shape.

    Shows the object structure (so models emit a keyed object, not positional
    values or an echoed schema) without being the JSON Schema itself.
    """
    placeholder = {"array": "[]", "string": '"..."', "integer": "0",
                   "number": "0", "boolean": "true", "object": "{}"}

    def _type_of(spec: dict[str, Any]) -> str:
        if "type" in spec:
            return str(spec["type"])
        for branch in spec.get("anyOf", []):
            if isinstance(branch, dict) and branch.get("type") not in (None, "null"):
                return str(branch["type"])
        return "any"

    props = schema.get("properties", {})
    parts = [f'"{name}": {placeholder.get(_type_of(spec), "null")}'
             for name, spec in props.items()]
    return "{" + ", ".join(parts) + "}"


def _is_chat_ollama(llm: Any) -> bool:
    return type(llm).__name__ == "ChatOllama"


def _to_ollama_messages(messages: Any) -> list[dict[str, Any]]:
    """Normalize dict messages or LangChain BaseMessages to ollama chat dicts.

    Carries assistant ``tool_calls`` and ``tool`` results so multi-turn
    tool-calling history round-trips correctly.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, dict):
            msg: dict[str, Any] = {"role": _ROLE_MAP.get(m["role"], m["role"]),
                                   "content": m.get("content", "")}
            if m.get("tool_calls"):
                msg["tool_calls"] = m["tool_calls"]
        else:  # LangChain BaseMessage
            msg = {"role": _ROLE_MAP.get(m.type, "user"), "content": m.content or ""}
            tool_calls = getattr(m, "tool_calls", None)
            if tool_calls:
                msg["tool_calls"] = [
                    {"type": "function",
                     "function": {"name": tc["name"], "arguments": tc["args"]}}
                    for tc in tool_calls
                ]
        out.append(msg)
    return out


def _ollama_client(llm: Any) -> Any:
    """Build a raw ollama client from a ChatOllama's host + auth headers."""
    import ollama
    return ollama.Client(host=llm.base_url, **(llm.client_kwargs or {}))


def _options(llm: Any) -> dict[str, Any]:
    opts: dict[str, Any] = {}
    if getattr(llm, "temperature", None) is not None:
        opts["temperature"] = llm.temperature
    return opts


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------

class _OllamaStructured:
    """Structured output via the raw ollama client (native ``format`` enforcement)."""

    def __init__(self, llm: Any, schema: type[BaseModel]) -> None:
        self._client = _ollama_client(llm)
        self._model = llm.model
        self._schema = schema
        self._options = _options(llm)

    def invoke(self, messages: Any) -> BaseModel:
        schema = self._schema.model_json_schema()
        # gpt-oss (and other reasoning models) treat ``format`` as best-effort and
        # otherwise emit prose/markdown, so pair it with an explicit JSON-only
        # directive placed just before the final turn — the combination reliably
        # yields a parseable object. Describe the fields by name/type rather than
        # dumping the JSON Schema: handed the schema object verbatim, some models
        # (gemma3) echo it back instead of producing an instance.
        directive = {
            "role": "system",
            "content": "Output ONLY a single JSON object — no prose, no markdown, no "
                       "code fences. Keys and types: " + _describe_fields(schema)
                       + ". Return one object with exactly these keys and real values, "
                         "not positional values and not this template: "
                       + _example_object(schema)
                       + ". Emit valid JSON — escape any double-quote (\\\") or newline "
                         "inside string values.",
        }
        msgs = _to_ollama_messages(messages)
        msgs.insert(max(len(msgs) - 1, 0), directive)
        resp = self._client.chat(
            model=self._model, messages=msgs, format=schema, options=self._options,
        )
        return self._parse(resp["message"]["content"])

    def _parse(self, content: str) -> BaseModel:
        """Validate model output into the schema, tolerating common deviations.

        Handles raw JSON, Markdown-fenced JSON (gemma3), trailing characters after
        the object, and an instance nested one level inside an envelope object.
        Genuinely non-conforming output (e.g. a model echoing the JSON schema)
        still raises, so the caller's retry/degrade path engages.
        """
        text = _strip_code_fences(content)
        try:
            return self._schema.model_validate_json(text)
        except (ValidationError, ValueError):
            pass
        # Decode just the first JSON value, tolerating trailing characters (extra
        # prose, a second object, positional leftovers) after a complete object.
        try:
            obj, _ = json.JSONDecoder().raw_decode(text.lstrip())
        except ValueError:
            try:
                obj = json.loads(text)
            except ValueError:
                return self._schema.model_validate_json(text)  # re-raise informative error
        try:
            return self._schema.model_validate(obj)
        except ValidationError:
            pass
        if isinstance(obj, dict):
            wrappers = [obj[k] for k in ("properties", "data", "result", "output", "value")
                        if isinstance(obj.get(k), dict)]
            for candidate in wrappers + [v for v in obj.values() if isinstance(v, dict)]:
                try:
                    return self._schema.model_validate(candidate)
                except ValidationError:
                    continue
        return self._schema.model_validate(obj)  # raise a clear validation error


def structured(llm: Any, schema: type[BaseModel]) -> Any:
    """Return an object with ``.invoke(messages) -> schema`` for the given model.

    ChatOllama → native ollama structured outputs; otherwise LangChain's
    ``with_structured_output`` (JSON-schema / tool-calling path).
    """
    if _is_chat_ollama(llm):
        return _OllamaStructured(llm, schema)
    return llm.with_structured_output(schema)


# ---------------------------------------------------------------------------
# Tool-calling
# ---------------------------------------------------------------------------

class _OllamaToolCaller:
    """Tool-calling via the raw ollama client, returning a LangChain AIMessage."""

    def __init__(self, llm: Any, tools: list[Any]) -> None:
        from langchain_core.utils.function_calling import convert_to_openai_tool
        self._client = _ollama_client(llm)
        self._model = llm.model
        self._tools = [convert_to_openai_tool(t) for t in tools]
        self._options = _options(llm)

    async def ainvoke(self, messages: Any) -> Any:
        return await asyncio.to_thread(self._invoke, messages)

    def _invoke(self, messages: Any) -> Any:
        from langchain_core.messages import AIMessage
        resp = self._client.chat(
            model=self._model, messages=_to_ollama_messages(messages),
            tools=self._tools, options=self._options,
        )
        msg = resp["message"]
        tool_calls = [
            {"name": tc.function.name, "args": dict(tc.function.arguments),
             "id": f"call_{i}", "type": "tool_call"}
            for i, tc in enumerate(msg.tool_calls or [])
        ]
        return AIMessage(content=msg.content or "", tool_calls=tool_calls)


def bind_tools(llm: Any, tools: list[Any]) -> Any:
    """Return a tool-bound model with ``.ainvoke(messages) -> AIMessage``.

    ChatOllama → raw ollama tool-calling; otherwise ``llm.bind_tools(tools)``.
    """
    if _is_chat_ollama(llm):
        return _OllamaToolCaller(llm, tools)
    return llm.bind_tools(tools)
