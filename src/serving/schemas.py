"""Request/response schemas for the query API.

The response exposes the citation surface of each retrieved doc (URL +
dates), not the snippet text the answer model saw — snippets are context,
not citations, and 16 × 1200-char JD excerpts would bloat every response.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from agents.query.state import RetrievedDoc


class QueryRequest(BaseModel):
    # The length cap is an input-side cost guardrail: the question is embedded
    # and fed to two LLM calls verbatim.
    question: str = Field(min_length=1, max_length=2000)


class Citation(BaseModel):
    """One retrieved doc, minus its snippet — the cite-or-abstain audit trail."""

    kind: Literal["posting", "dossier"]
    company_slug: str
    company_name: str
    title: str
    url: str | None
    distance: float
    posted_at: datetime | None
    first_seen_at: datetime | None
    last_seen_at: datetime | None
    generated_at: datetime | None

    @classmethod
    def from_doc(cls, doc: RetrievedDoc) -> Citation:
        return cls(**doc.model_dump(exclude={"snippet"}))


class QueryUsage(BaseModel):
    llm_calls: int
    # 0 when the backend reports no token usage (Ollama sometimes doesn't).
    # Counts parse + answer only: the query embedding is an embeddings call
    # and never fires LLM callbacks — that is by design, not a missing count.
    llm_tokens: int
    duration_ms: int


class QueryResponse(BaseModel):
    answer_markdown: str
    # True iff retrieval came back empty and the answer node abstained
    # deterministically (no LLM call was made).
    abstained: bool
    parse_fallback: bool
    # Postings first, then dossiers — the same continuous numbering the answer
    # prompt used, so citations[i] is the doc cited as [i+1] in answer_markdown.
    citations: list[Citation]
    request_id: str
    usage: QueryUsage
