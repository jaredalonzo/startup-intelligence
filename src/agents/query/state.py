"""State objects for the query head (RAG) graph.

Per CLAUDE.md the heads share no state objects: this state is per-request,
created fresh for each question and discarded after the answer.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Query plan (output of parse_query — doubles as its tool schema)
# ---------------------------------------------------------------------------

class QueryPlan(BaseModel):
    """Decompose the user's question into semantic search text plus structured filters.

    Closed world: every filter maps to a real typed column or join in the store.
    Anything the question implies that has no field here belongs in
    semantic_terms — never invent filters. Only set a filter the question
    explicitly implies.
    """

    semantic_terms: str = Field(
        description=(
            "Free text describing the roles/technologies/topics to search for, "
            "e.g. 'Rust distributed systems engineer'. Required — when no filters "
            "apply, restate the question's topical content here."
        )
    )
    companies: list[str] = Field(
        default_factory=list,
        description="Company slugs to restrict to: lowercase, no spaces, e.g. 'anthropic'.",
    )
    departments: list[str] = Field(
        default_factory=list,
        description="Department names to match (substring, case-insensitive), e.g. 'engineering'.",
    )
    teams: list[str] = Field(
        default_factory=list,
        description="Team names to match (substring, case-insensitive).",
    )
    seniorities: list[str] = Field(
        default_factory=list,
        description=(
            "Seniority levels to restrict to. Legal values: senior, staff, "
            "principal, junior, ic, manager."
        ),
    )
    locations: list[str] = Field(
        default_factory=list,
        description="Locations to match (substring, case-insensitive), e.g. 'london'.",
    )
    remote: bool | None = Field(
        default=None, description="True to require remote roles, False to exclude them."
    )
    posted_after_days: int | None = Field(
        default=None,
        description="Only postings the ATS says were posted within the last N days.",
    )
    first_seen_after_days: int | None = Field(
        default=None,
        description="Only postings that first appeared in our corpus within the last N days.",
    )
    comp_min: int | None = Field(
        default=None,
        description="Minimum annual compensation the posting's range must reach (USD-ish, raw units).",
    )
    skills: list[str] = Field(
        default_factory=list,
        description="Extracted skill names the posting must mention, e.g. 'Kubernetes', 'Rust'.",
    )
    platforms: list[str] = Field(
        default_factory=list,
        description="Extracted platform names the posting must mention, e.g. 'AWS'.",
    )
    growing_eng: bool = Field(
        default=False,
        description=(
            "True only when the question asks for companies whose engineering "
            "hiring is growing — filters on the snapshot time series."
        ),
    )
    corpus: Literal["postings", "dossiers", "both"] = Field(
        default="both",
        description=(
            "Which corpus to search: 'postings' for job-level questions, "
            "'dossiers' for company-trajectory questions, 'both' when unsure."
        ),
    )


# ---------------------------------------------------------------------------
# Retrieved document (output rows of retrieve, normalized across corpora)
# ---------------------------------------------------------------------------

class RetrievedDoc(BaseModel):
    """One retrieved posting or dossier, normalized for context assembly.

    `distance` is pgvector cosine distance to the query vector (lower = closer).
    Date fields are populated per kind: postings carry posted/first/last-seen,
    dossiers carry generated_at. `url` is the citation target (posting URL or
    the dossier's Notion page); it can be None when the source omitted it.
    """

    kind: Literal["posting", "dossier"]
    company_slug: str
    company_name: str
    title: str
    url: str | None = None
    snippet: str = ""
    distance: float
    posted_at: datetime | None = None
    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None
    generated_at: datetime | None = None


# ---------------------------------------------------------------------------
# Query head state
# ---------------------------------------------------------------------------

class QueryState(TypedDict):
    """State for the query head graph — one question in, one grounded answer out.

    question:        the user's natural-language question, verbatim
    plan:            parse_query's decomposition; None until that node runs
    parse_fallback:  True when parse_query degraded to a semantic-only plan
                     (the parse model failed or returned no usable tool call)
    postings:        retrieved posting docs; None until retrieve runs
    dossiers:        retrieved dossier docs; None until retrieve runs
    answer_markdown: the grounded, cited answer (or the deterministic abstain)
    """

    question: str
    plan: QueryPlan | None
    parse_fallback: bool
    postings: list[RetrievedDoc] | None
    dossiers: list[RetrievedDoc] | None
    answer_markdown: str | None
