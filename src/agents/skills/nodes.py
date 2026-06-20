"""Skills agent node implementations.

Deterministic nodes: load_deltas, normalize_taxonomy, aggregate_trends, route_outputs
LLM nodes:          extract_one, synthesize_radar
Coordinator:        extract_skills (emits Send fan-out; no computation of its own)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import anthropic
from pydantic import BaseModel

from agents.state import SkillExtraction, SkillsState
from config import EXTRACTION_MODEL, SKILLS_DEFAULT_WINDOW_DAYS
from store.db import get_connection


class _PostingExtraction(BaseModel):
    """LLM-extracted fields from a single job posting description."""
    skills: list[str]
    platforms: list[str]
    seniority: str | None        # senior | staff | principal | junior | ic | manager | null
    years_experience: int | None


# ---------------------------------------------------------------------------
# Deterministic nodes
# ---------------------------------------------------------------------------

def load_deltas(state: SkillsState) -> dict:
    """Load technical job postings updated since the last watermark.

    Queries postings by first_seen_at > watermark (not updated_at, which is NULL
    for Ashby and Workable boards that do not expose a true updatedAt field).
    Returns new_postings and advances the watermark to NOW().
    """
    now = datetime.now(timezone.utc)

    watermark_str = state.get("watermark")
    if watermark_str:
        watermark = datetime.fromisoformat(watermark_str)
    else:
        watermark = now - timedelta(days=SKILLS_DEFAULT_WINDOW_DAYS)

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT ats, id, company_slug, title, department, team,
                   description_text,
                   compensation_min, compensation_max,
                   compensation_currency, compensation_interval
            FROM postings
            WHERE first_seen_at > %(watermark)s
              AND description_text IS NOT NULL
              AND description_text <> ''
            ORDER BY first_seen_at
            """,
            {"watermark": watermark},
        ).fetchall()

    return {
        "new_postings": list(rows),
        "watermark": now.isoformat(),
    }


def extract_skills(state: SkillsState) -> dict:
    """Fan-out coordinator. Does no work itself — _fan_out_extractions emits Sends."""
    return {}


def normalize_taxonomy(state: SkillsState) -> dict:
    """Apply aliases.yaml to collapse synonyms across all extractions.

    e.g. k8s → Kubernetes, postgres → PostgreSQL.
    Unknown skills are flagged as a side output for review (non-blocking).

    Implemented in JAR-51.
    """
    raise NotImplementedError("normalize_taxonomy — implement in JAR-51")


def aggregate_trends(state: SkillsState) -> dict:
    """Compute frequency deltas, new skills, and co-occurrence from extractions.

    Returns a TrendReport. No LLM.

    Implemented in JAR-52.
    """
    raise NotImplementedError("aggregate_trends — implement in JAR-52")


def route_outputs(state: SkillsState) -> dict:
    """Write the radar digest to Notion and create Linear gap tasks if thresholds crossed.

    Deterministic. Implemented in JAR-53.
    """
    raise NotImplementedError("route_outputs — implement in JAR-53")


# ---------------------------------------------------------------------------
# LLM nodes
# ---------------------------------------------------------------------------

def extract_one(state: dict) -> dict:
    """Extract skills, platforms, seniority, and comp from a single posting.

    Receives {"posting": <postings row dict>} from Send.
    Returns {"extractions": [SkillExtraction]} — merged via operator.add.
    """
    posting = state["posting"]

    text = posting.get("description_text") or ""
    context_parts = [f"Title: {posting.get('title', '')}"]
    if posting.get("department"):
        context_parts.append(f"Department: {posting['department']}")
    if posting.get("team"):
        context_parts.append(f"Team: {posting['team']}")
    context_parts.append(f"\n{text}")

    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=EXTRACTION_MODEL,
        max_tokens=1024,
        system=(
            "Extract structured information from job postings. "
            "Use canonical names: 'Kubernetes' not 'k8s', 'PostgreSQL' not 'postgres', "
            "'TypeScript' not 'TS', 'JavaScript' not 'JS'. "
            "For seniority, return exactly one of: senior, staff, principal, junior, ic, manager — "
            "or null if not determinable. "
            "For platforms, include cloud providers and infrastructure (AWS, GCP, Azure, Kubernetes, "
            "Terraform, etc.) — not general-purpose languages or frameworks."
        ),
        messages=[{"role": "user", "content": "\n".join(context_parts)}],
        output_format=_PostingExtraction,
    )

    llm = response.parsed_output
    extraction = SkillExtraction(
        posting_id=posting["id"],
        ats=posting["ats"],
        company_slug=posting["company_slug"],
        skills=llm.skills,
        platforms=llm.platforms,
        seniority=llm.seniority,
        years_experience=llm.years_experience,
        comp_min=posting.get("compensation_min"),
        comp_max=posting.get("compensation_max"),
        comp_currency=posting.get("compensation_currency"),
        comp_interval=posting.get("compensation_interval"),
    )
    return {"extractions": [extraction]}


def synthesize_radar(state: SkillsState) -> dict:
    """Turn the TrendReport into a personalized skills radar digest.

    Targets FDE, TAM, CSE, and implementation archetypes. Highlights what's
    heating up, what's now table-stakes, what's fading, with example JDs and
    specific gaps to close.

    Implemented in JAR-53.
    """
    raise NotImplementedError("synthesize_radar — implement in JAR-53")
