"""Skills agent node implementations.

Deterministic nodes: load_deltas, normalize_taxonomy, aggregate_trends, route_outputs
LLM nodes:          extract_one, synthesize_radar
Coordinator:        extract_skills (emits Send fan-out; no computation of its own)
"""
from __future__ import annotations

from agents.state import SkillsState


# ---------------------------------------------------------------------------
# Deterministic nodes
# ---------------------------------------------------------------------------

def load_deltas(state: SkillsState) -> dict:
    """Load technical job postings updated since the last watermark.

    Queries the postings table for rows with updated_at > watermark.
    Returns new_postings and advances the watermark to NOW().

    Implemented in JAR-50.
    """
    raise NotImplementedError("load_deltas — implement in JAR-50")


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

    Implemented in JAR-50.
    """
    raise NotImplementedError("extract_one — implement in JAR-50")


def synthesize_radar(state: SkillsState) -> dict:
    """Turn the TrendReport into a personalized skills radar digest.

    Targets FDE, TAM, CSE, and implementation archetypes. Highlights what's
    heating up, what's now table-stakes, what's fading, with example JDs and
    specific gaps to close.

    Implemented in JAR-53.
    """
    raise NotImplementedError("synthesize_radar — implement in JAR-53")
