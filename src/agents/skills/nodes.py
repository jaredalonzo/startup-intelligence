"""Skills agent node implementations.

Deterministic nodes: load_deltas, normalize_taxonomy, aggregate_trends, route_outputs
LLM nodes:          extract_one, synthesize_radar
Coordinator:        extract_skills (emits Send fan-out; no computation of its own)
"""
from __future__ import annotations

import functools
import logging
import pathlib
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from itertools import combinations
from typing import Any

import psycopg
import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from psycopg.types.json import Jsonb
from pydantic import BaseModel

from agents.skills.state import SkillExtraction, SkillTrend, SkillsState, TrendReport
from config import (
    EXTRACTION_LLM,
    SKILLS_DEFAULT_WINDOW_DAYS,
    SKILLS_GAP_TASK_THRESHOLD_PCT,
    SKILLS_MIN_POSTING_COUNT,
    SKILLS_TOP_N,
    SKILLS_WATERMARK_KEY,
    SYNTHESIS_LLM,
    TARGET_ROLES,
)
from llm_structured import structured
from outputs.linear import create_gap_tasks
from outputs.notion import write_skills_digest
from roles import is_technical
from store.db import get_agent_watermark, get_connection, set_agent_watermark

logger = logging.getLogger(__name__)

_ALIASES_FILE = pathlib.Path(__file__).parent.parent.parent / "taxonomy" / "aliases.yaml"


class _PostingExtraction(BaseModel):
    """LLM-extracted fields from a single job posting description."""
    skills: list[str]
    platforms: list[str]
    seniority: str | None        # senior | staff | principal | junior | ic | manager | null
    years_experience: int | None


# ---------------------------------------------------------------------------
# Deterministic nodes
# ---------------------------------------------------------------------------

def load_deltas(state: SkillsState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Load technical job postings updated since the last watermark.

    Queries postings by first_seen_at > watermark (not updated_at, which is NULL
    for Ashby and Workable boards that do not expose a true updatedAt field), then
    keeps only technical roles — the skills radar targets FDE/TAM/CSE/eng, and the
    corpus is ~half GTM/ops/recruiting noise. Pass configurable ``all_roles=True``
    to skip the filter (a deliberate broad analysis).

    Watermark source, in priority order: an explicit ``state['watermark']``
    (set by ``--window-days`` to force a lookback), else the stored agent
    watermark from a previous run, else a default first-run window. The candidate
    new watermark (NOW()) is returned in state but only *committed* to the store in
    aggregate_trends, alongside the extractions it covers — so a crash before then
    re-processes the same postings next run rather than silently skipping them.
    """
    now = datetime.now(timezone.utc)
    all_roles = bool((config or {}).get("configurable", {}).get("all_roles", False))

    watermark_str = state.get("watermark")
    if watermark_str:
        watermark = datetime.fromisoformat(watermark_str)
    else:
        stored = get_agent_watermark(SKILLS_WATERMARK_KEY)
        watermark = stored or (now - timedelta(days=SKILLS_DEFAULT_WINDOW_DAYS))

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

    if all_roles:
        new_postings = list(rows)
    else:
        new_postings = [r for r in rows if is_technical(r.get("title"), r.get("department"))]
    logger.info(
        "load_deltas: %d postings since watermark, %d kept%s",
        len(rows), len(new_postings), "" if all_roles else " (technical only)",
    )

    return {
        "new_postings": new_postings,
        "watermark": now.isoformat(),
    }


def extract_skills(state: SkillsState) -> dict[str, Any]:
    """Fan-out coordinator. Does no work itself — _fan_out_extractions emits Sends."""
    return {}


@functools.cache
def _load_aliases() -> tuple[dict[str, str], frozenset[str]]:
    """Load alias table from aliases.yaml. Cached after first read.

    Returns (aliases, known_canonicals) where aliases keys are lowercased for
    case-insensitive lookup and known_canonicals is the set of all right-hand values.
    """
    with open(_ALIASES_FILE) as f:
        raw: dict[str, str] = yaml.safe_load(f) or {}
    aliases = {k.lower(): v for k, v in raw.items()}
    known = frozenset(raw.values())
    return aliases, known


def _normalize_list(
    items: list[str],
    aliases: dict[str, str],
    known: frozenset[str],
) -> tuple[list[str], list[str]]:
    """Normalize a list of skill/platform strings against the alias table.

    Returns (normalized, unknowns). Deduplicates after normalization.
    A skill is "unknown" if it maps to nothing in aliases and is not itself a
    known canonical — i.e. we can't verify it through our taxonomy.
    """
    normalized: list[str] = []
    unknowns: list[str] = []
    seen: set[str] = set()
    for item in items:
        canonical = aliases.get(item.lower(), item)
        if canonical not in seen:
            normalized.append(canonical)
            seen.add(canonical)
        if item.lower() not in aliases and item not in known:
            unknowns.append(item)
    return normalized, unknowns


def _as_extractions(items: list[SkillExtraction]) -> list[SkillExtraction]:
    """Coerce checkpointer-reloaded extractions back into SkillExtraction models.

    The Postgres checkpointer persists/reloads the `extractions` and
    `normalized_extractions` channels between super-steps, and JsonPlusSerializer
    does not reliably reconstruct the Pydantic type across the src-layout import
    path — items can come back as plain dicts. model_validate accepts a dict or an
    existing model, so this is a safe boundary coercion for both consumers.
    """
    return [SkillExtraction.model_validate(i) for i in items]


def normalize_taxonomy(state: SkillsState) -> dict[str, Any]:
    """Apply aliases.yaml to collapse synonyms across all extractions.

    Reads state["extractions"] (fan-in output from extract_one).
    Writes state["normalized_extractions"] — a replacement list, not an append,
    so this field has no operator.add reducer.
    Unknown skills (not in aliases keys or values) are collected into
    state["unknown_skills"] for periodic taxonomy review.
    """
    aliases, known = _load_aliases()
    normalized: list[SkillExtraction] = []
    all_unknowns: list[str] = []

    for extraction in _as_extractions(state.get("extractions", [])):
        norm_skills, skill_unknowns = _normalize_list(extraction.skills, aliases, known)
        norm_platforms, plat_unknowns = _normalize_list(extraction.platforms, aliases, known)
        all_unknowns.extend(skill_unknowns + plat_unknowns)
        normalized.append(extraction.model_copy(update={
            "skills": norm_skills,
            "platforms": norm_platforms,
        }))

    unique_unknowns = sorted(set(all_unknowns))
    if unique_unknowns:
        logger.info(
            "Unknown skills flagged for taxonomy review (%d): %s",
            len(unique_unknowns),
            unique_unknowns,
        )

    return {
        "normalized_extractions": normalized,
        "unknown_skills": unique_unknowns,
    }


def _persist_extractions(
    extractions: list[SkillExtraction], conn: psycopg.Connection[dict[str, Any]]
) -> None:
    """Insert normalized extractions into the DB. Append-only by design.

    Does not commit — the caller (aggregate_trends) commits the inserts and the
    watermark advance together so they can't desync. Duplicate rows on graph
    retry are otherwise safe: downstream reads use extracted_at DESC per posting
    to get the most recent extraction.
    """
    for ex in extractions:
        conn.execute(
            """
            INSERT INTO extractions
                (ats, posting_id, extracted_at, model, skills, platforms,
                 seniority_signal, comp_min, comp_max, comp_currency, comp_interval, raw)
            VALUES
                (%(ats)s, %(posting_id)s, NOW(), %(model)s, %(skills)s, %(platforms)s,
                 %(seniority)s, %(comp_min)s, %(comp_max)s, %(comp_currency)s, %(comp_interval)s, %(raw)s)
            """,
            {
                "ats": ex.ats,
                "posting_id": ex.posting_id,
                "model": EXTRACTION_LLM.model,
                "skills": ex.skills,
                "platforms": ex.platforms,
                "seniority": ex.seniority,
                "comp_min": ex.comp_min,
                "comp_max": ex.comp_max,
                "comp_currency": ex.comp_currency,
                "comp_interval": ex.comp_interval,
                "raw": Jsonb(ex.model_dump()),
            },
        )


def _query_prev_counts(
    conn: psycopg.Connection[dict[str, Any]], window_days: int
) -> tuple[Counter[str], Counter[str]]:
    """Query skill/platform counts from the previous equal-duration window.

    Previous window = [now - 2×window_days, now - window_days].
    On first run the table is empty, so both Counters return zero for every key.
    Uses timedelta params so psycopg3 maps them to PG interval natively.
    """
    rows = conn.execute(
        """
        SELECT skills, platforms
        FROM extractions
        WHERE extracted_at <  NOW() - %(current)s
          AND extracted_at >= NOW() - %(lookback)s
        """,
        {
            "current": timedelta(days=window_days),
            "lookback": timedelta(days=window_days * 2),
        },
    ).fetchall()

    skill_counts: Counter[str] = Counter()
    platform_counts: Counter[str] = Counter()
    for row in rows:
        for s in row["skills"] or []:
            skill_counts[s] += 1
        for p in row["platforms"] or []:
            platform_counts[p] += 1
    return skill_counts, platform_counts


def aggregate_trends(state: SkillsState) -> dict[str, Any]:
    """Compute frequency deltas, new skills, and co-occurrence from normalized extractions.

    Persists the current extractions to the DB first (so future runs have a
    previous window to diff against) and advances the agent watermark in the same
    transaction — committing them together means a later failure can't leave the
    watermark ahead of the extractions it covers. Then queries the previous
    equal-duration window to compute count_previous for each skill/platform. No LLM.
    """
    extractions = _as_extractions(state.get("normalized_extractions") or [])
    total = len(extractions)
    window_days = SKILLS_DEFAULT_WINDOW_DAYS
    candidate_watermark = state.get("watermark")

    with get_connection() as conn:
        _persist_extractions(extractions, conn)
        if candidate_watermark:
            set_agent_watermark(
                SKILLS_WATERMARK_KEY, datetime.fromisoformat(candidate_watermark), conn=conn
            )
        conn.commit()
        prev_skills, prev_platforms = _query_prev_counts(conn, window_days)

    # Count skill and platform frequencies in the current window
    curr_skills: Counter[str] = Counter()
    curr_platforms: Counter[str] = Counter()
    co_counts: Counter[tuple[str, str]] = Counter()

    for ex in extractions:
        for s in ex.skills:
            curr_skills[s] += 1
        for p in ex.platforms:
            curr_platforms[p] += 1
        # Pairs are sorted so (A, B) and (B, A) are the same bucket
        for pair in combinations(sorted(set(ex.skills)), 2):
            co_counts[pair] += 1

    def _trend(name: str, curr: int, prev: int) -> SkillTrend:
        return SkillTrend(
            skill=name,
            count_current=curr,
            count_previous=prev,
            delta=curr - prev,
            pct_of_postings=curr / total if total else 0.0,
        )

    # Only include skills that meet the minimum posting count threshold
    qualified = {s for s in curr_skills if curr_skills[s] >= SKILLS_MIN_POSTING_COUNT}
    skill_trends = [_trend(s, curr_skills[s], prev_skills[s]) for s in qualified]

    rising  = sorted([t for t in skill_trends if t.delta > 0], key=lambda t: -t.delta)[:SKILLS_TOP_N]
    falling = sorted([t for t in skill_trends if t.delta < 0], key=lambda t:  t.delta)[:SKILLS_TOP_N]
    new     = [s for s in qualified if prev_skills[s] == 0]

    platform_trends = sorted(
        [
            _trend(p, curr_platforms[p], prev_platforms[p])
            for p in curr_platforms
            if curr_platforms[p] >= SKILLS_MIN_POSTING_COUNT
        ],
        key=lambda t: -t.count_current,
    )[:SKILLS_TOP_N]

    co_occurrences = [(a, b, n) for (a, b), n in co_counts.most_common(SKILLS_TOP_N)]

    return {
        "trend_report": TrendReport(
            window_days=window_days,
            total_postings=total,
            rising=rising,
            falling=falling,
            new=new,
            top_platforms=platform_trends,
            co_occurrences=co_occurrences,
        )
    }


def route_outputs(state: SkillsState) -> dict[str, Any]:
    """Write the radar digest to Notion and open/refresh Linear gap tasks.

    Gap skills are any skill/platform whose pct_of_postings >= SKILLS_GAP_TASK_THRESHOLD_PCT.
    Each external write is guarded so an outputs failure doesn't lose the whole run.
    """
    digest = state.get("radar_digest") or ""
    report = state.get("trend_report")

    logger.info("=== SKILLS RADAR DIGEST ===\n%s", digest)

    if digest:
        try:
            page_url = write_skills_digest(digest)
            logger.info("Skills radar digest written to Notion: %s", page_url)
        except Exception:
            logger.exception("route_outputs: Notion digest write failed")

    if report:
        all_trends = report.rising + report.falling + report.top_platforms
        seen: set[str] = set()
        gap_skills = []
        for t in all_trends:
            if t.skill not in seen and t.pct_of_postings >= SKILLS_GAP_TASK_THRESHOLD_PCT:
                gap_skills.append(t)
                seen.add(t.skill)

        if gap_skills:
            logger.info(
                "Creating Linear gap tasks (%d skills above %.0f%% threshold): %s",
                len(gap_skills),
                SKILLS_GAP_TASK_THRESHOLD_PCT * 100,
                [t.skill for t in gap_skills],
            )
            try:
                identifiers = create_gap_tasks(
                    [(t.skill, t.pct_of_postings) for t in gap_skills]
                )
                logger.info("route_outputs: gap tasks upserted: %s", identifiers)
            except Exception:
                logger.exception("route_outputs: Linear gap-task creation failed")
        else:
            logger.info(
                "No skills above gap threshold (%.0f%%)",
                SKILLS_GAP_TASK_THRESHOLD_PCT * 100,
            )

    return {}


# ---------------------------------------------------------------------------
# LLM nodes
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = (
    "Extract structured information from job postings. "
    "Use canonical names: 'Kubernetes' not 'k8s', 'PostgreSQL' not 'postgres', "
    "'TypeScript' not 'TS', 'JavaScript' not 'JS'. "
    "For seniority, return exactly one of: senior, staff, principal, junior, ic, manager — "
    "or null if not determinable. "
    "For platforms, include cloud providers and infrastructure (AWS, GCP, Azure, Kubernetes, "
    "Terraform, etc.) — not general-purpose languages or frameworks."
)


def extract_posting_fields(posting: dict[str, Any], llm: Any = None) -> _PostingExtraction:
    """Run the single-posting skill/platform extraction with the given chat model.

    The model is a parameter so the eval harness can run the *exact* production
    prompt + schema through different models. Resolved at call time (not as a
    default-arg binding) so the graph's configured EXTRACTION_LLM stays patchable.
    """
    if llm is None:
        llm = EXTRACTION_LLM
    text = posting.get("description_text") or ""
    context_parts = [f"Title: {posting.get('title', '')}"]
    if posting.get("department"):
        context_parts.append(f"Department: {posting['department']}")
    if posting.get("team"):
        context_parts.append(f"Team: {posting['team']}")
    context_parts.append(f"\n{text}")

    chain = structured(llm, _PostingExtraction)
    result: _PostingExtraction = chain.invoke([
        SystemMessage(content=_EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content="\n".join(context_parts)),
    ])
    return result


_EMPTY_EXTRACTION = _PostingExtraction(skills=[], platforms=[], seniority=None, years_experience=None)


def _extract_with_retry(posting: dict[str, Any], attempts: int = 2) -> _PostingExtraction:
    """Extract one posting, retrying transient failures, then degrading to empty.

    Per-posting isolation: the fan-out runs serialized (one cloud call at a time),
    so a single transient error (429/timeout) or an unparseable model response must
    not abort the whole run. Retry covers transient blips; a final miss yields an
    empty extraction so the posting is excluded from trends rather than crashing.
    """
    for attempt in range(1, attempts + 1):
        try:
            return extract_posting_fields(posting)
        except Exception:
            logger.warning(
                "extract_one: extraction failed for posting %s (attempt %d/%d)",
                posting.get("id"), attempt, attempts,
                exc_info=(attempt == attempts),
            )
            if attempt < attempts:
                time.sleep(1.0 * attempt)
    logger.error(
        "extract_one: giving up on posting %s after %d attempts; emitting empty extraction",
        posting.get("id"), attempts,
    )
    return _EMPTY_EXTRACTION


def extract_one(state: dict[str, Any]) -> dict[str, Any]:
    """Extract skills, platforms, seniority, and comp from a single posting.

    Receives {"posting": <postings row dict>} from Send.
    Returns {"extractions": [SkillExtraction]} — merged via operator.add.
    """
    posting = state["posting"]
    fields = _extract_with_retry(posting)
    extraction = SkillExtraction(
        posting_id=posting["id"],
        ats=posting["ats"],
        company_slug=posting["company_slug"],
        skills=fields.skills,
        platforms=fields.platforms,
        seniority=fields.seniority,
        years_experience=fields.years_experience,
        comp_min=posting.get("compensation_min"),
        comp_max=posting.get("compensation_max"),
        comp_currency=posting.get("compensation_currency"),
        comp_interval=posting.get("compensation_interval"),
    )
    return {"extractions": [extraction]}


def synthesize_radar(state: SkillsState) -> dict[str, Any]:
    """Turn the TrendReport into a personalized skills radar digest.

    Targets FDE, TAM, CSE, and implementation archetypes. Highlights what's
    heating up, what's now table-stakes, what's fading, with specific gaps to close.
    """
    report = state.get("trend_report")
    if report is None:
        return {"radar_digest": None}

    def _trend_lines(trends: list[SkillTrend]) -> str:
        return "\n".join(
            f"  - {t.skill}: delta={t.delta:+d}, {t.pct_of_postings:.1%} of postings"
            for t in trends
        ) or "  none"

    rising_lines   = _trend_lines(report.rising)
    falling_lines  = _trend_lines(report.falling)
    platform_lines = _trend_lines(report.top_platforms)
    new_skills     = ", ".join(report.new) if report.new else "none"
    co_lines       = "\n".join(
        f"  - {a} + {b}: {n} postings"
        for a, b, n in report.co_occurrences[:10]
    ) or "  none"
    roles_str = ", ".join(TARGET_ROLES)

    prompt = f"""Hiring signal data from {report.total_postings} technical job postings across \
AI/data/infra startups over the past {report.window_days} days.

RISING SKILLS (largest increase vs previous window):
{rising_lines}

FALLING SKILLS (largest decrease vs previous window):
{falling_lines}

NEW SKILLS (appeared for the first time this window):
{new_skills}

TOP PLATFORMS by posting volume:
{platform_lines}

COMMON CO-OCCURRENCES (skills frequently required together):
{co_lines}

Produce a skills radar digest for someone targeting these roles: {roles_str}.

Structure your response as markdown with exactly these sections:
## Heating Up
Skills rising fast — worth prioritizing now.

## Table Stakes
High-frequency skills that are now baseline expectations.

## Fading
Skills declining — deprioritize unless already strong.

## New on the Radar
Skills newly appearing — early signal worth watching.

## Platform & Infrastructure Signals
Cloud/infra trends relevant to the target roles.

## Gaps to Close
Skills appearing in ≥{SKILLS_GAP_TASK_THRESHOLD_PCT:.0%} of postings that are likely gaps \
for the target archetypes.

Be specific and actionable. Name actual skills, not categories. Keep each section to 3–5 bullets."""

    response = SYNTHESIS_LLM.invoke([
        SystemMessage(content=(
            "You are a technical skills analyst for AI/data/infra roles. "
            "Write concise, actionable radar digests. "
            "Use specific skill names, not vague categories. "
            "Be direct about what to prioritize and why."
        )),
        HumanMessage(content=prompt),
    ])
    digest: str = response.content  # type: ignore[assignment]
    logger.info("synthesize_radar: generated %d-char digest", len(digest))
    return {"radar_digest": digest}
