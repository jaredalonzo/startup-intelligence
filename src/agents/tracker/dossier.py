"""Startup tracker — dossier synthesis tail of the per-company graph (JAR-56).

Pipeline after a board resolves:

  load_signals (det, reads the store)
    → branch: meaningful change? → END if not (cost saver)
    → synthesize_dossier (LLM)   — narrative across the five metrics
    → score_trending (det composite + LLM judgment flag)
    → write_dossier (det)        — upsert Notion page; flag top movers for Linear

Per the repo's "agent reads the store; ingestion writes the store" guardrail,
load_signals only reads the persisted snapshots/blog_posts/github_releases/
github_repo_stats — it never fetches. The fetch_signals / snapshot / diff nodes
(JAR-55) populate that store and slot in ahead of load_signals once built.

The five metrics: headcount (eng-weighted hiring velocity), open positions,
technology/product, product evolution, and customers. `customers` has no
ingestion source yet, so the dossier names it as pending rather than guessing.
"""
from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from agents.tracker.state import DossierInputs, TrackerState, TrendScore
from config import (
    SYNTHESIS_LLM,
    TRACKER_ACCEL_BAND,
    TRACKER_COOL_BAND,
    TRACKER_DOSSIER_SNAPSHOT_LOOKBACK,
    TRACKER_SCORE_CAPS,
    TRACKER_SCORE_WEIGHTS,
)
from outputs.linear import create_top_mover_task
from outputs.notion import upsert_company_dossier
from store.db import get_connection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# load_signals (deterministic — reads the store)
# ---------------------------------------------------------------------------

def _delta(curr: int | None, prev: int | None) -> int | None:
    if curr is None or prev is None:
        return None
    return curr - prev


def _count_since(rows: list[dict], since: datetime | None) -> int:
    """How many rows were first seen at/after *since* (the previous snapshot)."""
    if since is None:
        return 0
    return sum(1 for r in rows if r.get("first_seen_at") and r["first_seen_at"] >= since)


def _star_deltas(rows: list[dict]) -> list[tuple[str, int]]:
    """Latest-vs-previous star delta per repo, from measured_at-DESC rows."""
    seen: dict[str, list[int]] = {}
    for r in rows:
        seen.setdefault(r["repo"], []).append(r["star_count"])
    deltas: list[tuple[str, int]] = []
    for repo, stars in seen.items():
        if len(stars) >= 2:                      # stars[0] latest, stars[1] previous
            deltas.append((repo, stars[0] - stars[1]))
    return sorted(deltas, key=lambda t: -t[1])


def load_signals(state: TrackerState) -> dict:
    """Assemble the per-company dossier inputs from the persisted store.

    Reads the most recent snapshots (for the headcount/open-position deltas),
    the currently-live postings, and the blog/GitHub signal tables. Also decides
    deterministically whether anything meaningful changed, so the graph can skip
    the LLM synthesis tail when nothing did.
    """
    company = state["company"]
    slug = company["slug"]
    name = company.get("name") or slug

    with get_connection() as conn:
        # get_connection yields a dict_row connection at runtime; alias it as Any
        # so row["col"] access type-checks (the shared type is unparameterized).
        db: Any = conn
        snaps = db.execute(
            """
            SELECT snapshot_at, posting_count, eng_count, new_ids, removed_ids
            FROM snapshots
            WHERE company_slug = %s
            ORDER BY snapshot_at DESC
            LIMIT %s
            """,
            (slug, TRACKER_DOSSIER_SNAPSHOT_LOOKBACK),
        ).fetchall()

        latest = snaps[0] if snaps else None
        prev = snaps[1] if len(snaps) > 1 else None
        oldest = snaps[-1] if len(snaps) > 1 else None    # window start for the trend
        prev_at = prev["snapshot_at"] if prev else None
        since_at = latest["snapshot_at"] if latest else None

        # Currently-live postings = those seen on the most recent ingestion run.
        if since_at is not None:
            postings = db.execute(
                "SELECT title, department FROM postings "
                "WHERE company_slug = %s AND last_seen_at >= %s",
                (slug, since_at),
            ).fetchall()
        else:
            postings = db.execute(
                "SELECT title, department FROM postings WHERE company_slug = %s",
                (slug,),
            ).fetchall()

        blog_rows = db.execute(
            "SELECT title, url, published_at, first_seen_at FROM blog_posts "
            "WHERE company_slug = %s ORDER BY published_at DESC NULLS LAST LIMIT 10",
            (slug,),
        ).fetchall()

        release_rows = db.execute(
            "SELECT repo, release_tag, release_name, published_at, first_seen_at "
            "FROM github_releases WHERE company_slug = %s "
            "ORDER BY published_at DESC NULLS LAST LIMIT 10",
            (slug,),
        ).fetchall()

        star_rows = db.execute(
            "SELECT repo, measured_at, star_count FROM github_repo_stats "
            "WHERE company_slug = %s ORDER BY repo, measured_at DESC",
            (slug,),
        ).fetchall()

    dept_counts = Counter(p["department"] for p in postings if p.get("department"))

    new_blog_count = _count_since(blog_rows, prev_at)
    new_release_count = _count_since(release_rows, prev_at)

    signals = DossierInputs(
        company_slug=slug,
        company_name=name,
        snapshots_available=len(snaps),
        posting_count=latest["posting_count"] if latest else None,
        posting_count_delta=_delta(
            latest["posting_count"] if latest else None,
            prev["posting_count"] if prev else None,
        ),
        posting_count_window_delta=_delta(
            latest["posting_count"] if latest else None,
            oldest["posting_count"] if oldest else None,
        ),
        eng_count=latest["eng_count"] if latest else None,
        eng_count_delta=_delta(
            latest["eng_count"] if latest else None,
            prev["eng_count"] if prev else None,
        ),
        eng_count_window_delta=_delta(
            latest["eng_count"] if latest else None,
            oldest["eng_count"] if oldest else None,
        ),
        new_postings=len(latest["new_ids"]) if latest else 0,
        removed_postings=len(latest["removed_ids"]) if latest else 0,
        open_by_department=dept_counts.most_common(8),
        sample_titles=[p["title"] for p in postings[:8]],
        recent_releases=[
            {"repo": r["repo"], "tag": r["release_tag"], "name": r["release_name"],
             "published_at": r["published_at"].isoformat() if r["published_at"] else None}
            for r in release_rows
        ],
        star_delta_by_repo=_star_deltas(star_rows),
        new_release_count=new_release_count,
        recent_blog_posts=[
            {"title": b["title"], "url": b["url"],
             "published_at": b["published_at"].isoformat() if b["published_at"] else None}
            for b in blog_rows
        ],
        new_blog_count=new_blog_count,
    )

    has_delta = any([
        signals.new_postings > 0,
        signals.removed_postings > 0,
        (signals.posting_count_delta or 0) != 0,
        (signals.eng_count_delta or 0) != 0,
        new_release_count > 0,
        new_blog_count > 0,
    ])
    # A newly-tracked company (no prior snapshot to diff) still warrants an
    # initial dossier the first time we have any data for it.
    first_dossier = signals.snapshots_available <= 1 and signals.posting_count is not None
    meaningful_change = has_delta or first_dossier

    logger.info(
        "load_signals: %s — meaningful_change=%s (postings Δ%s, eng Δ%s, +%d/-%d roles, "
        "%d new releases, %d new posts)",
        slug, meaningful_change, signals.posting_count_delta, signals.eng_count_delta,
        signals.new_postings, signals.removed_postings, new_release_count, new_blog_count,
    )

    return {"signals": signals, "meaningful_change": meaningful_change}


def route_after_signals(state: TrackerState) -> Literal["synthesize_dossier", "__end__"]:
    """Skip the LLM synthesis tail when nothing meaningful changed (cost control)."""
    if state.get("meaningful_change"):
        return "synthesize_dossier"
    return "__end__"


# ---------------------------------------------------------------------------
# synthesize_dossier (LLM)
# ---------------------------------------------------------------------------

def _format_pairs(pairs: list[tuple[str, int]]) -> str:
    return ", ".join(f"{label} ({n})" for label, n in pairs) or "none"


def _build_dossier_prompt(s: DossierInputs) -> str:
    releases = "\n".join(
        f"  - {r['repo']} {r['tag']}: {r['name'] or ''} ({r['published_at'] or 'n/a'})"
        for r in s.recent_releases
    ) or "  none recorded"
    posts = "\n".join(
        f"  - {p['title']} ({p['published_at'] or 'n/a'})"
        for p in s.recent_blog_posts
    ) or "  none recorded"
    stars = _format_pairs([(repo, d) for repo, d in s.star_delta_by_repo]) or "none"
    titles = "\n".join(f"  - {t}" for t in s.sample_titles) or "  none"

    return f"""Company: {s.company_name} (slug: {s.company_slug})
Snapshots of history available: {s.snapshots_available}

HEADCOUNT / HIRING VELOCITY (eng-weighted; a proxy, not true headcount):
  Total live postings: {s.posting_count} (run-over-run: {s.posting_count_delta}, \
over tracked window: {s.posting_count_window_delta})
  Eng/product/data postings: {s.eng_count} (run-over-run: {s.eng_count_delta}, \
over tracked window: {s.eng_count_window_delta})
  Roles opened this run: {s.new_postings}; closed: {s.removed_postings}

OPEN POSITIONS:
  By department: {_format_pairs(s.open_by_department)}
  Sample titles:
{titles}

TECHNOLOGY / PRODUCT (GitHub):
  New releases since last run: {s.new_release_count}
  Recent releases:
{releases}
  Star delta by repo: {stars}

PRODUCT EVOLUTION (engineering blog / changelog):
  New posts since last run: {s.new_blog_count}
  Recent posts:
{posts}

CUSTOMERS:
  No customers/logo-wall signal source is wired yet — note this as pending; do not invent customers.

Write a concise dossier for {s.company_name}. Lead with what CHANGED this run
(the deltas), not static facts. Use markdown with exactly these sections:
## Summary
One or two sentences on the company's current trajectory.

## Hiring & Headcount
What the hiring-velocity proxy and open roles say about where they're investing.

## Open Positions
Notable roles and where the org is growing.

## Technology & Product
What releases and repo activity reveal about product direction.

## Product Evolution
What the blog/changelog signals about roadmap.

## Customers
State that no customer signal is wired yet.

Be specific and cite the numbers above. Keep each section to 2–4 sentences or bullets."""


def synthesize_dossier(state: TrackerState) -> dict:
    """Write the per-company narrative across the five metrics, emphasizing deltas."""
    signals = state.get("signals")
    if signals is None:
        return {"dossier_markdown": None}

    response = SYNTHESIS_LLM.invoke([
        SystemMessage(content=(
            "You are an analyst tracking AI/data/infra startups for go-to-market and "
            "implementation roles. Write tight, factual dossiers that foreground change "
            "over time. Cite the provided numbers; never invent data you weren't given."
        )),
        HumanMessage(content=_build_dossier_prompt(signals)),
    ])
    dossier: str = response.content  # type: ignore[assignment]
    logger.info("synthesize_dossier: %s — %d-char dossier", signals.company_slug, len(dossier))
    return {"dossier_markdown": dossier}


# ---------------------------------------------------------------------------
# score_trending (deterministic composite + classification; LLM rationale only)
# ---------------------------------------------------------------------------

class _TrendRationale(BaseModel):
    """The LLM's one-sentence explanation of the (deterministic) classification."""
    rationale: str


def _growth(curr: int | None, window_delta: int | None) -> float:
    """Window growth rate: delta / base, where base is the value at window start.

    Returns 0.0 when there's no prior window or the base is non-positive — a
    company we can't yet measure a trend for contributes nothing, rather than a
    spurious spike.
    """
    if curr is None or window_delta is None:
        return 0.0
    base = curr - window_delta
    return window_delta / base if base > 0 else 0.0


def _composite(s: DossierInputs) -> float:
    """Bounded, hiring-led momentum index (~0-100). Each component is normalized
    so no single signal can saturate the score; activity terms are capped.

    Hiring carries the most weight by design (it's the core growth proxy) and
    will dominate once snapshot history is deep; the GitHub/blog activity terms
    corroborate it. See TRACKER_SCORE_WEIGHTS / _CAPS for the calibration.
    """
    w = TRACKER_SCORE_WEIGHTS
    caps = TRACKER_SCORE_CAPS
    star_growth = sum(d for _, d in s.star_delta_by_repo)

    eng_growth = _growth(s.eng_count, s.eng_count_window_delta)
    post_growth = _growth(s.posting_count, s.posting_count_window_delta)
    release_act = min(s.new_release_count, caps["release_cadence"]) / caps["release_cadence"]
    blog_act = min(s.new_blog_count, caps["blog_cadence"]) / caps["blog_cadence"]
    star_act = max(-1.0, min(star_growth / caps["star_growth"], 1.0))

    raw = (
        w["eng_velocity"] * eng_growth
        + w["posting_growth"] * post_growth
        + w["release_cadence"] * release_act
        + w["blog_cadence"] * blog_act
        + w["star_growth"] * star_act
    )
    return round(100.0 * raw, 2)


def _classify(composite: float) -> tuple[Literal["accelerating", "steady", "cooling"], bool]:
    """Map the composite onto a band. A company is a top mover iff accelerating,
    so the deterministic flag can never contradict the label (the live-run bug).
    """
    if composite >= TRACKER_ACCEL_BAND:
        return "accelerating", True
    if composite <= TRACKER_COOL_BAND:
        return "cooling", False
    return "steady", False


def score_trending(state: TrackerState) -> dict:
    """Composite momentum score + classification (both deterministic); the LLM
    only writes the rationale prose, so it cannot disagree with the score.
    """
    signals = state.get("signals")
    if signals is None:
        return {}

    composite = _composite(signals)
    classification, is_top_mover = _classify(composite)

    explain = SYNTHESIS_LLM.with_structured_output(_TrendRationale)
    out: _TrendRationale = explain.invoke([  # type: ignore[assignment]
        SystemMessage(content=(
            "A startup's momentum has already been classified as accelerating, steady, or "
            "cooling from its hiring and engineering-activity deltas. In one sentence, explain "
            "why that label fits, citing the deltas. Do not contradict the given classification."
        )),
        HumanMessage(content=(
            f"Company: {signals.company_name}\n"
            f"Classification: {classification} (composite score {composite:.1f})\n"
            f"Eng postings — run-over-run: {signals.eng_count_delta}, "
            f"over window: {signals.eng_count_window_delta}\n"
            f"Total postings — run-over-run: {signals.posting_count_delta}, "
            f"over window: {signals.posting_count_window_delta}\n"
            f"Roles opened/closed this run: {signals.new_postings}/{signals.removed_postings}\n"
            f"New releases: {signals.new_release_count}; new blog posts: {signals.new_blog_count}\n"
            f"Star deltas: {signals.star_delta_by_repo}"
        )),
    ])

    score = TrendScore(
        composite=composite,
        classification=classification,
        rationale=out.rationale,
        is_top_mover=is_top_mover,
    )
    logger.info(
        "score_trending: %s — composite=%.2f, %s, top_mover=%s",
        signals.company_slug, composite, classification, is_top_mover,
    )
    return {"trend_score": score}


# ---------------------------------------------------------------------------
# write_dossier (deterministic — Notion upsert + Linear top-mover flag)
# ---------------------------------------------------------------------------

def write_dossier(state: TrackerState) -> dict:
    """Upsert the company's Notion dossier; open/refresh a Linear task for top movers.

    An outputs failure (Notion or Linear) must not abort the map over companies,
    so each external write is guarded independently and degrades to a logged warning.
    """
    dossier = state.get("dossier_markdown")
    signals = state.get("signals")
    score = state.get("trend_score")

    if not dossier or signals is None:
        logger.info("write_dossier: no dossier to write (synthesis skipped); nothing to do")
        return {}

    url: str | None = None
    try:
        url = upsert_company_dossier(dossier, signals.company_name)
        logger.info("write_dossier: %s dossier written to Notion: %s", signals.company_slug, url)
    except Exception:
        logger.exception("write_dossier: Notion upsert failed for %s", signals.company_slug)

    if score and score.is_top_mover:
        try:
            identifier = create_top_mover_task(
                company_name=signals.company_name,
                company_slug=signals.company_slug,
                composite=score.composite,
                classification=score.classification,
                rationale=score.rationale,
                dossier_url=url,
            )
            logger.info("write_dossier: %s top-mover Linear task %s",
                        signals.company_slug, identifier)
        except Exception:
            logger.exception("write_dossier: Linear task failed for %s", signals.company_slug)

    return {"dossier_url": url}
