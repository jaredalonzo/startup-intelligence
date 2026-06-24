"""Snapshot and posting persistence for the ingestion pipeline."""
from __future__ import annotations

import re
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from ingestion.ats.models import Posting

# Departments / titles that count toward the engineering hiring-velocity proxy.
_ENG_RE = re.compile(
    r"\b(engineer|engineering|software|data|ml|machine.?learn|ai|infra|"
    r"platform|backend|frontend|full.?stack|devops|sre|security|"
    r"product|research|scientist|science)\b",
    re.IGNORECASE,
)


def _is_eng(p: Posting) -> bool:
    return bool(
        _ENG_RE.search(p.department or "")
        or _ENG_RE.search(p.team or "")
        or _ENG_RE.search(p.title)
    )


def upsert_postings(postings: list[Posting], conn: psycopg.Connection[dict[str, Any]]) -> None:
    """Insert new postings or update existing ones; always advances last_seen_at."""
    for p in postings:
        conn.execute(
            """
            INSERT INTO postings (
                ats, id, company_slug, title, url, department, team, location,
                remote, employment_type, seniority, description_html, description_text,
                compensation_min, compensation_max, compensation_currency, compensation_interval,
                posted_at, updated_at, last_seen_at, raw
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, NOW(), %s
            )
            ON CONFLICT (ats, id) DO UPDATE SET
                title                = EXCLUDED.title,
                url                  = EXCLUDED.url,
                department           = EXCLUDED.department,
                team                 = EXCLUDED.team,
                location             = EXCLUDED.location,
                remote               = EXCLUDED.remote,
                employment_type      = EXCLUDED.employment_type,
                seniority            = EXCLUDED.seniority,
                description_html     = EXCLUDED.description_html,
                description_text     = EXCLUDED.description_text,
                compensation_min     = EXCLUDED.compensation_min,
                compensation_max     = EXCLUDED.compensation_max,
                compensation_currency = EXCLUDED.compensation_currency,
                compensation_interval = EXCLUDED.compensation_interval,
                posted_at            = EXCLUDED.posted_at,
                updated_at           = EXCLUDED.updated_at,
                last_seen_at         = NOW(),
                raw                  = EXCLUDED.raw
            """,
            (
                p.ats, p.id, p.company_slug, p.title, p.url,
                p.department, p.team, p.location,
                p.remote, p.employment_type, p.seniority,
                p.description_html, p.description_text,
                p.compensation_min, p.compensation_max,
                p.compensation_currency, p.compensation_interval,
                p.posted_at, p.updated_at,
                Jsonb(p.raw),
            ),
        )


def write_snapshot(
    company_slug: str,
    current_postings: list[Posting],
    new_ids: list[str],
    removed_ids: list[str],
    conn: psycopg.Connection[dict[str, Any]],
) -> int:
    """Append a snapshot row; never updates existing rows. Returns the new snapshot ID."""
    eng_count = sum(1 for p in current_postings if _is_eng(p))
    row = conn.execute(
        """
        INSERT INTO snapshots (company_slug, posting_count, eng_count, new_ids, removed_ids)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """,
        (company_slug, len(current_postings), eng_count, new_ids, removed_ids),
    ).fetchone()
    assert row is not None
    return int(row["id"])


def update_watermark(
    company_slug: str,
    ats: str,
    conn: psycopg.Connection[dict[str, Any]],
) -> None:
    conn.execute(
        """
        INSERT INTO watermarks (company_slug, ats, last_fetched_at)
        VALUES (%s, %s, NOW())
        ON CONFLICT (company_slug, ats) DO UPDATE SET last_fetched_at = NOW()
        """,
        (company_slug, ats),
    )
