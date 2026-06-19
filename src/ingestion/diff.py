"""Delta computation for the ingestion pipeline.

compute_diff must be called BEFORE upsert_postings — it relies on last_seen_at
still reflecting the previous run. After upsert_postings runs, last_seen_at is
updated to NOW() for all currently-live postings, making the comparison invalid.
"""
from __future__ import annotations

import psycopg


def compute_diff(
    company_slug: str,
    ats: str,
    current_ids: set[str],
    conn: psycopg.Connection,  # type: ignore[type-arg]
) -> tuple[list[str], list[str]]:
    """Return (new_ids, removed_ids) relative to the previous ingestion run.

    On the first run for a company there is no watermark, so every current
    posting is treated as new and removed_ids is empty.
    """
    wm = conn.execute(
        "SELECT last_fetched_at FROM watermarks WHERE company_slug = %s AND ats = %s",
        (company_slug, ats),
    ).fetchone()

    if wm is None:
        return list(current_ids), []

    prev_live: set[str] = {
        row["id"]
        for row in conn.execute(
            """
            SELECT id FROM postings
            WHERE company_slug = %s AND ats = %s AND last_seen_at >= %s
            """,
            (company_slug, ats, wm["last_fetched_at"]),
        ).fetchall()
    }

    return list(current_ids - prev_live), list(prev_live - current_ids)
