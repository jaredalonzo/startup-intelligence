"""One-off dedup migration for the `extractions` table (JAR-69).

Before the accumulation fix (commit cb5da49) the skills graph ran under a fixed
checkpointer thread with an `operator.add` `extractions` channel, so each run
reloaded and re-persisted the previous run's accumulated extractions. The table
therefore holds duplicate/stale rows (observed extractions=176 vs postings=64 in
one run). `aggregate_trends._query_prev_counts` counts *every* row in the
previous window, so these inflate `count_previous` and skew the rising/falling
deltas as they age into the lookback window.

Dedup rule: keep the single latest extraction per (ats, posting_id); delete the
rest. The dry-run confirmed the table is entirely bug-era — all rows fall in a
two-day window and every posting was re-persisted 4-5× across both days — so
there is no genuine day-over-day history to preserve here. Collapsing to one row
per posting is what `_query_prev_counts` needs (it counts every row in the
window; one row per posting = one count per posting).

DESTRUCTIVE — dry-run by default. The DELETE runs only behind --apply, inside a
single committed transaction, after first snapshotting the table to
`extractions_backup_<utc-ts>` (unless --no-backup). The DELETE is idempotent;
re-running it deletes nothing.

Usage:
    python scripts/dedup_extractions.py                 # dry run (read-only)
    python scripts/dedup_extractions.py --apply         # backup + delete + commit
    python scripts/dedup_extractions.py --apply --no-backup
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from datetime import timedelta
from pathlib import Path

import psycopg
from dotenv import load_dotenv

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

load_dotenv()

from config import SKILLS_DEFAULT_WINDOW_DAYS  # noqa: E402
from roles import is_technical  # noqa: E402
from store.db import get_connection  # noqa: E402

# Rows ranked within their (ats, posting_id) group, newest first. rn > 1 is
# everything but the single latest keeper per posting.
_RANKED = """
    SELECT id,
           row_number() OVER (
             PARTITION BY ats, posting_id
             ORDER BY extracted_at DESC
           ) AS rn
    FROM extractions
"""

_DELETE_SQL = f"""
DELETE FROM extractions e
USING ({_RANKED}) d
WHERE e.id = d.id AND d.rn > 1
"""

_COUNT_DUPS_SQL = f"SELECT count(*) AS n FROM ({_RANKED}) d WHERE d.rn > 1"


def _scalar(conn: psycopg.Connection[dict], sql: str, params: object = None) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(next(iter(row.values()))) if row else 0


def _technical_postings_in_window(conn: psycopg.Connection[dict], window_days: int) -> int:
    """Distinct technical postings first seen within the window (reuses roles.is_technical)."""
    rows = conn.execute(
        "SELECT title, department FROM postings WHERE first_seen_at >= NOW() - %(w)s",
        {"w": timedelta(days=window_days)},
    ).fetchall()
    return sum(1 for r in rows if is_technical(r["title"], r["department"]))


def _report(conn: psycopg.Connection[dict], window_days: int) -> int:
    """Print the current state of the table. Returns the rows-to-delete count."""
    total = _scalar(conn, "SELECT count(*) AS n FROM extractions")
    distinct = _scalar(conn, "SELECT count(*) AS n FROM (SELECT DISTINCT ats, posting_id FROM extractions) s")
    to_delete = _scalar(conn, _COUNT_DUPS_SQL)
    win_extractions = _scalar(
        conn,
        "SELECT count(*) AS n FROM extractions WHERE extracted_at >= NOW() - %(w)s",
        {"w": timedelta(days=window_days)},
    )
    win_tech_postings = _technical_postings_in_window(conn, window_days)

    print(f"\n{'='*64}")
    print("extractions — current state")
    print(f"{'='*64}")
    print(f"  total rows                         {total:>8,}")
    print(f"  distinct (ats, posting_id)         {distinct:>8,}")
    print(f"  rows to delete (older dupes)       {to_delete:>8,}")
    print(f"  projected remaining after delete   {total - to_delete:>8,}")
    print(f"\n  last {window_days}d — extraction rows         {win_extractions:>8,}")
    print(f"  last {window_days}d — technical postings     {win_tech_postings:>8,}")
    print(f"{'='*64}")
    return to_delete


def _apply(conn: psycopg.Connection[dict], backup: bool) -> tuple[str | None, int]:
    """Snapshot (optional) then delete, in the caller's transaction. Returns (backup_name, deleted)."""
    backup_name: str | None = None
    if backup:
        backup_name = "extractions_backup_" + dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
        conn.execute(f'CREATE TABLE "{backup_name}" AS SELECT * FROM extractions')
        print(f"\nBackup table created: {backup_name}")
    deleted = conn.execute(_DELETE_SQL).rowcount
    return backup_name, deleted


def main() -> None:
    parser = argparse.ArgumentParser(description="Dedup the extractions table (JAR-69). Dry run unless --apply.")
    parser.add_argument("--apply", action="store_true", help="perform the backup + DELETE (destructive)")
    parser.add_argument("--no-backup", action="store_true", help="skip the backup table (only with --apply)")
    parser.add_argument("--window-days", type=int, default=SKILLS_DEFAULT_WINDOW_DAYS,
                        help="window for the verification comparison (default: %(default)s)")
    args = parser.parse_args()

    with get_connection() as conn:
        to_delete = _report(conn, args.window_days)

        if not args.apply:
            print("\nDRY RUN — no changes made. Re-run with --apply to delete.\n")
            return

        if to_delete == 0:
            print("\nNothing to delete (already deduped). No changes made.\n")
            return

        backup_name, deleted = _apply(conn, backup=not args.no_backup)
        conn.commit()
        print(f"\nDeleted {deleted:,} duplicate rows; committed.")
        if backup_name:
            print(f"Rollback if needed: INSERT INTO extractions SELECT * FROM \"{backup_name}\" "
                  f"(after TRUNCATE), or restore selectively.")
        print("\n--- post-cleanup state ---")
        _report(conn, args.window_days)


if __name__ == "__main__":
    main()
