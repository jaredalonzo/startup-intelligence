# Ingest Summary

Show what was ingested since a given date, broken down by company and ATS.

## Usage

```
/ingest-summary [date]
```

`date` is an ISO date or datetime string (e.g. `2026-06-20` or `2026-06-20T00:00:00`).
If omitted, defaults to 24 hours ago.

## Instructions

When this skill is invoked:

1. Parse the argument as a date. If no argument is given, use 24 hours ago (today's date minus 1 day). Convert any natural-language date like "yesterday" or "last week" to an ISO timestamp.

2. Run the following query via a Python one-liner (use `source .venv/bin/activate` first, and `load_dotenv()`):

```python
from dotenv import load_dotenv; load_dotenv()
from store.db import get_connection
from datetime import datetime, timezone

since = "<ISO_DATE>"  # replace with parsed date

with get_connection() as conn:
    # Total new postings since date
    total = conn.execute(
        "SELECT COUNT(*) as n FROM postings WHERE first_seen_at >= %s", (since,)
    ).fetchone()["n"]

    # Breakdown by company
    by_company = conn.execute(
        """
        SELECT company_slug, ats, COUNT(*) as n
        FROM postings
        WHERE first_seen_at >= %s
        GROUP BY company_slug, ats
        ORDER BY n DESC
        """,
        (since,),
    ).fetchall()

    # Breakdown by ATS
    by_ats = conn.execute(
        """
        SELECT ats, COUNT(*) as n
        FROM postings
        WHERE first_seen_at >= %s
        GROUP BY ats
        ORDER BY n DESC
        """,
        (since,),
    ).fetchall()

    # Snapshots taken
    snapshots = conn.execute(
        """
        SELECT company_slug, snapshot_at, posting_count, eng_count,
               cardinality(new_ids) AS new_count,
               cardinality(removed_ids) AS removed_count
        FROM snapshots
        WHERE snapshot_at >= %s
        ORDER BY snapshot_at DESC
        """,
        (since,),
    ).fetchall()

    # Companies with no new postings (in companies table but not in results above)
    ingested_slugs = {r["company_slug"] for r in by_company}
    all_companies = conn.execute("SELECT slug, ats FROM companies ORDER BY slug").fetchall()
    skipped = [r for r in all_companies if r["slug"] not in ingested_slugs]

print(f"Postings ingested since {since}: {total}")
print(f"\nBy ATS:")
for r in by_ats:
    print(f"  {r['ats']:12s}  {r['n']:4d} postings")
print(f"\nBy company:")
for r in by_company:
    print(f"  {r['company_slug']:20s}  ({r['ats']:10s})  {r['n']:4d}")
print(f"\nSnapshots taken: {len(snapshots)}")
for r in snapshots:
    print(f"  {r['company_slug']:20s}  total={r['posting_count']}  eng={r['eng_count']}  new={r['new_count']}  removed={r['removed_count']}  at={r['snapshot_at']}")
if skipped:
    print(f"\nNo new postings (skipped or not yet ingested): {[r['slug'] for r in skipped]}")
```

3. Present the results as a clean summary with:
   - Total postings ingested in the period
   - ATS breakdown (counts per provider)
   - Per-company table (company, ATS, new postings count)
   - Snapshot activity (new / removed counts per company)
   - Any companies that had no new postings ingested

Use the `scripts/` directory as the working directory context and `src/` on the Python path (the project uses `pip install -e .`).
