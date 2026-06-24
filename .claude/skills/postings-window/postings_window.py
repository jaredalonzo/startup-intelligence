"""How many postings were ingested within a time window — with a role breakdown.

Read-only. Counts postings by first_seen_at and classifies each (from title +
department) as technical / non-technical / stub, using the project's shared
roles classifier so "technical" matches the bake-off dataset's definition.

Usage:
    python .claude/skills/postings-window/postings_window.py            # ladder: 1 3 7 14 30
    python .claude/skills/postings-window/postings_window.py 7          # focus on 7 days
    python .claude/skills/postings-window/postings_window.py 3 7 30
"""
from __future__ import annotations

import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from dotenv import load_dotenv

load_dotenv()

from roles import classify  # noqa: E402
from store.db import get_connection  # noqa: E402

_BUCKETS = ("technical", "non_technical", "stub")


def _pct(part: int, whole: int) -> str:
    return f"{(100 * part / whole):.0f}%" if whole else "—"


def main() -> None:
    args = sys.argv[1:]
    try:
        windows = sorted({int(a) for a in args}) if args else [1, 3, 7, 14, 30]
    except ValueError:
        print("Windows must be integers (days). e.g. postings_window.py 3 7 30")
        sys.exit(2)

    focus = int(args[0]) if args else (7 if 7 in windows else windows[-1])
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max(windows))

    # Pull the widest window once; slice the narrower windows in memory.
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT company_slug, title, department, first_seen_at "
            "FROM postings WHERE first_seen_at > %s",
            (cutoff,),
        ).fetchall()

    for r in rows:
        r["_class"] = classify(r["title"], r["department"])

    print("Postings by time window (role classified from title + department)\n")
    print(f"{'window':>8}  {'total':>6}  {'technical':>10}  {'non-tech':>9}  {'stub':>5}  {'tech%':>6}")
    print("-" * 56)
    for d in windows:
        wc = now - timedelta(days=d)
        sub = [r for r in rows if r["first_seen_at"] > wc]
        counts = Counter(r["_class"] for r in sub)
        total = len(sub)
        print(f"{d:>6}d  {total:>6}  {counts['technical']:>10}  "
              f"{counts['non_technical']:>9}  {counts['stub']:>5}  "
              f"{_pct(counts['technical'], total):>6}")

    # Per-company breakdown for the focus window.
    wc = now - timedelta(days=focus)
    focus_rows = [r for r in rows if r["first_seen_at"] > wc]
    per: dict[str, Counter] = {}
    for r in focus_rows:
        per.setdefault(r["company_slug"], Counter())[r["_class"]] += 1

    print(f"\nPer-company — {focus}-day window ({len(focus_rows)} postings):")
    print(f"  {'company':16} {'total':>6}  {'tech':>5}  {'non-tech':>9}  {'stub':>5}")
    for slug, c in sorted(per.items(), key=lambda kv: -sum(kv[1].values())):
        tot = sum(c.values())
        print(f"  {slug:16} {tot:>6}  {c['technical']:>5}  "
              f"{c['non_technical']:>9}  {c['stub']:>5}")


if __name__ == "__main__":
    main()
