---
name: postings-window
description: >
  Show how many job postings were ingested within a look-back window, broken down
  by role: technical / non-technical / stub, overall and per company. Read-only —
  counts only, never runs an agent or writes anything. Use to gauge posting volume
  for a time window, or to see the technical-vs-noise mix in the corpus (the same
  "technical" definition the model bake-off dataset uses, via src/roles.py).
---

# Postings by time window

Quick read on corpus volume and composition for a given look-back window.

## How to run

```bash
python .claude/skills/postings-window/postings_window.py            # ladder: 1 3 7 14 30
python .claude/skills/postings-window/postings_window.py 7          # one window
python .claude/skills/postings-window/postings_window.py 3 7 30     # specific windows
```

Pass any window(s) the user names as day counts. With no args it scans a ladder
(1/3/7/14/30) so the trend is visible. It reads `DATABASE_URL` from `.env` and
runs only `SELECT`s.

## What it reports

- A per-window table: **total** postings (`first_seen_at` within the window) and
  the split into **technical / non-technical / stub** plus the technical share.
- A per-company breakdown for the focus window (the first window argument, else 7d).

Classification is deterministic, from title + department, via `src/roles.py`
(`classify`) — the same gate the bake-off dataset uses, so "technical" is
consistent project-wide.

## How to report back

Lead with the headline: total postings for the window the user cares about and the
technical share. Call out anything notable — e.g. a large non-technical fraction
(the corpus isn't role-filtered, so GTM/recruiting postings are counted), or a
window where the total jumps sharply (the one-time initial-ingest backfill, since
`first_seen_at` is when we ingested a posting, not when it was posted).
