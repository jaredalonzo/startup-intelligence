# CLAUDE.md

Context and conventions for working in this repo with Claude Code.

## What this project is

An agentic system with **one shared data plane and two analytical heads**:

- **Startup tracker** — builds per-company dossiers across five metrics: headcount (hiring-velocity proxy), customers, technology/product, product evolution, and open positions.
- **Skills trend agent** — aggregates engineering/technical job descriptions *across* companies to surface rising / falling / newly-appearing skills, platforms, and seniority signals relevant to FDE, TAM, CSE, and implementation roles.

Both agents read from the **same** ingestion + snapshot layer. The tracker reads it per-company; the skills agent reads it as a cross-company corpus. There is one substrate and two synthesis heads — do not duplicate ingestion per agent.

Architecture-of-record lives in Notion ("Startup & Skills Intelligence Agents"); execution/tasks live in the matching Linear project.

## Non-negotiable design principles

1. **Do not agentify deterministic work.** Fetching, dedup, snapshotting, and diffing are plain Python functions on a scheduler. LLM calls are reserved for (a) fuzzy synthesis — dossiers, skills radar — and (b) genuine tool-use — resolving a company's ATS board. If a node can be a pure function, it must be a pure function. Over-agentifying is the primary failure mode: slow, costly, nondeterministic.
2. **The value is the time series, not the snapshot.** Every metric matters only as a delta. Always snapshot-and-diff; never overwrite. Persist history.
3. **Free, clean data first.** Public ATS APIs are the foundation. Never scrape LinkedIn (ToS-hostile, litigated). Headcount is approximated by eng-weighted hiring velocity, not real headcount.
4. **Structured extraction over prose.** Any LLM extraction node returns a fixed schema (Pydantic), never free text that downstream code has to parse.

## Tech stack

- **Orchestration:** LangGraph (the two synthesis graphs). Ingestion is plain Python, not a graph.
- **Storage:** Postgres (Neon or Supabase free tier). Raw payloads in `JSONB`, queryable fields in typed columns.
- **Scheduler:** GitHub Actions cron for ingestion and scheduled agent runs.
- **Outputs:** Notion (dossiers + knowledge hub), Linear (engineering tasks + skill-gap tasks).
- **Observability:** LangSmith or Arize Phoenix — instrument both graphs.
- **LLM:** Anthropic API. Pin model strings in config, not inline.
- Deliberately **not** using a vector DB yet; counted taxonomy aggregation beats vector clustering on interpretability here.

## Proposed repo layout

```
src/
  ingestion/        # deterministic; no LLM calls
    ats/            # greenhouse.py, lever.py, ashby.py -> unified Posting schema
    signals/        # blog_rss.py, github_org.py, customers_diff.py
    watchlist.py    # company slugs + ATS resolution cache
    snapshot.py     # write snapshots
    diff.py         # compute deltas vs previous snapshot
  store/
    schema.sql      # companies, postings, snapshots, extractions
    db.py
  agents/
    skills/         # LangGraph: load_deltas, extract_skills, normalize, aggregate, synthesize, route
    tracker/        # LangGraph: resolve_board, fetch_signals, snapshot, diff, synthesize_dossier, score
    state.py        # shared TypedDict state objects
  outputs/
    notion.py       # dossier + digest writers
    linear.py       # gap-task creator
  taxonomy/
    aliases.yaml    # k8s -> Kubernetes, etc. (deterministic normalization)
  config.py
scripts/
  ingest.py         # entrypoint for scheduled ingestion
  run_skills.py     # entrypoint for the skills agent
  run_tracker.py    # entrypoint for the tracker agent
.github/workflows/  # cron schedules
tests/
```

## Common commands

```bash
# setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# run the deterministic ingestion pass (populates Postgres)
python scripts/ingest.py

# run the agents (read from the store; do not fetch ATS themselves)
python scripts/run_skills.py
python scripts/run_tracker.py

# tests / lint / types
pytest
ruff check .
mypy src
```

## Data sources (canonical)

Public, unauthenticated ATS job boards — the same endpoints their careers pages call:

| ATS | Endpoint |
| --- | --- |
| Greenhouse | `GET https://api.greenhouse.io/v1/boards/{slug}/jobs?content=true` |
| Lever | `GET https://api.lever.co/v0/postings/{slug}?mode=json` (filters: team, department, location, commitment, level) |
| Ashby | `GET https://api.ashbyhq.com/posting-api/job-board/{slug}?includeCompensation=true` |

There is **no cross-company search API** — fetch each board by slug. The watchlist is the list of slugs. Supplementary signals: engineering blog/changelog RSS, GitHub org (releases, stars), npm/PyPI download trends, and `/customers` page diffs.

## Skills agent — graph contract

`load_deltas` (det) → branch `new postings?` (→ END if none) → `extract_skills` map (LLM, `Send` fan-out, `operator.add` reducer) → `normalize_taxonomy` (det, alias table) → `aggregate_trends` (det) → `synthesize_radar` (LLM, personalized to target roles) → `route_outputs` (det). Compile with a Postgres checkpointer.

## Tracker agent — graph contract

Map over companies. Per company: `resolve_board` (LLM + tool-use, cache result) → `fetch_signals` (det, parallel) → `snapshot` (det) → `diff` (det) → branch `meaningful change?` (skip synthesis if not) → `synthesize_dossier` (LLM) → `score_trending`. Then global `rank_and_route` writes Notion dossiers + Linear tasks.

## Conventions

- Do not reference Linear ticket IDs (e.g. JAR-47) in commit messages.
- Python 3.12+, full type hints, Pydantic for all extraction schemas and the unified `Posting` model.
- Ingestion modules must be **pure and independently testable** — pass in an HTTP client, return data; no global state, no LLM imports.
- One canonical `Posting` schema; every ATS adapter normalizes into it (fields present-but-null when the source omits them).
- Secrets via env vars only (`ANTHROPIC_API_KEY`, `DATABASE_URL`, `NOTION_TOKEN`, `LINEAR_API_KEY`, `LANGSMITH_API_KEY`). Never commit them.
- Every LLM node must emit a trace span. If you add a node, instrument it.
- Watermark-based incremental reads: agents process only `updated_at > last_run`.

## Guardrails — do NOT

- Do not scrape LinkedIn or any auth-walled source.
- Do not make ingestion, dedup, normalization, diffing, or aggregation into LLM calls.
- Do not let an agent fetch ATS endpoints directly — it reads the store; ingestion writes the store.
- Do not overwrite snapshots; always append (the time series is the product).
- Do not add a vector DB without a written reason in the Notion decision log.

## Status

Phase M1 (shared ingestion + storage) first, then M2 (skills agent — the wedge), then M3 (tracker), then M4 (outputs + observability). See the Linear project for current issues.
