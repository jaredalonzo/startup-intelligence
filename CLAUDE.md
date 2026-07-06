# CLAUDE.md

Context and conventions for working in this repo with Claude Code.

## What this project is

An agentic system with **one shared data plane and three heads** — two analytical synthesis heads and one read-only query head:

- **Startup tracker** — builds per-company dossiers across five metrics: headcount (hiring-velocity proxy), customers, technology/product, product evolution, and open positions.
- **Skills trend agent** — aggregates engineering/technical job descriptions *across* companies to surface rising / falling / newly-appearing skills, platforms, and seniority signals relevant to FDE, TAM, CSE, and implementation roles.
- **Query head (RAG)** — an on-demand natural-language question-answering surface over the corpus (postings + dossiers). It only *reads* the store; it writes nothing back. Retrieval is **hybrid** — pgvector similarity over embedded postings/dossiers combined with SQL filters on the existing typed columns and time series (e.g. "growing eng headcount" is a `snapshots` delta filter, not a semantic match). See the decision log entry "RAG query head over the shared corpus". Built through the local CLI (`scripts/query.py`); serving surface pending — see Status (M-RAG).

All three heads read from the **same** ingestion + snapshot layer. The tracker reads it per-company; the skills agent reads it as a cross-company aggregate; the query head reads it as a retrieval index. There is one substrate and N heads — do not duplicate ingestion per head.

Architecture-of-record lives in Notion ("Startup & Skills Intelligence Agents"); execution/tasks live in the matching Linear project.

## Non-negotiable design principles

1. **Do not agentify deterministic work.** Fetching, dedup, snapshotting, and diffing are plain Python functions on a scheduler. LLM calls are reserved for (a) fuzzy synthesis — dossiers, skills radar — and (b) genuine tool-use — resolving a company's ATS board. If a node can be a pure function, it must be a pure function. Over-agentifying is the primary failure mode: slow, costly, nondeterministic.
2. **The value is the time series, not the snapshot.** Every metric matters only as a delta. Always snapshot-and-diff; never overwrite. Persist history.
3. **Free, clean data first.** Public ATS APIs are the foundation. Never scrape LinkedIn (ToS-hostile, litigated). Headcount is approximated by eng-weighted hiring velocity, not real headcount.
4. **Structured extraction over prose.** Any LLM extraction node returns a fixed schema (Pydantic), never free text that downstream code has to parse.

## Tech stack

- **Orchestration:** LangGraph (the synthesis graphs). Ingestion is plain Python, not a graph.
- **Storage:** Postgres (Neon or Supabase free tier). Raw payloads in `JSONB`, queryable fields in typed columns.
- **Retrieval:** `pgvector` **extension on the same Postgres** — an `embedding` column on postings/dossiers plus an HNSW index. Not a separate vector DB (see the decision log). Embeddings are generated deterministically in ingestion via the Ollama backend (a local embedding model); the query head's answer-synthesis model is pinned in config like every other LLM.
- **Scheduler:** GitHub Actions cron for ingestion and scheduled agent runs. The query head is *interactive* (per-request), so it needs a serving surface — the first always-on component in the system.
- **Outputs:** Notion (dossiers + knowledge hub), Linear (engineering tasks + skill-gap tasks).
- **Observability:** LangSmith or Arize Phoenix — instrument every graph.
- **LLM:** Ollama backend, pinned per-node in `config.py` (never instantiate a client in a node). Higher-quality answer synthesis for the query head may pin a stronger model there.
- **Counted taxonomy stays the analytical spine.** The query head is *additive* — it does not replace the skills head's counted aggregation with vector clustering. Aggregation remains deterministic and interpretable; retrieval is a separate read surface, not a new way to compute trends.

## Proposed repo layout

```
src/
  ingestion/        # deterministic; no LLM calls
    ats/            # greenhouse.py, lever.py, ashby.py -> unified Posting schema
    signals/        # blog_rss.py, github_org.py, customers_diff.py
    watchlist.py    # company slugs + ATS resolution cache
    snapshot.py     # write snapshots
    diff.py         # compute deltas vs previous snapshot
    embed.py        # watermarked: embed new/updated postings+dossiers (Ollama), write pgvector column
  store/
    schema.sql      # companies, postings, snapshots, extractions, dossiers, + embedding columns (pgvector)
    db.py
  agents/
    skills/         # LangGraph: load_deltas, extract_skills, normalize, aggregate, synthesize, route
      state.py      # SkillsState + extraction/trend schemas
    tracker/        # LangGraph: resolve_board, fetch_signals, snapshot, diff, synthesize_dossier, score
      state.py      # TrackerState + BoardResolution
    query/          # RAG query head (read-only): parse_query (LLM tool-use) -> retrieve (det, hybrid) -> answer (LLM, cited)
      state.py      # QueryState + parsed-filter / retrieval schemas
                    # state lives with each graph; the heads share no state objects
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

# ask a natural-language question over the corpus (query head; read-only)
python scripts/query.py "which companies are hiring Rust + distributed systems and growing eng headcount?"

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

## Query head (RAG) — graph contract

Read-only; runs per user request, not on a cron. `parse_query` (LLM tool-use — decompose the NL question into `{semantic_terms, structured_filters}`; the fuzzy→structured step, the only place an LLM touches retrieval) → `retrieve` (det — **hybrid**: pgvector similarity `ORDER BY embedding <=> q` combined with SQL `WHERE`/`JOIN` on the typed columns and `snapshots` deltas, in one query) → `answer` (LLM — grounded synthesis, **cite-or-abstain**: every claim must trace to a retrieved posting/dossier with its URL + snapshot date; never assert a trend the aggregation layer didn't compute). Embedding generation is **not** part of this graph — it is a deterministic ingestion step (`ingestion/embed.py`).

Retrieval never re-derives trends by clustering; when a question is about trends it reads the counted aggregation, not the vectors. Vectors are for *finding relevant documents*, structured columns are for *filtering*, the aggregation head is for *trends*.

## Conventions

- Do not reference Linear ticket IDs (e.g. JAR-47) in commit messages.
- Python 3.12+, full type hints, Pydantic for all extraction schemas and the unified `Posting` model.
- Ingestion modules must be **pure and independently testable** — pass in an HTTP client, return data; no global state, no LLM imports.
- One canonical `Posting` schema; every ATS adapter normalizes into it (fields present-but-null when the source omits them).
- Secrets via env vars only (`ANTHROPIC_API_KEY`, `DATABASE_URL`, `NOTION_TOKEN`, `LINEAR_API_KEY`, `LANGSMITH_API_KEY`). Never commit them.
- Every LLM node must emit a trace span. If you add a node, instrument it.
- Watermark-based incremental reads: agents process only `updated_at > last_run`. Embedding generation follows the same discipline — embed only new/updated postings and dossiers, never re-embed the whole corpus per run.
- The query head reads the store; it does not write to it. Answers must cite their sources (posting URL / dossier + snapshot date). No citation → don't assert it.

## Guardrails — do NOT

- Do not scrape LinkedIn or any auth-walled source.
- Do not make ingestion, dedup, normalization, diffing, or aggregation into LLM calls.
- Do not let an agent fetch ATS endpoints directly — it reads the store; ingestion writes the store.
- Do not overwrite snapshots; always append (the time series is the product).
- `pgvector` (extension on the existing Postgres) is **approved** for the query head — see the decision log entry "RAG query head over the shared corpus". A **standalone/separate vector DB** (Weaviate, Pinecone, Qdrant, etc.) still requires its own written reason in the decision log; it duplicates the store and breaks the one-substrate principle, so the bar is high.
- Do not make embedding, hybrid retrieval, or metadata filtering into LLM calls — only query parsing (fuzzy→structured) and final answer synthesis are LLM nodes in the query head.
- Do not let the query head compute trends by clustering vectors; trends come from the counted aggregation head. Retrieval finds documents, it does not replace the analytical spine.
- Do not let the query head write to the store, and do not let it emit ungrounded claims (cite-or-abstain).

## Status

Phase M1 (shared ingestion + storage) first, then M2 (skills agent — the wedge), then M3 (tracker), then M4 (outputs + observability). See the Linear project for current issues.

**M-RAG (query head)** — phased so retrieval is proven before any serving infra:
- **M-RAG.1** — **done.** Data plane: `pgvector` extension + `embedding` columns, `ingestion/embed.py` (hash-gated incremental embed), and tracker dossiers persisted into the store so they can be embedded. Embedding model: `granite-embedding:278m` (replaced nomic-embed-text, whose Ollama build produced degenerate geometry — see config.py); model-specific task prefixes are pinned in `config.py` (empty for granite).
- **M-RAG.2** — **done.** Hybrid retrieval + grounded answer as a **local CLI** (`scripts/query.py`; graph in `src/agents/query/`), with retrieval-recall + answer-faithfulness evals reusing the existing LLM-as-judge harness (`src/eval/`, dataset seed in `src/eval/query_dataset_seed.py`, runner `scripts/eval_query.py`).
- **M-RAG.3** — the external serving surface (the first always-on component; leaning Slack for built-in auth + rate limiting). Next up.
