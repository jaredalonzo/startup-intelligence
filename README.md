# Startup & Skills Intelligence Agents

An agentic system with one shared data plane and two analytical heads:

- **Startup tracker** — per-company dossiers across headcount (hiring-velocity proxy), customers, technology/product, product evolution, and open positions.
- **Skills trend agent** — aggregates engineering job descriptions across companies to surface rising/falling/new skills, platforms, and seniority signals relevant to FDE, TAM, CSE, and implementation roles.

Both agents read from the same ingestion layer. The tracker reads it per-company; the skills agent reads it as a cross-company corpus.

## Architecture

```
Ingestion (plain Python, no LLM)
  └── ATS fetchers (Greenhouse / Lever / Ashby / Workable)
  └── Watchlist + ATS resolver
  └── Snapshot + diff (append-only time series)
        │
        ▼
  Postgres
  companies / postings / snapshots / extractions / watermarks
        │
        ├── Skills agent (LangGraph)
        │     load_deltas → extract_skills → normalize_taxonomy
        │     → aggregate_trends → synthesize_radar → route_outputs
        │
        └── Tracker agent (LangGraph)
              resolve_board → fetch_signals → snapshot → diff
              → synthesize_dossier → score_trending → rank_and_route
```

## Watchlist

21 companies across 4 ATSs:

| Company | ATS | Slug |
|---------|-----|------|
| Anthropic | Greenhouse | anthropic |
| Arize AI | Greenhouse | arizeai |
| Cognition | Greenhouse | cognitionlabs |
| Glean | Greenhouse | gleanwork |
| Scale AI | Greenhouse | scaleai |
| Together AI | Greenhouse | togetherai |
| Anyscale | Lever | anyscale |
| Mistral AI | Lever | mistral |
| Cartesia AI | Ashby | cartesia |
| Character.AI | Ashby | character |
| Cohere | Ashby | cohere |
| Harvey | Ashby | harvey |
| LangChain | Ashby | langchain |
| Modal | Ashby | modal |
| OpenAI | Ashby | openai |
| Perplexity AI | Ashby | perplexity |
| Pinecone | Ashby | pinecone |
| Runway | Ashby | runway |
| Sierra | Ashby | sierra |
| Weaviate | Ashby | weaviate |
| Hugging Face | Workable | huggingface |

## Setup

**Prerequisites:** Python 3.12+, a Postgres instance.

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Configure
cp .env.example .env   # then fill in DATABASE_URL and API keys

# Apply schema
psql $DATABASE_URL -f src/store/schema.sql

# Run a manual ingestion pass
python scripts/ingest.py
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | Postgres connection string |
| `ANTHROPIC_API_KEY` | M2+ | For LLM extraction and synthesis nodes |
| `NOTION_TOKEN` | M4 | Dossier and digest writer |
| `LINEAR_API_KEY` | M4 | Skill-gap task creator |
| `LANGSMITH_API_KEY` | M4 | Trace observability |

## Repo layout

```
src/
  ingestion/
    ats/              # Greenhouse, Lever, Ashby, Workable adapters → Posting model
    watchlist.py      # Company seed list + 4-way ATS probe + DB cache
    snapshot.py       # upsert_postings, write_snapshot, update_watermark
    diff.py           # compute_diff (watermark-based, run before upsert)
  store/
    schema.sql        # companies, postings, snapshots, extractions, watermarks
    db.py             # sync/async connection factories
  agents/             # (M2/M3) LangGraph skills + tracker graphs
  outputs/            # (M4) Notion + Linear writers
  taxonomy/           # (M2) aliases.yaml for skill normalization
scripts/
  ingest.py           # Scheduled ingestion entrypoint
  run_skills.py       # (M2) Skills agent entrypoint
  run_tracker.py      # (M3) Tracker agent entrypoint
.github/workflows/
  ingest.yml          # Daily cron at 08:00 UTC
```

## CI / scheduled runs

The ingestion cron runs daily at 08:00 UTC via GitHub Actions. Requires a `DATABASE_URL` secret in the repo's Actions settings (Settings → Secrets → Actions → New repository secret).

Can also be triggered manually from the Actions tab via **workflow_dispatch**.

## Development

```bash
pytest          # run tests
ruff check .    # lint
mypy src        # type check
```
