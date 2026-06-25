---
name: model-bakeoff
description: >
  Run the skills-extraction model bake-off (scripts/eval_models.py): compare
  candidate LLMs on quality x latency, scored by an LLM-as-judge, recorded as
  LangSmith experiments. Covers candidate / judge / backend selection (local
  Ollama vs Ollama Cloud), the Cloud model catalog, judge neutrality, and reading
  results back from LangSmith. Use to evaluate or choose the extraction model.
---

# Model bake-off

Compares candidate LLMs on the **exact production extraction**
(`agents.skills.nodes.extract_posting_fields`) over a fixed dataset of real
postings, scoring each with an LLM-as-judge. One LangSmith Experiment per model;
latency is captured per example. Quality x latency, side by side.

## Prerequisites

- `.env` has `LANGSMITH_API_KEY` (results live in LangSmith), `DATABASE_URL`, and
  `OLLAMA_API_KEY` (for any Cloud model).
- The eval dataset `skills-extraction-eval` already exists. Rebuild only if asked:
  `python scripts/eval_models.py --build-dataset --sample 30`.
- For **local** candidates: a local Ollama daemon running with the models pulled
  (`ollama serve`, `ollama pull <model>`, check `ollama list`). Ollama **Cloud does
  not host the qwen2.5 models** — those are local-only (a 404 means you tried Cloud).

## Preflight first (required)

Always validate the exact model set **before** the bake-off — a full run is slow
(minutes per model) and the candidates have repeatedly failed in ways only a live
call reveals: a model not on the chosen backend (404), a subscription-gated model
(403), or structured-output non-compliance (schema echo / positional values / bad
JSON). Preflight runs the real extraction + one judge call on a few sample JDs in
~30s and tells you exactly which models are safe.

```bash
source .venv/bin/activate
python .claude/skills/model-bakeoff/preflight.py \
  --models gpt-oss:20b gemma3:12b gemma3:27b \
  --candidate-backend cloud --judge-backend cloud
```

It mirrors the bake-off flags (pass the same `--models` / `--judge-model` /
`--*-backend`). Read the result:
- **ALL CLEAR** (exit 0) → run the bake-off as-is.
- **NOT READY** (exit 1) → **drop the FAIL/PARTIAL models** from `--models` (and
  swap the judge if it failed), then bake off only the models it listed as safe.
  Do not run the full bake-off with a failing model — it just burns minutes
  producing empty extractions.

A `PARTIAL` result (e.g. 2/3) means the model is reachable but only sometimes
produces valid structured output — usable but expect a depressed score; mention
it when reporting.

## How to run

```bash
source .venv/bin/activate

# All-cloud bake-off (judge defaults to gpt-oss:120b)
python scripts/eval_models.py \
  --models gpt-oss:20b deepseek-v4-flash gemma3:12b ministral-3:8b \
  --candidate-backend cloud --judge-backend cloud

# Local candidates judged by a cloud model
python scripts/eval_models.py \
  --models qwen2.5:14b qwen2.5:7b \
  --candidate-backend local --judge-backend cloud --label baseline
```

Flags: `--models` (one experiment each), `--judge-model` (default `gpt-oss:120b`),
`--candidate-backend` / `--judge-backend` (`auto|local|cloud`; `auto` = Cloud iff
`OLLAMA_API_KEY` is set), `--label` (free-form tag). The run is serialized
(`max_concurrency=1`) to respect Ollama Cloud's one-request cap and keep the
latency comparison fair, so it is slow — minutes per model.

Each experiment is named:
`skills-extract=candidate[<model>]--<backend>|judge[<judge_model>]--<backend>[|<label>]`,
so the candidate, the judge, and where each ran are all legible (plus a random
LangSmith suffix).

## Choosing candidates (the extraction node is high-volume — favor efficient models)

| Candidate (Cloud) | Why |
| --- | --- |
| `gpt-oss:20b` | Current production extractor — the baseline to always include. |
| `deepseek-v4-flash` | Fast/cheap "flash" tier; strong reasoning. Top challenger. |
| `gemma3:12b` / `gemma3:27b` | Great instruction-following at small size. |
| `ministral-3:8b` | Smallest serious option — how cheap before quality drops. |
| `gemini-3-flash-preview` | Fast, cheap, good structured output. |
| `nemotron-3-nano:30b` | Efficient; adds family diversity. |

Skip as candidates: the frontier behemoths (`qwen3.5:397b`, `mistral-large-3:675b`,
`deepseek-v3.1:671b`) — too slow/costly for a per-posting node — and `qwen3-coder*`
(code-specialized). To list the live Cloud catalog:
`KEY=$(grep -E '^OLLAMA_API_KEY=' .env | cut -d= -f2- | tr -d '"'); curl -s https://ollama.com/api/tags -H "Authorization: Bearer $KEY"`.

## Choosing the judge (avoid family bias)

The judge should be **stronger** than the candidates and from a **different family**
than what it scores — a same-family judge is lenient and poorly discriminating (a
qwen judge scored its own siblings ~0.8 with stdev ~0.05; neutral gpt-oss:120b
spread them to stdev ~0.15 and flipped the winner).

- Default `gpt-oss:120b` is a good neutral judge for qwen / mistral / gemma candidates.
- If the candidate set **includes** `gpt-oss:20b`, pick a non-gpt-oss judge for
  neutrality — e.g. `--judge-model glm-5.2` or `--judge-model kimi-k2.6`.
- Never use a candidate as its own judge; avoid the tiny models as judges.

## Reading results back from LangSmith

The script prints a compare URL per experiment. To report quality + latency, pull
aggregate stats for the dataset's recent experiments:

```python
from dotenv import load_dotenv; load_dotenv()
from langsmith import Client
c = Client()
ds = c.read_dataset(dataset_name="skills-extraction-eval")
projs = sorted(c.list_projects(reference_dataset_id=ds.id),
               key=lambda p: p.start_time or "", reverse=True)
for p in projs[:6]:
    p = c.read_project(project_id=p.id, include_stats=True)
    fs = (p.feedback_stats or {}).get("extraction_quality", {})
    md = p.metadata or {}
    print(p.name)
    print(f"  model={md.get('model')} judge={md.get('judge_model')} "
          f"backends={md.get('candidate_backend')}/{md.get('judge_backend')}")
    print(f"  quality avg={fs.get('avg')} median={fs.get('median')} "
          f"min={fs.get('min')} max={fs.get('max')} stdev={fs.get('stdev')}")
    print(f"  runs={p.run_count} latency_p50={p.latency_p50}")
```

Note: tokens come back **0** for local Ollama models (the local path doesn't report
usage to LangSmith) — use latency as the cost proxy there; Cloud models populate tokens.

## How to report back

Lead with the verdict: which candidate wins on **quality** (judge avg, and check the
**spread** — a tight stdev near the top often means a lenient judge, not great models)
and which wins on **latency**. State the gap relative to stdev — if the quality gap is
well inside ~1 stdev, call it a tie and let latency/cost decide. Always name the judge
and flag any family overlap between judge and candidates that could bias the read.
Give the LangSmith compare URL.
