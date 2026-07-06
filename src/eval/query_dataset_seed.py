"""Hand-built eval seed for the query head — checked into git, uploaded to LangSmith.

Each entry: a question, the gold documents a good retrieval should surface
(company slug + title substring — see eval.query_quality.GoldSpec), and notes
on what the question exercises. Authored 2026-07-04 against the live corpus
(2,881 embedded postings, 21 watchlist companies); gold specs use titles
verified present at that time. Postings churn — when recall drops, first check
whether the gold posting still exists before blaming retrieval.

Abstain questions carry gold: [] — the correct behavior is an explicit
abstention, which the faithfulness judge scores high and retrieval recall
treats as trivially satisfied.

Upload via: python scripts/eval_query.py --build-dataset
"""
from __future__ import annotations

from typing import Any

QUERY_EVAL_DATASET_NAME = "query-head-eval"

QUERY_EVAL_SEED: list[dict[str, Any]] = [
    # --- pure semantic ------------------------------------------------------
    {
        "question": "which companies are hiring forward-deployed engineers?",
        "gold": [
            {"company_slug": "scaleai", "title_contains": "Forward Deployed"},
            {"company_slug": "arize", "title_contains": "Forward Deployed"},
        ],
        "notes": "pure semantic; FDE is a target archetype",
    },
    {
        "question": "who is hiring engineers to work on LLM evaluation and observability?",
        "gold": [
            {"company_slug": "langchain", "title_contains": "Evals"},
            {"company_slug": "glean", "title_contains": "LLM Evals"},
        ],
        "notes": "pure semantic; evals/observability theme",
    },
    {
        "question": "customer support engineers for GPU cluster infrastructure",
        "gold": [
            {"company_slug": "togetherai", "title_contains": "Customer Support Engineer (GPU Cluster)"},
        ],
        "notes": "pure semantic; near-exact title exists",
    },
    {
        "question": "developer relations engineer roles",
        "gold": [
            {"company_slug": "sierra", "title_contains": "Developer Relations"},
            {"company_slug": "modal-labs", "title_contains": "Developer Relations"},
        ],
        "notes": "pure semantic; short canonical title",
    },
    {
        "question": "engineers working on inference infrastructure and serving",
        "gold": [
            {"company_slug": "cohere", "title_contains": "Inference Infrastructure"},
            {"company_slug": "perplexityai", "title_contains": "Inference"},
        ],
        "notes": "pure semantic; inference theme",
    },
    {
        "question": "site reliability engineer openings",
        "gold": [
            {"company_slug": "mistral", "title_contains": "Site Reliability"},
            {"company_slug": "harvey", "title_contains": "Site Reliability"},
        ],
        "notes": "pure semantic",
    },
    {
        "question": "security engineering roles at AI companies",
        "gold": [
            {"company_slug": "scaleai", "title_contains": "Security Engineer"},
            {"company_slug": "modal-labs", "title_contains": "Security Engineer"},
        ],
        "notes": "pure semantic",
    },
    {
        "question": "roles involving Rust and systems programming",
        "gold": [
            {"company_slug": "togetherai", "title_contains": "Backend Engineer"},
            {"company_slug": "scaleai", "title_contains": "Data Infrastructure"},
        ],
        "notes": "semantic where the keyword lives in the JD body, not the title",
    },
    {
        "question": "technical account manager positions",
        "gold": [
            {"company_slug": "harvey", "title_contains": "Technical Account Manager"},
            {"company_slug": "cartesia", "title_contains": "Technical Account Manager"},
        ],
        "notes": "TAM is a target archetype; beware 'Technical Accounting' distractors",
    },
    # --- filter-heavy (company) ---------------------------------------------
    {
        "question": "what is Anthropic hiring for right now?",
        "gold": [
            {"company_slug": "anthropic", "title_contains": ""},
        ],
        "notes": "company filter; any anthropic posting satisfies the gold",
    },
    {
        "question": "is Harvey hiring for trust or security work?",
        "gold": [
            {"company_slug": "harvey", "title_contains": "Trust"},
        ],
        "notes": "company filter + theme",
    },
    # --- skills join (extractions) ------------------------------------------
    {
        "question": "which companies need Kubernetes experience?",
        "gold": [
            {"company_slug": "scaleai", "title_contains": ""},
            {"company_slug": "togetherai", "title_contains": ""},
        ],
        "notes": "skills filter via latest extractions join",
    },
    # --- snapshots delta (growing_eng) ---------------------------------------
    {
        "question": (
            "which companies with growing engineering headcount are hiring "
            "infrastructure engineers?"
        ),
        "gold": [
            {"company_slug": "anthropic", "title_contains": ""},
            {"company_slug": "mistral", "title_contains": ""},
        ],
        "notes": "growing_eng snapshots CTE + semantic; gold from the 30d growth set",
    },
    # --- dossier corpus -------------------------------------------------------
    {
        "question": "which tracked companies are classified as accelerating?",
        "gold": [],
        "notes": (
            "dossier-trajectory question; dossiers table is empty until the "
            "tracker persists them — correct behavior today is abstention, and "
            "the entry starts scoring retrieval once dossiers exist"
        ),
    },
    # --- abstain --------------------------------------------------------------
    {
        "question": "which companies are hiring COBOL mainframe engineers?",
        "gold": [],
        "notes": "abstain: nothing in the corpus; near-miss retrieval must not fabricate",
    },
    {
        "question": "what is Anthropic's annual recurring revenue?",
        "gold": [],
        "notes": "abstain: financial fact the corpus cannot support",
    },
    {
        "question": "who is hiring embedded firmware engineers for automotive ECUs?",
        "gold": [],
        "notes": "abstain: plausible-sounding domain absent from an AI-startup watchlist",
    },
]
