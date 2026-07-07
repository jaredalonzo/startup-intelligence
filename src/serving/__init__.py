"""HTTP serving surface for the query head (M-RAG.3).

Transport shell over `agents.query.run.run_query`; owns auth, rate limiting,
and per-request budgets/timeout. Never touches the store or the graph
directly — the query head stays read-only and cite-or-abstain by
construction. Entrypoint: scripts/serve.py (uvicorn, single worker).
"""
