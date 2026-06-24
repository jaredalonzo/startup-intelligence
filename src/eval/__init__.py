"""Evaluation helpers — LLM-as-judge for extraction quality.

Shared by the offline bake-off (scripts/eval_models.py) and the online
evaluator over live traces (scripts/online_eval.py). Kept provider-agnostic and
dependency-injected (pass in the judge model) so the judging core is pure and
unit-testable without LangSmith or a live LLM.
"""
