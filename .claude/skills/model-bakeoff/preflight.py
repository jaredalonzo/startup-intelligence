"""Preflight check for the model bake-off.

Runs the REAL extraction (and one judge call) against each candidate model on a
few sample JDs, so reachability and structured-output problems surface in ~30s
instead of midway through a 30-minute bake-off. Catches the failure modes we've
hit: model not on the chosen backend (404), subscription-gated model (403), and
structured-output non-compliance (schema echo / positional values / bad JSON).

Mirrors eval_models.py's flags so you can preflight the exact command you intend
to run. Exits non-zero if any candidate or the judge fails, so a wrapper can gate.

Usage:
    python .claude/skills/model-bakeoff/preflight.py \
        --models gpt-oss:20b gemma3:12b --judge-model gpt-oss:120b \
        --candidate-backend cloud --judge-backend cloud [--samples 3]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from dotenv import load_dotenv

load_dotenv()

from agents.skills.nodes import extract_posting_fields  # noqa: E402
from eval.extraction_quality import judge_extraction  # noqa: E402
from eval.llm import build_llm  # noqa: E402

_BACKEND = {"auto": None, "local": False, "cloud": True}

# Representative JDs spanning roles/stacks — enough to exercise skills, platforms,
# seniority, and years_experience extraction and shake out structured-output quirks.
_SAMPLE_JDS = [
    {"title": "Senior Platform Engineer", "department": "Engineering",
     "description_text": "5+ years operating Kubernetes and AWS. Strong Python and "
     "Terraform; you own CI/CD and observability (Prometheus, Grafana)."},
    {"title": "Staff Backend Engineer", "department": "Engineering",
     "description_text": "Go and PostgreSQL at scale, gRPC, distributed systems on "
     "GCP. 8+ years building reliable services."},
    {"title": "Machine Learning Engineer", "department": "AI",
     "description_text": "PyTorch and JAX, training and serving LLMs on GPUs. "
     "Python. Take research models to production."},
    {"title": "Account Executive", "department": "Sales",
     "description_text": "Own a book of enterprise SaaS accounts, manage the full "
     "sales cycle, hit quota. Salesforce experience preferred."},
]


def _classify(exc: Exception) -> str:
    """Map an exception to a short, actionable reason."""
    status = getattr(exc, "status_code", None)
    if status == 404:
        return "model not available on this backend (404) — wrong backend or bad name"
    if status == 403:
        return "subscription required for this model (403) — upgrade or drop it"
    name = type(exc).__name__
    if name == "ValidationError":
        return "structured-output non-compliant (bad/echoed/positional JSON)"
    return f"{name}: {str(exc).splitlines()[0][:80]}"


def _check_model(model: str, cloud: bool | None, samples: int) -> tuple[int, str]:
    """Run `samples` extractions; return (passes, last_reason)."""
    try:
        llm = build_llm(model, cloud=cloud)
    except Exception as exc:  # construction failure (bad name, missing key)
        return 0, _classify(exc)
    passes, reason = 0, ""
    for jd in _SAMPLE_JDS[:samples]:
        try:
            extract_posting_fields(jd, llm)
            passes += 1
        except Exception as exc:
            reason = _classify(exc)
    return passes, reason


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight the model bake-off.")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Candidate models to validate (same list as the bake-off).")
    parser.add_argument("--judge-model", default="gpt-oss:120b",
                        help="Judge model to validate (default: gpt-oss:120b).")
    parser.add_argument("--candidate-backend", choices=("auto", "local", "cloud"), default="auto")
    parser.add_argument("--judge-backend", choices=("auto", "local", "cloud"), default="auto")
    parser.add_argument("--samples", type=int, default=3,
                        help="JDs per candidate (default 3; max %d)." % len(_SAMPLE_JDS))
    args = parser.parse_args()

    samples = max(1, min(args.samples, len(_SAMPLE_JDS)))
    cand_cloud = _BACKEND[args.candidate_backend]
    judge_cloud = _BACKEND[args.judge_backend]

    print(f"\nPreflight — {samples} sample JD(s) per model\n" + "=" * 60)

    ok_models, bad_models = [], []
    for model in args.models:
        passes, reason = _check_model(model, cand_cloud, samples)
        status = "PASS" if passes == samples else ("PARTIAL" if passes else "FAIL")
        (ok_models if passes == samples else bad_models).append(model)
        line = f"  [{status:7}] candidate {model:24} {passes}/{samples}"
        print(line if passes == samples else f"{line}  — {reason}")

    # Judge: one call on a known-good extraction.
    judge_ok = False
    try:
        verdict = judge_extraction(
            _SAMPLE_JDS[0]["description_text"],
            {"skills": ["Python", "Terraform"], "platforms": ["Kubernetes", "AWS"],
             "seniority": "senior", "years_experience": 5},
            build_llm(args.judge_model, cloud=judge_cloud),
        )
        judge_ok = isinstance(verdict.score, (int, float))
        print(f"  [{'PASS' if judge_ok else 'FAIL':7}] judge     {args.judge_model:24} "
              f"score={verdict.score}")
    except Exception as exc:
        print(f"  [{'FAIL':7}] judge     {args.judge_model:24} — {_classify(exc)}")

    print("=" * 60)
    if bad_models or not judge_ok:
        print(f"\nNOT READY. Drop/fix: {bad_models}" + ("" if judge_ok else " + judge"))
        print(f"Safe to bake off: {ok_models or 'none'}\n")
        sys.exit(1)
    print(f"\nALL CLEAR — {len(ok_models)} candidate(s) + judge ready. Run the bake-off.\n")


if __name__ == "__main__":
    main()
