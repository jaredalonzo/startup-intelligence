"""Deterministic role classification of job postings from title + department.

One definition of "technical" shared across the project — the model bake-off
dataset builder (scripts/eval_models.py) and the posting-volume skill both use
this, so the two never drift. Title-driven: the customer-facing technical roles
(FDE/TAM/CSE) live under non-Engineering departments, so department alone is
insufficient, and an exclude pass kills GTM/ops/HR noise that trips the include
pass (e.g. "Developer Marketing", "Technical Program Manager").
"""
from __future__ import annotations

import re
from typing import Literal

# Engineering / technical-field signals.
TECH_INCLUDE = re.compile(
    r"\b(engineer|engineering|developer|forward[\s-]?deployed|solutions?\s+architect|"
    r"solutions?\s+engineer|implementation|technical\s+(account|deployment|solutions)|"
    r"platform|infrastructure|data\s+(scientist|engineer)|machine\s+learning|\bml\b|"
    r"\bai\b|security|devops|\bsre\b|reliability|architect|deploy(ment)?|"
    r"research\s+(engineer|scientist|intern))\b",
    re.IGNORECASE,
)
# Non-technical signals — checked first, so GTM/ops/HR roles are rejected even
# when they trip the include pass.
NONTECH_EXCLUDE = re.compile(
    r"\b(sales|marketing|market\b|finance|financial|legal|counsel|recruit|recruiting|"
    r"people\b|talent|account\s+executive|\bsdr\b|\bbdr\b|communications|social\s+media|"
    r"events?\s+manager|field\s+marketer|deal\s+strategy|storytelling|"
    r"program\s+manager|product\s+manager|strategist)\b",
    re.IGNORECASE,
)
# Placeholder "postings" whose title is a notice, not a role (e.g. an ATS stub
# "We have moved our Careers Page to <url>").
STUB_TITLE = re.compile(
    r"we have moved|careers?\s+page|https?://|board\s+has\s+moved", re.IGNORECASE
)

RoleClass = Literal["technical", "non_technical", "stub"]


def is_technical(title: str | None, department: str | None) -> bool:
    """True for engineering + technical-field roles; False for GTM/ops/HR/stubs."""
    title = title or ""
    if STUB_TITLE.search(title):
        return False
    text = f"{title} {department or ''}"
    if NONTECH_EXCLUDE.search(text):
        return False
    return bool(TECH_INCLUDE.search(text))


def classify(title: str | None, department: str | None) -> RoleClass:
    """Bucket a posting into technical | non_technical | stub."""
    if STUB_TITLE.search(title or ""):
        return "stub"
    return "technical" if is_technical(title, department) else "non_technical"
