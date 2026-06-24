"""Ashby ATS adapter.

Endpoint: GET https://api.ashbyhq.com/posting-api/job-board/{slug}?includeCompensation=true
Returns a `jobPostings` array. Ashby has the best structured compensation data of the three.
`publishedDate` is a date string (YYYY-MM-DD); `updatedAt` is a full ISO datetime.
"""
from __future__ import annotations

from typing import Any

import httpx

from ._utils import parse_dt, strip_html
from .models import Posting

_BASE = "https://api.ashbyhq.com/posting-api/job-board/{slug}"

_EMPLOYMENT_TYPE: dict[str, str] = {
    "FullTime": "full-time",
    "PartTime": "part-time",
    "Contract": "contract",
    "Intern": "intern",
    "Temporary": "temporary",
}


async def fetch_postings(slug: str, client: httpx.AsyncClient) -> list[Posting]:
    resp = await client.get(
        _BASE.format(slug=slug), params={"includeCompensation": "true"}
    )
    resp.raise_for_status()
    body = resp.json()
    # API returns "jobPostings" or "jobs" depending on board configuration
    postings = body.get("jobPostings") or body.get("jobs") or []
    return [_normalize(slug, j) for j in postings]


def _normalize(company_slug: str, j: dict[str, Any]) -> Posting:
    comp_entry = _primary_comp_entry(j.get("compensation"))
    raw_et = j.get("employmentType") or ""
    # Ashby only exposes publishedAt; no separate updatedAt on the public API.
    published = parse_dt(j.get("publishedAt") or j.get("publishedDate"))
    return Posting(
        id=j["id"],
        company_slug=company_slug,
        ats="ashby",
        title=j["title"],
        url=j.get("jobUrl") or j.get("applyUrl") or j.get("externalLink"),
        department=j.get("department") or j.get("departmentName"),
        team=j.get("team") or j.get("teamName"),
        location=j.get("location") or j.get("locationName"),
        remote=j.get("isRemote"),
        employment_type=_EMPLOYMENT_TYPE.get(raw_et, raw_et.lower() or None),
        seniority=None,
        description_html=j.get("descriptionHtml"),
        description_text=j.get("descriptionPlain") or strip_html(j.get("descriptionHtml")),
        compensation_min=_int_or_none(comp_entry, "minValue"),
        compensation_max=_int_or_none(comp_entry, "maxValue"),
        compensation_currency=comp_entry.get("currency") if comp_entry else None,
        compensation_interval=(comp_entry.get("payPeriod") or "").lower() or None
        if comp_entry
        else None,
        posted_at=published,
        updated_at=None,  # Ashby public API has no updatedAt; use first_seen_at for watermark queries
        raw=j,
    )


def _primary_comp_entry(compensation: dict[str, Any] | None) -> dict[str, Any] | None:
    if not compensation:
        return None
    entries = compensation.get("entries") or []
    # Prefer a salary entry; fall back to the first entry present
    return next((e for e in entries if e.get("type") == "Salary"), entries[0] if entries else None)


def _int_or_none(entry: dict[str, Any] | None, key: str) -> int | None:
    if not entry:
        return None
    val = entry.get(key)
    return int(val) if val is not None else None
