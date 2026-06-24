"""Greenhouse ATS adapter.

Endpoint: GET https://api.greenhouse.io/v1/boards/{slug}/jobs?content=true
Returns a `jobs` array. Departments and offices are arrays; we take the first element.
Greenhouse does not provide seniority, compensation, or remote flags as structured fields.
"""
from __future__ import annotations

from typing import Any

import httpx

from ._utils import parse_dt, strip_html
from .models import Posting

_BASE = "https://api.greenhouse.io/v1/boards/{slug}/jobs"


async def fetch_postings(slug: str, client: httpx.AsyncClient) -> list[Posting]:
    resp = await client.get(_BASE.format(slug=slug), params={"content": "true"})
    resp.raise_for_status()
    return [_normalize(slug, job) for job in resp.json().get("jobs", [])]


def _normalize(company_slug: str, job: dict[str, Any]) -> Posting:
    departments = job.get("departments") or []
    return Posting(
        id=str(job["id"]),
        company_slug=company_slug,
        ats="greenhouse",
        title=job["title"],
        url=job.get("absolute_url"),
        department=departments[0].get("name") if departments else None,
        team=None,
        location=(job.get("location") or {}).get("name"),
        remote=None,
        employment_type=None,
        seniority=None,
        description_html=job.get("content"),
        description_text=strip_html(job.get("content")),
        compensation_min=None,
        compensation_max=None,
        compensation_currency=None,
        compensation_interval=None,
        posted_at=None,
        updated_at=parse_dt(job.get("updated_at")),
        raw=job,
    )
