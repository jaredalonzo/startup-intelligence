"""Workable ATS adapter.

List:   POST https://apply.workable.com/api/v2/accounts/{slug}/jobs
Detail: GET  https://apply.workable.com/api/v2/accounts/{slug}/jobs/{shortcode}

Description lives only in the detail endpoint; we fetch all details concurrently.
Workable uses `shortcode` (e.g. "F8427A442D") as the posting ID.
"""
from __future__ import annotations

import asyncio

import httpx

from ._utils import parse_dt, strip_html
from .models import Posting

_LIST_URL   = "https://apply.workable.com/api/v2/accounts/{slug}/jobs"
_DETAIL_URL = "https://apply.workable.com/api/v2/accounts/{slug}/jobs/{shortcode}"
_JOB_URL    = "https://apply.workable.com/{slug}/j/{shortcode}"

_LIST_BODY = {"query": "", "location": [], "department": [], "worktype": [], "remote": []}

_TYPE_MAP = {"full": "full-time", "part": "part-time", "contract": "contract", "intern": "internship"}


async def fetch_postings(slug: str, client: httpx.AsyncClient) -> list[Posting]:
    resp = await client.post(_LIST_URL.format(slug=slug), json=_LIST_BODY, timeout=10.0)
    resp.raise_for_status()
    shortcodes = [j["shortcode"] for j in resp.json().get("results", [])]
    details = await asyncio.gather(*[_fetch_detail(slug, sc, client) for sc in shortcodes])
    return [_normalize(slug, d) for d in details]


async def _fetch_detail(slug: str, shortcode: str, client: httpx.AsyncClient) -> dict:
    r = await client.get(_DETAIL_URL.format(slug=slug, shortcode=shortcode), timeout=10.0)
    r.raise_for_status()
    return r.json()


def _normalize(company_slug: str, job: dict) -> Posting:
    dept_list: list[str] = job.get("department") or []
    loc: dict = job.get("location") or {}
    location_str = ", ".join(filter(None, [loc.get("city"), loc.get("country")])) or None

    # Concatenate all HTML description sections
    parts = [job.get("description") or "", job.get("requirements") or "", job.get("benefits") or ""]
    desc_html = "\n".join(p for p in parts if p) or None

    return Posting(
        id=job["shortcode"],
        company_slug=company_slug,
        ats="workable",
        title=job["title"],
        url=_JOB_URL.format(slug=company_slug, shortcode=job["shortcode"]),
        department=dept_list[0] if dept_list else None,
        team=None,
        location=location_str,
        remote=job.get("remote") or job.get("workplace") == "remote" or None,
        employment_type=_TYPE_MAP.get(job.get("type", ""), job.get("type")),
        seniority=None,
        description_html=desc_html,
        description_text=strip_html(desc_html),
        compensation_min=None,
        compensation_max=None,
        compensation_currency=None,
        compensation_interval=None,
        posted_at=parse_dt(job.get("published")),
        updated_at=parse_dt(job.get("published")),  # Workable doesn't expose updated_at
        raw=job,
    )
