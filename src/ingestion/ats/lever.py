"""Lever ATS adapter.

Endpoint: GET https://api.lever.co/v0/postings/{slug}?mode=json
Returns a top-level array. Timestamps are milliseconds since epoch.
Lever is the only ATS that provides seniority (categories.level) as a structured field.
"""
from __future__ import annotations

import httpx

from ._utils import ms_to_dt
from .models import Posting

_BASE = "https://api.lever.co/v0/postings/{slug}"


async def fetch_postings(slug: str, client: httpx.AsyncClient) -> list[Posting]:
    resp = await client.get(_BASE.format(slug=slug), params={"mode": "json"})
    resp.raise_for_status()
    return [_normalize(slug, p) for p in resp.json()]


def _normalize(company_slug: str, p: dict) -> Posting:
    cats = p.get("categories") or {}
    # Lever splits description into main body + "additional" (requirements/benefits)
    html_parts = [p.get("description") or "", p.get("additional") or ""]
    description_html = "".join(html_parts) or None
    return Posting(
        id=p["id"],
        company_slug=company_slug,
        ats="lever",
        title=p["text"],
        url=p.get("hostedUrl"),
        department=cats.get("department"),
        team=cats.get("team"),
        location=cats.get("location"),
        remote=None,
        employment_type=cats.get("commitment"),
        seniority=cats.get("level"),
        description_html=description_html,
        description_text=p.get("descriptionPlain") or None,
        compensation_min=None,
        compensation_max=None,
        compensation_currency=None,
        compensation_interval=None,
        posted_at=ms_to_dt(p.get("createdAt")),
        updated_at=ms_to_dt(p.get("updatedAt")),
        raw=p,
    )
