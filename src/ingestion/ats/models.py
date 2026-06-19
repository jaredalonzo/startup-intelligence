from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

ATSSource = Literal["greenhouse", "lever", "ashby"]


class Posting(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Identity
    id: str
    company_slug: str
    ats: ATSSource

    # Core
    title: str
    url: str | None = None

    # Classification
    department: str | None = None
    team: str | None = None        # Lever/Ashby distinguish team from dept
    location: str | None = None
    remote: bool | None = None
    employment_type: str | None = None   # full-time, part-time, contract, intern
    seniority: str | None = None         # populated only when the ATS provides it structurally

    # Content
    description_html: str | None = None
    description_text: str | None = None  # plain-text strip of description_html

    # Compensation — stored as integers (minor units or raw) + metadata
    compensation_min: int | None = None
    compensation_max: int | None = None
    compensation_currency: str | None = None   # ISO 4217
    compensation_interval: str | None = None   # annual | hourly | monthly

    # Timestamps
    posted_at: datetime | None = None
    updated_at: datetime | None = None         # used as the ingestion watermark

    # Verbatim ATS payload — never modified after ingestion
    raw: dict[str, Any]
