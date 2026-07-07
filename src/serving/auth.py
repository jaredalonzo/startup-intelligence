"""Static API-key auth for the query API.

One shared key from QUERY_API_KEY (single-user phase). Comparison is
constant-time via secrets.compare_digest on encoded bytes.
"""
from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Header, HTTPException, Request


async def require_api_key(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    expected: str = request.app.state.settings.api_key
    if x_api_key is None or not secrets.compare_digest(
        x_api_key.encode(), expected.encode()
    ):
        raise HTTPException(status_code=401, detail="invalid or missing API key")
