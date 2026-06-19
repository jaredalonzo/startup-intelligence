from __future__ import annotations

import re
from datetime import datetime, timezone
from html.parser import HTMLParser


class _StripHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        self._chunks.append(data)


def strip_html(html: str | None) -> str | None:
    if not html:
        return None
    parser = _StripHTMLParser()
    parser.feed(html)
    text = re.sub(r"\s+", " ", "".join(parser._chunks)).strip()
    return text or None


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def ms_to_dt(ms: int | None) -> datetime | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
