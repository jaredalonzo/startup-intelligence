from __future__ import annotations

import html as _html_module
import re
from datetime import datetime, timezone
from html.parser import HTMLParser


class _StripHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        self._chunks.append(data)


def strip_html(raw: str | None) -> str | None:
    """Strip HTML tags from *raw*, unescaping entities first.

    Greenhouse returns entity-encoded HTML (e.g. ``&lt;div&gt;``); calling
    ``html.unescape`` before parsing ensures the tags are recognised.
    """
    if not raw:
        return None
    parser = _StripHTMLParser()
    parser.feed(_html_module.unescape(raw))
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
