"""Unit tests for the embedding backfill (M-RAG.1).

Exercise the content-hash re-embed gate and batch write/failure handling with a
fake conn and a stub embeddings client — no live Postgres, pgvector, or Ollama.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion import embed  # noqa: E402
from ingestion.embed import content_hash, embed_postings, posting_text  # noqa: E402


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class _Cursor:
    def __init__(self, rows: list) -> None:
        self._rows = rows

    def fetchall(self) -> list:
        return self._rows


class _FakeConn:
    """Returns the canned rows for the SELECT; records every execute + commit."""

    def __init__(self, rows: list) -> None:
        self.rows = rows
        self.calls: list[tuple] = []
        self.commits = 0

    def execute(self, sql, params=None):  # noqa: ANN001 - test double
        self.calls.append((sql, params))
        return _Cursor(self.rows)

    def commit(self) -> None:
        self.commits += 1

    def updates(self) -> list[tuple]:
        """The recorded UPDATE calls (those carrying params — the SELECT has none)."""
        return [(sql, params) for sql, params in self.calls if params is not None]


class _FakeEmbeddings:
    def __init__(self, *, fail_calls: tuple[int, ...] = ()) -> None:
        self.calls: list[list[str]] = []
        self._fail = set(fail_calls)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        i = len(self.calls)
        self.calls.append(list(texts))
        if i in self._fail:
            raise RuntimeError("embedding backend boom")
        return [[0.1, 0.2, 0.3] for _ in texts]


def _posting_row(pk_id: str, *, model=None, chash=None, title="Software Engineer",
                 body="Build things.") -> dict:
    return {"ats": "greenhouse", "id": pk_id, "title": title, "description_text": body,
            "embedding_model": model, "content_hash": chash}


def _fresh_row(pk_id: str, **over) -> dict:
    """A row already embedded by the current model with a matching hash (should skip)."""
    row = _posting_row(pk_id, **over)
    row["embedding_model"] = embed.EMBEDDING_MODEL_NAME
    row["content_hash"] = content_hash(posting_text(row))
    return row


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------

def test_content_hash_stable_and_sensitive():
    assert content_hash("abc") == content_hash("abc")
    assert content_hash("abc") != content_hash("abd")


def test_posting_text_combines_title_and_body():
    assert posting_text({"title": "SRE", "description_text": "oncall"}) == "SRE\n\noncall"
    # tolerates missing fields
    assert posting_text({"title": None, "description_text": None}) == "\n\n"


def test_posting_text_caps_length_to_backstop():
    # A pathologically long JD is truncated so it can't overflow the embedding
    # model's context window and 400 the whole batch.
    long_body = "x" * (embed.EMBED_MAX_CHARS + 5000)
    out = posting_text({"title": "T", "description_text": long_body})
    assert len(out) == embed.EMBED_MAX_CHARS


# ---------------------------------------------------------------------------
# re-embed gate
# ---------------------------------------------------------------------------

def test_embeds_new_and_skips_unchanged(monkeypatch):
    fake = _FakeEmbeddings()
    monkeypatch.setattr(embed, "EMBEDDING_LLM", fake)
    new = _posting_row("new")                 # no hash/model → new
    unchanged = _fresh_row("old")             # hash + model match → skip

    written = embed_postings(_conn := _FakeConn([new, unchanged]))

    assert written == 1
    assert fake.calls == [[posting_text(new)]]        # only the new row embedded
    [(_sql, params)] = _conn.updates()
    assert params[3:] == ("greenhouse", "new")        # UPDATE targets the new row's PK
    assert _conn.commits == 1


def test_reembeds_edited_row(monkeypatch):
    fake = _FakeEmbeddings()
    monkeypatch.setattr(embed, "EMBEDDING_LLM", fake)
    # Same PK, current model, but a stale hash (the JD text changed since last embed).
    edited = _posting_row("e", model=embed.EMBEDDING_MODEL_NAME, chash="stalehash")

    assert embed_postings(_FakeConn([edited])) == 1
    assert fake.calls == [[posting_text(edited)]]


def test_reembeds_on_model_change(monkeypatch):
    fake = _FakeEmbeddings()
    monkeypatch.setattr(embed, "EMBEDDING_LLM", fake)
    # Hash matches its text, but a different model produced it → must re-embed.
    row = _posting_row("m")
    row["content_hash"] = content_hash(posting_text(row))
    row["embedding_model"] = "some-old-model"

    assert embed_postings(_FakeConn([row])) == 1
    assert fake.calls == [[posting_text(row)]]


def test_nothing_to_do_when_all_current(monkeypatch):
    fake = _FakeEmbeddings()
    monkeypatch.setattr(embed, "EMBEDDING_LLM", fake)
    conn = _FakeConn([_fresh_row("a"), _fresh_row("b")])

    assert embed_postings(conn) == 0
    assert fake.calls == []          # model never called
    assert conn.updates() == []      # nothing written
    assert conn.commits == 0


def test_failed_batch_skips_only_that_batch(monkeypatch):
    # One row per batch; the first batch's embed call fails, the second succeeds.
    monkeypatch.setattr(embed, "EMBED_BATCH_SIZE", 1)
    fake = _FakeEmbeddings(fail_calls=(0,))
    monkeypatch.setattr(embed, "EMBEDDING_LLM", fake)
    conn = _FakeConn([_posting_row("first"), _posting_row("second")])

    written = embed_postings(conn)

    assert written == 1                       # only the surviving batch
    [(_sql, params)] = conn.updates()
    assert params[3:] == ("greenhouse", "second")
    assert conn.commits == 1                   # the failed batch neither wrote nor committed
