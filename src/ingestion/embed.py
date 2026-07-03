"""Deterministic embedding backfill for the RAG data plane (M-RAG.1).

This is the one place ingestion touches an LLM backend, and only for embeddings —
a mechanical, deterministic step (no synthesis, no tool use). It embeds new and
edited postings + dossiers into their pgvector columns so the query head can
retrieve and cite them.

Incremental by a per-row content hash, not a timestamp watermark: a row is
(re)embedded only when its text is new, has changed, or was last embedded by a
different model. Unchanged rows never hit the model, so a run never re-embeds the
whole corpus — but an edit is caught on every source, including Ashby/Workable
boards that expose no ``updated_at`` a timestamp watermark could key on.

Run via ``scripts/embed.py`` after ingestion has populated the store.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable, Sequence

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from config import EMBEDDING_LLM, EMBEDDING_MODEL_NAME
from store.db import get_connection

logger = logging.getLogger(__name__)

# Cap per embed request so a first-run backfill doesn't build one giant call.
EMBED_BATCH_SIZE = 64

# (primary-key params, text to embed, content hash of that text)
_Pending = tuple[tuple[Any, ...], str, str]


def content_hash(text: str) -> str:
    """Stable fingerprint of the embedded text; the re-embed gate compares on it."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def posting_text(row: dict[str, Any]) -> str:
    """The text embedded for a posting: title (carries signal) then the JD body."""
    title = row.get("title") or ""
    body = row.get("description_text") or ""
    return f"{title}\n\n{body}"


def _needs_embedding(stored_hash: str | None, stored_model: str | None, text_hash: str) -> bool:
    """True when a row is new, its text changed, or a different model produced it.

    A new row has ``stored_hash is None`` (≠ text_hash), so it embeds; an edit
    changes text_hash; a model swap changes ``stored_model``. When both match, the
    stored embedding is current and the row is skipped.
    """
    return stored_hash != text_hash or stored_model != EMBEDDING_MODEL_NAME


def _embed_and_write(
    conn: psycopg.Connection[dict[str, Any]],
    pending: Sequence[_Pending],
    *,
    update_sql: str,
    label: str,
) -> int:
    """Embed the pending texts in batches and write vectors back. Returns rows written.

    Each batch is embedded then committed on its own, so progress survives a
    mid-run failure. A failed batch is logged and skipped — the rows keep their
    stale hash and are retried next run rather than being written half-embedded.
    """
    if not pending:
        logger.info("embed: %s already up to date, nothing to embed", label)
        return 0

    written = 0
    for start in range(0, len(pending), EMBED_BATCH_SIZE):
        batch = pending[start : start + EMBED_BATCH_SIZE]
        texts = [text for _pk, text, _h in batch]
        try:
            vectors = EMBEDDING_LLM.embed_documents(texts)
        except Exception:
            logger.exception(
                "embed: %s batch of %d failed; skipping (retries next run)", label, len(batch)
            )
            continue
        for (pk, _text, text_hash), vec in zip(batch, vectors):
            conn.execute(update_sql, (Vector(vec), EMBEDDING_MODEL_NAME, text_hash, *pk))
        conn.commit()
        written += len(batch)

    logger.info("embed: %s embedded %d row(s)", label, written)
    return written


def _collect_pending(
    rows: Sequence[dict[str, Any]],
    *,
    text_of: Callable[[dict[str, Any]], str],
    pk_of: Callable[[dict[str, Any]], tuple[Any, ...]],
) -> list[_Pending]:
    """Filter rows to those needing (re)embedding, pairing each with its text + hash."""
    pending: list[_Pending] = []
    for row in rows:
        text = text_of(row)
        text_hash = content_hash(text)
        if _needs_embedding(row["content_hash"], row["embedding_model"], text_hash):
            pending.append((pk_of(row), text, text_hash))
    return pending


def embed_postings(conn: psycopg.Connection[dict[str, Any]]) -> int:
    """Embed new/edited postings (title + description). Returns rows (re)embedded."""
    rows = conn.execute(
        """
        SELECT ats, id, title, description_text, content_hash, embedding_model
        FROM postings
        WHERE description_text IS NOT NULL
        """
    ).fetchall()
    pending = _collect_pending(
        rows, text_of=posting_text, pk_of=lambda r: (r["ats"], r["id"])
    )
    return _embed_and_write(
        conn,
        pending,
        update_sql="""
            UPDATE postings
            SET embedding = %s, embedding_model = %s, content_hash = %s, embedded_at = NOW()
            WHERE ats = %s AND id = %s
        """,
        label="postings",
    )


def embed_dossiers(conn: psycopg.Connection[dict[str, Any]]) -> int:
    """Embed new dossiers (dossiers are immutable, so only new rows arise). Returns count."""
    rows = conn.execute(
        """
        SELECT id, dossier_markdown, content_hash, embedding_model
        FROM dossiers
        """
    ).fetchall()
    pending = _collect_pending(
        rows, text_of=lambda r: r["dossier_markdown"], pk_of=lambda r: (r["id"],)
    )
    return _embed_and_write(
        conn,
        pending,
        update_sql="""
            UPDATE dossiers
            SET embedding = %s, embedding_model = %s, content_hash = %s, embedded_at = NOW()
            WHERE id = %s
        """,
        label="dossiers",
    )


def run_embeddings() -> dict[str, int]:
    """Embed all pending postings and dossiers. Returns per-table counts written.

    Registers the pgvector type on the connection first (the extension must already
    exist — apply_schema creates it), then runs the two incremental passes.
    """
    with get_connection() as conn:
        register_vector(conn)
        counts = {
            "postings": embed_postings(conn),
            "dossiers": embed_dossiers(conn),
        }
    logger.info("embed: run complete — %s", counts)
    return counts
