"""Database connection helpers.

Reads DATABASE_URL from the environment (load .env before importing if needed).
Provides both sync and async connection factories so ingestion scripts and
LangGraph nodes can each use the appropriate interface.
"""

import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

import psycopg
import psycopg.rows
from psycopg_pool import ConnectionPool, AsyncConnectionPool

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    return url


def apply_schema() -> None:
    """Apply ``schema.sql`` against the database, idempotently.

    schema.sql is written to be safe to re-run (every statement uses
    IF NOT EXISTS / CREATE TABLE IF NOT EXISTS / ADD COLUMN IF NOT EXISTS), so
    this both creates a fresh database and migrates an existing one to the
    current schema. Run it before ingestion so new columns/tables land in the
    live DB without a manual psql step — schema lives in one file, applied in
    one place.
    """
    sql = _SCHEMA_PATH.read_text()
    with get_connection() as conn:
        conn.execute(sql)
        conn.commit()


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:  # type: ignore[type-arg]
    """Context manager: yields a psycopg3 connection and always closes it on exit.

    psycopg3's Connection.__exit__ manages transactions only — it does not close
    the connection. This wrapper adds the close() so connections are never leaked.
    Transaction commit/rollback remains the caller's responsibility.
    """
    conn = psycopg.connect(_url(), row_factory=psycopg.rows.dict_row)
    try:
        yield conn
    finally:
        conn.close()


def make_pool(min_size: int = 1, max_size: int = 5) -> ConnectionPool:
    """Synchronous connection pool for ingestion scripts."""
    return ConnectionPool(_url(), min_size=min_size, max_size=max_size,
                          kwargs={"row_factory": psycopg.rows.dict_row})


async def make_async_pool(min_size: int = 1, max_size: int = 5) -> AsyncConnectionPool:
    """Async connection pool for LangGraph nodes."""
    pool = AsyncConnectionPool(_url(), min_size=min_size, max_size=max_size,
                               kwargs={"row_factory": psycopg.rows.dict_row},
                               open=False)
    await pool.open()
    return pool


# ---------------------------------------------------------------------------
# Agent watermarks (incremental-read marks, kept out of the LangGraph checkpointer)
# ---------------------------------------------------------------------------

def get_agent_watermark(agent: str) -> datetime | None:
    """Return the agent's last-run watermark, or None if it has never run.

    The skills/tracker graphs run under a fresh checkpointer thread_id per run
    (so accumulating channels don't carry across runs), so their incremental-read
    watermark must live here in the store instead of in checkpointed state.
    """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT last_run_at FROM agent_watermarks WHERE agent = %s",
            (agent,),
        ).fetchone()
    return row["last_run_at"] if row else None


def set_agent_watermark(
    agent: str,
    ts: datetime,
    conn: psycopg.Connection | None = None,  # type: ignore[type-arg]
) -> None:
    """Upsert the agent's last-run watermark.

    Pass an existing ``conn`` to advance the watermark in the same transaction
    that persists the run's output, so a later step failing can't desync them
    (the caller owns the commit). With no ``conn``, opens its own and commits.
    """
    sql = """
        INSERT INTO agent_watermarks (agent, last_run_at, updated_at)
        VALUES (%s, %s, NOW())
        ON CONFLICT (agent) DO UPDATE
            SET last_run_at = EXCLUDED.last_run_at, updated_at = NOW()
    """
    if conn is not None:
        conn.execute(sql, (agent, ts))
        return
    with get_connection() as own:
        own.execute(sql, (agent, ts))
        own.commit()
