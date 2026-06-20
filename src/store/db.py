"""Database connection helpers.

Reads DATABASE_URL from the environment (load .env before importing if needed).
Provides both sync and async connection factories so ingestion scripts and
LangGraph nodes can each use the appropriate interface.
"""

import os
from contextlib import contextmanager
from typing import Iterator

import psycopg
import psycopg.rows
from psycopg_pool import ConnectionPool, AsyncConnectionPool


def _url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    return url


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
