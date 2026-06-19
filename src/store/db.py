"""Database connection helpers.

Reads DATABASE_URL from the environment (load .env before importing if needed).
Provides both sync and async connection factories so ingestion scripts and
LangGraph nodes can each use the appropriate interface.
"""

import os
import psycopg
import psycopg.rows
from psycopg_pool import ConnectionPool, AsyncConnectionPool


def _url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    return url


def get_connection() -> psycopg.Connection:
    """Return a synchronous psycopg3 connection. Caller is responsible for closing."""
    return psycopg.connect(_url(), row_factory=psycopg.rows.dict_row)


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
