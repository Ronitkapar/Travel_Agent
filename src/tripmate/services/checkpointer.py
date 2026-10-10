"""Checkpoint persistence.

Two backends:

* **postgres** -- ``AsyncPostgresSaver``. Threads survive process restarts and
  are shared across instances. Requires ``DATABASE_URL``.
* **memory**   -- ``MemorySaver``. Threads live only inside the running process.
  This is the free-tier default: a cold start or restart loses any thread that
  is waiting for human approval.

PostgreSQL is always optional. If it is configured but unreachable at startup,
the planner degrades to memory instead of failing to boot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from ..config import Settings
from ..logging_config import get_logger

logger = get_logger(__name__)

# Render free-tier Postgres (and most managed providers) reject connections
# that do not negotiate TLS.
POOL_KWARGS: dict[str, Any] = {"autocommit": True, "prepare_threshold": 0}

# A hosted database that is unreachable must not stall startup: Render's health
# check has to answer, and psycopg's default 30s pool timeout is long enough to
# trip it.
POOL_CONNECT_TIMEOUT_SECONDS = 10
POOL_WAIT_TIMEOUT_SECONDS = 10


@dataclass(slots=True)
class CheckpointerHandle:
    saver: BaseCheckpointSaver
    backend: str
    detail: str = ""
    _pool: Any = field(default=None, repr=False)

    def describe(self) -> dict[str, Any]:
        return {"backend": self.backend, "detail": self.detail}

    async def aclose(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


async def create_checkpointer(settings: Settings) -> CheckpointerHandle:
    """Return the best available checkpointer, degrading to memory on failure."""

    if not settings.database_url:
        return CheckpointerHandle(
            saver=MemorySaver(),
            backend="memory",
            detail="DATABASE_URL is not set; threads do not survive a restart.",
        )

    try:
        return await _create_postgres_checkpointer(settings.database_url)
    except Exception as exc:
        logger.error(
            "PostgreSQL checkpointer unavailable (%s: %s); using in-memory state. "
            "Threads will not survive a restart.",
            type(exc).__name__,
            exc,
        )
        return CheckpointerHandle(
            saver=MemorySaver(),
            backend="memory",
            detail=(
                f"PostgreSQL checkpointer failed to start ({type(exc).__name__}); "
                "threads do not survive a restart."
            ),
        )


async def _create_postgres_checkpointer(database_url: str) -> CheckpointerHandle:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg_pool import AsyncConnectionPool

    pool = AsyncConnectionPool(
        conninfo=database_url,
        open=False,
        kwargs={"connect_timeout": POOL_CONNECT_TIMEOUT_SECONDS, **POOL_KWARGS},
        timeout=POOL_WAIT_TIMEOUT_SECONDS,
        max_size=5,
    )
    await pool.open(wait=True, timeout=POOL_WAIT_TIMEOUT_SECONDS)

    try:
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
    except Exception:
        await pool.close()
        raise

    logger.info("PostgreSQL checkpointer ready.")
    return CheckpointerHandle(saver=saver, backend="postgres", _pool=pool)
