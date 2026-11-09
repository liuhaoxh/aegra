"""Standalone job-queue worker entrypoint: ``python -m aegra_api.worker``.

Runs the same execution stack as the legacy single-process deployment —
``WorkerExecutor`` consumer loops, ``LeaseReaper``, Redis broker cancel
listener, cron scheduler — without mounting any HTTP server. The API process
(``python -m aegra_api.main`` / ``aegra serve``) keeps enqueueing jobs through
Redis; this process drains the queue.

Lifecycle:

* Startup order mirrors the API lifespan so shutdown can unwind in reverse:
  DB → observability → LangGraph registry → Redis → broker → executor
  consumers → lease reaper → cron scheduler.
* Any startup failure (DB, Redis, invalid WORKER_ROLE, graph registry) exits
  non-zero after unwinding what already started, so the container supervisor
  sees a clean crash instead of a half-alive worker.
* ``executor.stop()`` drains in-flight runs within ``WORKER_DRAIN_TIMEOUT``
  and requeues cancelled ones; the lease reaper covers anything lost to a
  hard kill.

Role configuration comes from trusted deployment env only (``WORKER_ROLE``,
``REDIS_BROKER_ENABLED``, …). ``WORKER_ROLE=api`` refuses to boot here.
"""

import asyncio
import sys
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager

import structlog

from aegra_api.core.database import db_manager
from aegra_api.core.process_role import ProcessRoleError, ensure_worker_entrypoint_role
from aegra_api.core.redis_manager import redis_manager
from aegra_api.observability.setup import setup_observability
from aegra_api.services.broker import broker_manager
from aegra_api.services.cron_scheduler import cron_scheduler
from aegra_api.services.executor import executor
from aegra_api.services.graph_lifecycle import run_graph_shutdown_hooks
from aegra_api.services.langgraph_service import get_langgraph_service
from aegra_api.services.lease_reaper import lease_reaper
from aegra_api.settings import settings
from aegra_api.utils.setup_logging import setup_logging

setup_logging()
logger = structlog.getLogger(__name__)


@asynccontextmanager
async def worker_lifespan() -> AsyncGenerator[None]:
    """Bring up the execution stack a queue consumer needs; no HTTP, no migrations."""
    try:
        await db_manager.initialize()
    except (ConnectionRefusedError, OSError) as e:
        logger.error(
            "Worker cannot connect to Postgres. Check DATABASE_URL.",
            error=str(e),
        )
        raise

    async with AsyncExitStack() as stack:
        stack.push_async_callback(db_manager.close)

        setup_observability()

        langgraph_service = get_langgraph_service()
        await langgraph_service.initialize()

        # Same contract as the API lifespan (LIFO): executor drain → graph hooks
        # → DB close.
        stack.push_async_callback(run_graph_shutdown_hooks)

        try:
            await redis_manager.initialize()
        except (ConnectionError, OSError) as e:
            logger.error(
                "Worker cannot connect to Redis. The job queue is Redis-backed; "
                "ensure Redis is running at REDIS_URL.",
                redis_url=settings.redis.REDIS_URL,
                error=str(e),
            )
            raise
        stack.push_async_callback(redis_manager.close)

        # Cancel commands published by API instances reach this worker's runs.
        await broker_manager.start()
        stack.push_async_callback(broker_manager.stop)

        await executor.start()
        stack.push_async_callback(executor.stop)

        await lease_reaper.start()
        stack.push_async_callback(lease_reaper.stop)

        if settings.cron.CRON_ENABLED:
            await cron_scheduler.start()
            stack.push_async_callback(cron_scheduler.stop)

        logger.info(
            "Worker started",
            worker_count=settings.worker.WORKER_COUNT,
            jobs_per_worker=settings.worker.N_JOBS_PER_WORKER,
            queue_key=settings.worker.WORKER_QUEUE_KEY,
        )
        yield

    logger.info("Worker stopped")


async def run_worker() -> None:
    """Run the worker until cancelled."""
    ensure_worker_entrypoint_role()
    async with worker_lifespan():
        await asyncio.Event().wait()  # Sleep forever; SIGTERM/SIGINT cancels us.


def main() -> None:
    """Console entrypoint: ``python -m aegra_api.worker``."""
    try:
        asyncio.run(run_worker())
    except (KeyboardInterrupt, ProcessRoleError):
        # SIGINT during drain or a role misconfiguration: exit non-zero so the
        # supervisor can distinguish a clean stop from a usable worker.
        sys.exit(1)


if __name__ == "__main__":
    main()
