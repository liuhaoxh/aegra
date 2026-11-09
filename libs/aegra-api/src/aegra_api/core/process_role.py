"""Deployment process roles: HTTP API versus job-queue worker.

``WORKER_ROLE`` is trusted deployment configuration, never a request or job
field. It only selects *where* the existing ``WorkerExecutor`` consumer loops
run; job execution itself is unchanged.

Role semantics (backward compatible with the pre-role fork deployment):

* unset — legacy single-process mode, identical to before role separation:
  the HTTP API process consumes the queue too.
* ``all`` — same legacy behaviour, stated explicitly.
* ``api`` — API-only: the API process never consumes the queue; the queue
  consumer is ``python -m aegra_api.worker``.
* ``worker`` — the standalone worker entrypoint is the designated consumer;
  the API entrypoint refuses to start (workers are started with
  ``WORKER_ROLE=worker``).

Without Redis (``REDIS_BROKER_ENABLED=false``) there is no job queue: the API
process executes runs in-process through ``LocalExecutor``. That dev-mode
behaviour is independent of ``WORKER_ROLE``.
"""

import structlog

from aegra_api.settings import settings

logger = structlog.getLogger(__name__)


class ProcessRoleError(RuntimeError):
    """A process was started with a role that conflicts with its entrypoint."""


def api_process_consumes_queue() -> bool:
    """Whether the HTTP API process must start Redis job-queue consumers.

    Fails fast when the deployment assigns the API process a worker-only role,
    so a misconfigured pod cannot silently look healthy while executing
    another process's jobs.
    """
    role = settings.worker.WORKER_ROLE
    if role is None or role == "all":
        return True
    if role == "api":
        return False
    raise ProcessRoleError(
        f"WORKER_ROLE={role!r} is worker-only; start `python -m aegra_api.worker` instead of the HTTP API process"
    )


def ensure_worker_entrypoint_role() -> None:
    """Fail fast when the standalone worker entrypoint is told to be API-only."""
    role = settings.worker.WORKER_ROLE
    if role == "api":
        raise ProcessRoleError(
            "WORKER_ROLE=api forbids starting `python -m aegra_api.worker`; "
            "unset WORKER_ROLE (or use 'worker') on the worker process"
        )
