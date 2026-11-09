"""Run preparation logic extracted from api/runs.py.

Contains the shared run-creation helper, thread metadata updates,
resume-command validation, and config/context merging logic.
"""

import asyncio
import hashlib
import json
import os
from datetime import UTC, datetime
from functools import cache
from typing import Any
from uuid import uuid4

import structlog
from asgi_correlation_id import correlation_id
from fastapi import HTTPException
from pydantic import TypeAdapter, ValidationError
from sqlalchemy import ColumnElement, case, func, literal_column, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.config import load_checkpointer_config
from aegra_api.core.orm import Assistant as AssistantORM
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.models import Run, RunCreate, User
from aegra_api.models.run_job import (
    RunBehavior,
    RunExecution,
    RunIdentity,
    RunJob,
    UnapprovedPrincipalClaim,
    project_principal,
)
from aegra_api.models.runs import Durability
from aegra_api.services.executor import executor
from aegra_api.services.langgraph_service import get_langgraph_service
from aegra_api.services.run_status import set_thread_status
from aegra_api.settings import settings
from aegra_api.utils.assistants import resolve_assistant_id
from aegra_api.utils.jsonb import jsonb_patch, jsonb_shallow_merge
from aegra_api.utils.run_utils import _merge_jsonb

logger = structlog.getLogger(__name__)


# The interrupt reaches the client (via the broker/SSE) before the run executor
# commits thread_status="interrupted" in finalize_run. A client that resumes the
# instant it sees the interrupt can beat that commit, so after the first read poll
# a few times on FRESH sessions (each a new snapshot that sees the commit) before
# rejecting — otherwise a valid resume races to a spurious 400.
_RESUME_SETTLE_ATTEMPTS = 10
_RESUME_SETTLE_INTERVAL_SECONDS = 0.1


async def _validate_resume_command(
    session: AsyncSession, thread_id: str, command: dict[str, Any] | None, *, wait: bool = True
) -> None:
    """Validate resume command requirements.

    Guard on the presence of the ``resume`` key, not a non-None value: ``None``
    is a valid resume payload, and a ``{"resume": None}`` command must still be
    rejected against a thread that is not interrupted.

    ``wait=True`` (default) tolerates a ``finalize_run`` commit still in flight
    by polling fresh sessions; the caller must therefore hold no lock on the
    thread row. ``wait=False`` is the locked, in-transaction authoritative check.
    """
    if not command or "resume" not in command:
        return

    thread_stmt = select(ThreadORM).where(ThreadORM.thread_id == thread_id)
    thread = await session.scalar(thread_stmt)
    if not thread:
        raise HTTPException(404, f"Thread '{thread_id}' not found")
    if thread.status == "interrupted":
        return
    if not wait:
        # Called inside the admission transaction under a row lock: waiting here
        # would block finalize_run (it writes this same row), so the locked
        # state read is authoritative — no polling.
        raise HTTPException(400, "Cannot resume: thread is not in interrupted state")

    # Not interrupted on the request session's snapshot — poll fresh sessions in
    # case finalize_run's commit is still in flight. The caller holds no row
    # lock, so finalize can commit and be observed.
    maker = _get_session_maker()
    for _ in range(_RESUME_SETTLE_ATTEMPTS):
        await asyncio.sleep(_RESUME_SETTLE_INTERVAL_SECONDS)
        async with maker() as fresh:
            fresh_thread = await fresh.scalar(thread_stmt)
        if fresh_thread is not None and fresh_thread.status == "interrupted":
            return

    raise HTTPException(400, "Cannot resume: thread is not in interrupted state")


_THREAD_NAME_MAX_LENGTH = 100


def _resolve_content_text(content: Any) -> str:
    """Extract plain text from a message content field.

    Handles both plain strings and list-of-blocks format used by some SDKs::

        [{"type": "text", "text": "Hello world"}]
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(parts)
    return ""


def _extract_thread_name(input_data: dict[str, Any]) -> str:
    """Derive a thread name from the first human message in the run input.

    Supports the common ``{"messages": [{"role": "human", "content": "..."}]}``
    shape emitted by agent-chat-ui, LangGraph Studio, and the JS/Python SDKs.
    Also handles list-of-blocks content from OpenAI-compatible APIs.
    Returns an empty string when no suitable message is found.
    """
    messages = input_data.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""
    _HUMAN_ROLES = {"human", "user"}
    for msg in messages:
        raw_content: Any = None
        role: str | None = None
        if isinstance(msg, dict):
            role_val = msg.get("role")
            role = role_val if isinstance(role_val, str) else None
            if role is None:
                type_val = msg.get("type")
                role = type_val if isinstance(type_val, str) else None
            raw_content = msg.get("content")
        elif hasattr(msg, "content"):
            raw_content = getattr(msg, "content", None)
            msg_type = getattr(msg, "type", None)
            role = msg_type if isinstance(msg_type, str) else None
        if role not in _HUMAN_ROLES:
            continue
        text = _resolve_content_text(raw_content)
        if text.strip():
            name = text.strip()
            if len(name) > _THREAD_NAME_MAX_LENGTH:
                return name[:_THREAD_NAME_MAX_LENGTH].rsplit(" ", 1)[0] + "..."
            return name
    return ""


# Deployment-wide tenant for installs whose auth handler does not assert a
# tenant claim. Never read from the request: clients cannot address a tenant.
DEFAULT_TENANT_ID = os.environ.get("AEGRA_DEFAULT_TENANT_ID") or "default"


async def update_thread_metadata(
    session: AsyncSession,
    thread_id: str,
    assistant_id: str,
    graph_id: str,
    *,
    user_id: str | None = None,
    tenant_id: str = DEFAULT_TENANT_ID,
    input_data: dict[str, Any] | None = None,
) -> None:
    """Update thread metadata with assistant and graph information.

    If thread doesn't exist, auto-creates it under ``tenant_id``.
    When *input_data* is provided and the thread has no name yet, the first
    human message content is used as ``thread_name``.
    Does NOT commit — the caller controls the transaction boundary.
    """
    # This read decides whether to auto-create; the update below merges in the
    # database, so a PATCH /threads/{id} racing this run cannot drop the keys
    # written here (or have its own dropped).
    thread = await session.scalar(
        select(ThreadORM).where(ThreadORM.thread_id == thread_id, ThreadORM.tenant_id == tenant_id)
    )

    thread_name = _extract_thread_name(input_data or {})

    if not thread:
        # Auto-create thread if it doesn't exist
        if not user_id:
            raise HTTPException(400, "Cannot auto-create thread: user_id is required")

        metadata = {
            "owner": user_id,
            "assistant_id": str(assistant_id),
            "graph_id": graph_id,
            "thread_name": thread_name,
        }

        thread_orm = ThreadORM(
            thread_id=thread_id,
            status="idle",
            metadata_json=metadata,
            user_id=user_id,
            tenant_id=tenant_id,
        )
        session.add(thread_orm)
        return

    patches: list[ColumnElement[Any]] = [
        jsonb_patch({"assistant_id": str(assistant_id), "graph_id": graph_id}, "metadata_patch")
    ]
    if thread_name:
        # Only name a thread that has no name yet. The condition is evaluated at
        # write time against the row the UPDATE locks, not against the read above.
        patches.append(
            case(
                (
                    func.coalesce(ThreadORM.metadata_json["thread_name"].astext, "") == "",
                    jsonb_patch({"thread_name": thread_name}, "thread_name_patch"),
                ),
                else_=literal_column("'{}'::jsonb"),
            )
        )

    await session.execute(
        update(ThreadORM)
        .where(ThreadORM.thread_id == thread_id, ThreadORM.tenant_id == tenant_id)
        .values(
            metadata_json=jsonb_shallow_merge(ThreadORM.metadata_json, *patches),
            updated_at=datetime.now(UTC),
        )
        .execution_options(synchronize_session=False)
    )


def _resolve_checkpoint(request: RunCreate) -> dict[str, Any] | None:
    """Fold the top-level ``checkpoint_id`` into ``checkpoint``; ``checkpoint`` keys win."""
    if request.checkpoint_id is None:
        return request.checkpoint
    return {"checkpoint_id": str(request.checkpoint_id), **(request.checkpoint or {})}


#: Identity value the no-auth middleware assigns when no handler is configured.
ANONYMOUS_IDENTITY = "anonymous"

#: Reserved system principal that owns shared, tenant-independent records. An
#: authenticated caller must never be able to claim it (otherwise it could
#: publish private rows globally or overwrite system entries). Only server-side
#: registration paths may act as it, and only when explicitly enabled.
SYSTEM_IDENTITY = "system"

#: Explicit, trusted single-tenant opt-in. Only when the operator sets this may a
#: principal without a tenant claim be scoped to it; it is never a client input.
SINGLE_TENANT_ID_ENV = "AEGRA_SINGLE_TENANT_ID"

#: Comma-separated claim names the deployment's auth backend may carry into a
#: run on top of the built-in core fields (``PERSISTED_USER_FIELDS``). Claim
#: names that look like credentials are still dropped, so this list can never
#: re-admit a token or password; any claim neither built in nor listed here
#: keeps failing admission closed.
ALLOWED_PRINCIPAL_CLAIMS_ENV = "AEGRA_ALLOWED_PRINCIPAL_CLAIMS"


def configured_principal_claims() -> tuple[str, ...]:
    """Claim names approved by the deployment's auth configuration."""
    raw = os.environ.get(ALLOWED_PRINCIPAL_CLAIMS_ENV, "")
    return tuple(name.strip() for name in raw.split(",") if name.strip())


def trusted_identity(user: User) -> tuple[str, str]:
    """Resolve ``(tenant_id, owner_id)`` from the authenticated context only.

    Fail-closed contract (02-01/02-02):

    * ``identity`` must be a non-empty string that is not the no-auth
      ``anonymous`` placeholder;
    * ``is_authenticated`` must not be explicitly ``False``;
    * ``tenant_id`` must be a claim on the authenticated user dict. When it is
      missing the request is refused unless the operator has configured
      ``AEGRA_SINGLE_TENANT_ID`` — an explicit, server-side single-tenant
      decision. A multi-tenant deployment never silently scopes an unscoped
      principal to a shared tenant;
    * a non-string claim is refused as a server-side contract violation.

    Request bodies, query strings and headers are never consulted.
    """
    owner_id = user.identity
    if not isinstance(owner_id, str) or not owner_id.strip() or owner_id.strip() == ANONYMOUS_IDENTITY:
        raise HTTPException(401, "Authenticated identity required")
    if "*" in owner_id:
        # Identity is a literal that becomes a fixed store-scope label; a
        # wildcard there would turn the injected prefix into a pattern. (Caller
        # namespaces below the scope may still use LangGraph wildcards.)
        raise HTTPException(403, "Identity must not contain wildcards")
    # Strict: the User model always populates ``is_authenticated`` (default
    # True) and every construction path goes through it, so an absent/false
    # value means "not verified" and is refused — never defaulted to True.
    if owner_id.strip() == SYSTEM_IDENTITY:
        # Reserved shared owner. System records are registered directly through
        # the ORM by server-side code; no request principal may claim it, and
        # there is deliberately no environment bypass to forge it.
        raise HTTPException(403, "Reserved system identity cannot be used by a request principal")
    if user.is_authenticated is not True:
        raise HTTPException(401, "Authenticated identity required")

    claim = getattr(user, "tenant_id", None)
    if claim is None or (isinstance(claim, str) and not claim.strip()):
        # Legacy-session compatibility is explicit, never silent. Existing
        # trusted identities in this fork map to a tenant in exactly one way:
        # an operator-configured single-tenant deployment
        # (``AEGRA_SINGLE_TENANT_ID``), where every verified principal belongs
        # to that one tenant. The migration does NOT backfill tenants: rows
        # written before attribution keep a NULL tenant and are only visible
        # through that explicit single-tenant branch (see ``tenant_scope``).
        # A multi-tenant deployment (no (a) configured) refuses an unscoped
        # principal with 403: there is no verifiable mapping, and collapsing it
        # into a shared tenant would silently merge principals. The refusal is
        # the observable failure; nothing papering over it.
        single_tenant = os.environ.get(SINGLE_TENANT_ID_ENV, "").strip()
        if not single_tenant:
            raise HTTPException(
                403,
                f"Tenant claim missing on the authenticated principal; set {SINGLE_TENANT_ID_ENV} "
                "only for a deliberately single-tenant deployment",
            )
        return single_tenant, owner_id.strip()
    if not isinstance(claim, str):
        raise HTTPException(500, "Invalid tenant claim type in the authenticated context")
    tenant_id = claim.strip()
    if "*" in tenant_id:
        raise HTTPException(403, "Tenant claim must not contain wildcards")
    configured = os.environ.get(SINGLE_TENANT_ID_ENV, "").strip()
    if configured and tenant_id != configured:
        # A deployment that declared itself single-tenant has exactly one
        # tenant. A principal claiming another one is refused rather than being
        # allowed to read the unattributed legacy rows of that one tenant.
        raise HTTPException(403, "Tenant claim does not match the configured single tenant")
    return tenant_id, owner_id.strip()


def intent_digest(request: RunCreate) -> str:
    """Frozen digest of the run intent carried by one operation key.

    Covers every field that changes what executes, what is persisted, what
    side effects occur, or what events are emitted — verified against the
    RunJob/RunExecution construction and the executor's consumers:

    * execution content: ``assistant_id``, ``input``, ``command``, ``config``,
      ``context``
    * checkpoint selection: ``checkpoint`` (with ``checkpoint_id`` folded in)
    * persistence: ``durability`` (with the deprecated ``checkpoint_during``
      alias resolved the same way the executor resolves it)
    * concurrency/interruption: ``multitask_strategy``, ``interrupt_before``,
      ``interrupt_after``
    * event production: ``stream``, ``stream_mode``, ``stream_subgraphs``
    * lifecycle: ``on_disconnect``, ``on_completion``
    * run metadata: ``metadata`` (persisted into ``execution_params``/trace)

    Only pure response-transport concerns are excluded — and none of the
    RunCreate fields qualify today: even ``stream`` selects whether the
    endpoint streams, and ``on_completion`` decides whether the ephemeral
    thread is deleted. ``operation_key`` itself is the key, not the intent.

    Two requests under one key with different digests are a conflict, never a
    replay.
    """
    intent = {
        "assistant_id": request.assistant_id,
        "input": request.input,
        "command": request.command,
        "config": request.config,
        "context": request.context,
        "checkpoint": _resolve_checkpoint(request),
        "durability": _resolve_durability(request),
        "multitask_strategy": request.multitask_strategy,
        "interrupt_before": request.interrupt_before,
        "interrupt_after": request.interrupt_after,
        "stream": request.stream,
        "stream_mode": request.stream_mode,
        "stream_subgraphs": bool(request.stream_subgraphs),
        "on_disconnect": request.on_disconnect,
        "on_completion": request.on_completion,
        "metadata": request.metadata,
    }
    canonical = json.dumps(intent, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def tenant_scope(column: Any, tenant_id: str) -> ColumnElement[bool]:
    """Row-visibility predicate for a trusted tenant.

    ``column == tenant_id`` alone would make every pre-attribution row (NULL
    tenant) unreachable even for its real owner; ``col IS NULL`` alone would
    grant access to any principal. The compatibility branch is therefore
    explicit and narrow: NULL rows are visible only when the operator has
    declared a single-tenant deployment (``AEGRA_SINGLE_TENANT_ID``), where
    there is exactly one tenant by construction. Multi-tenant deployments get
    strict equality and legacy rows stay unreachable (fail-closed).
    """
    configured = os.environ.get(SINGLE_TENANT_ID_ENV, "").strip()
    # The NULL branch is only sound when the caller's tenant IS the configured
    # single tenant; otherwise an unrelated tenant would gain access to rows
    # that belong to that one tenant.
    if configured and configured == tenant_id:
        return or_(column == tenant_id, column.is_(None))
    return column == tenant_id


async def authorize_thread_scope(session: AsyncSession, thread_id: str, user: User) -> None:
    """Unlocked preflight: reject a thread bound to another party.

    Deliberately takes NO row lock: callers still have to wait for another
    transaction (e.g. ``finalize_run`` committing ``interrupted``), and holding
    the thread lock here would deadlock that writer. A foreign (tenant, owner)
    is reported exactly like a missing thread. The binding is re-verified under
    lock by :func:`bind_thread_scope` before any write.
    """
    tenant_id, owner_id = trusted_identity(user)
    row = (
        await session.execute(
            select(ThreadORM.user_id, ThreadORM.tenant_id).where(ThreadORM.thread_id == thread_id)
        )
    ).one_or_none()
    if row is not None and (row[0] != owner_id or row[1] != tenant_id):
        raise HTTPException(404, f"Thread '{thread_id}' not found")


async def bind_thread_scope(session: AsyncSession, thread_id: str, user: User) -> None:
    """Atomically bind the thread to ``(trusted tenant, owner)`` and lock it.

    A plain ``SELECT`` under READ COMMITTED proves nothing: another
    transaction may change the owner/tenant between the read and the run
    insert. So this

    1. creates the thread row if absent under the trusted scope, letting the
       primary key arbitrate concurrent creation (``ON CONFLICT DO NOTHING``);
    2. re-reads it ``FOR UPDATE``, blocking until any concurrent writer
       commits, and compares the committed scope.

    The lock is held to the end of the caller's transaction, so the admission
    write that follows cannot cross parties. Callers must only invoke this when
    they are not waiting on another transaction (see the resume-wait ordering in
    ``_prepare_run``).
    """
    tenant_id, owner_id = trusted_identity(user)
    await session.execute(
        pg_insert(ThreadORM)
        .values(thread_id=thread_id, user_id=owner_id, tenant_id=tenant_id)
        .on_conflict_do_nothing(index_elements=[ThreadORM.thread_id])
    )
    row = (
        await session.execute(
            select(ThreadORM.user_id, ThreadORM.tenant_id)
            .where(ThreadORM.thread_id == thread_id)
            .with_for_update()
        )
    ).one_or_none()
    if row is None or row[0] != owner_id or row[1] != tenant_id:
        raise HTTPException(404, f"Thread '{thread_id}' not found")


#: The only integrity failure that means "this operation key is already
#: admitted". Everything else (FK to a deleted thread, NOT NULL, another
#: unique index) must propagate.
ADMISSION_UNIQUE_INDEX = "uq_runs_tenant_owner_thread_opkey"

#: Persisted on every run row this admission code writes (execution_params).
#: Absent => pre-migration legacy row with the relaxed identity contract.
ADMISSION_SCHEMA_VERSION = 1


def _is_admission_conflict(exc: IntegrityError) -> bool:
    """True only for a violation of the run admission unique index."""
    orig = getattr(exc, "orig", None)
    # asyncpg exposes constraint_name; psycopg exposes diag.constraint_name.
    constraint = getattr(orig, "constraint_name", None) or getattr(
        getattr(orig, "diag", None), "constraint_name", None
    )
    if constraint:
        return constraint == ADMISSION_UNIQUE_INDEX
    # Fallback for drivers that only carry the message text.
    return ADMISSION_UNIQUE_INDEX in str(exc)


_DURABILITY_ADAPTER: TypeAdapter[Durability] = TypeAdapter(Durability)


@cache
def get_default_durability() -> Durability | None:
    """Server-wide durability: AEGRA_CHECKPOINT_DURABILITY wins over aegra.json checkpointer.durability.

    None when neither is set, so runs keep LangGraph's own default. An invalid
    value raises; the lifespan resolves this at startup so a typo fails the boot.
    """
    raw: object = settings.checkpointer.AEGRA_CHECKPOINT_DURABILITY
    source = "AEGRA_CHECKPOINT_DURABILITY"
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        checkpointer_config = load_checkpointer_config()
        raw = checkpointer_config.get("durability") if checkpointer_config else None
        source = "checkpointer.durability"
    if raw is None:
        return None
    try:
        return _DURABILITY_ADAPTER.validate_python(raw.strip() if isinstance(raw, str) else raw)
    except ValidationError:
        raise ValueError(f"{source} must be 'sync', 'async' or 'exit', got {raw!r}") from None


def _resolve_durability(request: RunCreate) -> Durability | None:
    """Per-run durability, then the deprecated checkpoint_during alias, then the server default."""
    if request.durability is not None:
        return request.durability
    if request.checkpoint_during is not None:
        return "async" if request.checkpoint_during else "exit"
    return get_default_durability()


async def _find_admitted_run(
    session: AsyncSession,
    tenant_id: str,
    owner_id: str,
    thread_id: str,
    operation_key: str,
) -> RunORM | None:
    """The run already admitted under this (tenant, owner, thread, key)."""
    return await session.scalar(
        select(RunORM).where(
            tenant_scope(RunORM.tenant_id, tenant_id),
            RunORM.user_id == owner_id,
            RunORM.thread_id == thread_id,
            RunORM.operation_key == operation_key,
        )
    )


def _replay(admitted: RunORM, digest: str) -> Run:
    """Build the replay response for an admitted run, fail-closed on drift.

    Same key + same digest replays the admitted run. Same key + different
    digest means the client reused a key for a different intent: HTTP 409,
    never a silent replay, never a second run.
    """
    if admitted.intent_digest != digest:
        raise HTTPException(
            409,
            f"operation_key already used by a different request for this thread "
            f"(run {admitted.run_id})",
        )
    return Run.model_validate(admitted)



async def _resubmit_if_pending(admitted: RunORM) -> RunJob | None:
    """Restore an enqueue lost in the commit->submit window.

    A run committed to Postgres but never enqueued stays ``pending`` and
    unclaimed. Replaying the operation key re-submits that SAME run_id; the
    worker's ``status='pending' AND claimed_by IS NULL`` CAS keeps duplicate
    deliveries from executing twice, so this is at-least-once, never a second
    logical run. Returns the job it submitted, or ``None`` when the run is
    already claimed/terminal (nothing to restore).
    """
    if admitted.status != "pending" or admitted.claimed_by:
        return None
    job = RunJob.from_run_orm(admitted)
    await executor.submit(job)
    logger.info("Re-enqueued admitted run still pending", run_id=admitted.run_id)
    return job

async def _prepare_run(
    session: AsyncSession,
    thread_id: str,
    request: RunCreate,
    user: User,
    *,
    initial_status: str,
    event_streaming_v2: bool = False,
) -> tuple[str, Run, RunJob | None]:

    """Shared run-creation logic used by create, stream, and wait endpoints.

    Validates inputs, authorizes the thread within the trusted (tenant, owner)
    scope, resolves the assistant, admits exactly one run per
    ``(tenant, owner, thread, operation key)``, persists it, builds a RunJob,
    submits it to the executor, and returns ``(run_id, run_model, job)``.
    A replayed request returns the already admitted run and is not enqueued
    again.
    """
    # A ``None`` job means this was a replay of an already admitted run whose
    # enqueue was not lost; callers only need run_id/run.
    tenant_id, owner_id = trusted_identity(user)
    # One principal projection for every path: the in-process execution uses it
    # now, the restart path re-validates the same stored copy. Unknown
    # non-credential claims are refused here (fail-closed) instead of being
    # silently dropped and changing the worker-side identity.
    try:
        principal = project_principal(user, allowed_claims=configured_principal_claims())
    except UnapprovedPrincipalClaim as exc:
        raise HTTPException(422, str(exc)) from None
    # Stamp the SERVER-resolved scope onto the projected principal (never a
    # client value): a single-tenant deployment resolves the tenant from
    # configuration, and the worker's cross-check compares against the row,
    # so the stored principal must carry that same value or every run would
    # be refused after restart.
    principal_data = principal.model_dump()
    principal_data["identity"] = owner_id
    principal_data["tenant_id"] = tenant_id
    principal = User.model_validate(principal_data)
    # Unlocked preflight: refuse a foreign thread before doing any work, and
    # never hold its row lock while the resume wait below can outlast a
    # finalize_run commit.
    await authorize_thread_scope(session, thread_id, user)

    # A client key is reused for retries; an absent one is server-generated and
    # therefore never deduplicated (the legacy at-least-once behaviour) — no
    # admission lookup is needed for it either, the unique index cannot fire.
    # The intent digest is persisted on EVERY new row (see ADMISSION_SCHEMA_VERSION),
    # so the worker distinguishes rows this admission code wrote from pre-migration
    # rows without guessing from a missing operation key. The partial unique index
    # still covers only keyed rows: keyless rows never deduplicate — the no-key
    # client contract is unchanged.
    digest = intent_digest(request)
    if request.operation_key:
        admitted = await _find_admitted_run(session, tenant_id, owner_id, thread_id, request.operation_key)
        if admitted is not None:
            # 409 before any enqueue when the key is reused for another intent.
            run = _replay(admitted, digest)
            return admitted.run_id, run, await _resubmit_if_pending(admitted)
        operation_key: str | None = request.operation_key
    else:
        operation_key = None

    # No lock is held here, so finalize_run can commit "interrupted" and be
    # observed by the fresh-session polls; waiting under the row lock would
    # deadlock it into a guaranteed spurious 400.
    await _validate_resume_command(session, thread_id, request.command)

    run_id = str(uuid4())
    langgraph_service = get_langgraph_service()
    logger.info(
        "Scheduling run",
        run_id=run_id,
        thread_id=thread_id,
        user_id=owner_id,
        tenant_id=tenant_id,
        status=initial_status,
    )

    # Resolve assistant / graph
    requested_id = str(request.assistant_id)
    available_graphs = langgraph_service.list_graphs()
    resolved_assistant_id = resolve_assistant_id(requested_id, available_graphs)

    # Config / context merging
    config = request.config or {}
    context = request.context or {}
    configurable = config.get("configurable", {})
    if not isinstance(configurable, dict):
        raise HTTPException(status_code=422, detail="`config.configurable` must be a mapping")

    if not context:
        context = configurable.copy()

    assistant_stmt = select(AssistantORM).where(
        AssistantORM.assistant_id == resolved_assistant_id,
        or_(AssistantORM.user_id == owner_id, AssistantORM.user_id == "system"),
    )
    assistant = await session.scalar(assistant_stmt)
    if not assistant:
        raise HTTPException(404, f"Assistant '{request.assistant_id}' not found")

    config = _merge_jsonb(assistant.config, config)
    context = _merge_jsonb(assistant.context, context)

    # Validate the assistant's graph exists
    available_graphs = langgraph_service.list_graphs()
    if assistant.graph_id not in available_graphs:
        raise HTTPException(404, f"Graph '{assistant.graph_id}' not found for assistant")

    # Bind under lock only now that nothing waits on another transaction: the
    # FOR UPDATE re-verifies scope (a concurrent owner/tenant change cannot slip
    # in) and, because finalize_run has settled, the resume state is re-checked
    # authoritatively against the locked row.
    await bind_thread_scope(session, thread_id, user)
    await _validate_resume_command(session, thread_id, request.command, wait=False)

    # Mark thread as busy and update metadata
    await update_thread_metadata(
        session,
        thread_id,
        assistant.assistant_id,
        assistant.graph_id,
        user_id=owner_id,
        tenant_id=tenant_id,
        input_data=request.input,
    )
    await set_thread_status(session, thread_id, "busy")

    # Build the RunJob before persisting so we can store execution_params
    job = RunJob(
        identity=RunIdentity(run_id=run_id, thread_id=thread_id, graph_id=assistant.graph_id),
        user=principal,
        execution=RunExecution(
            input_data=request.input,  # preserve None so LangGraph resumes from checkpoint
            config=config,
            context=context,
            stream_mode=request.stream_mode,
            checkpoint=_resolve_checkpoint(request),
            command=request.command,
            durability=_resolve_durability(request),
            event_streaming_v2=event_streaming_v2,
        ),
        behavior=RunBehavior(
            interrupt_before=request.interrupt_before,
            interrupt_after=request.interrupt_after,
            multitask_strategy=request.multitask_strategy,
            subgraphs=request.stream_subgraphs or False,
        ),
        run_metadata=request.metadata or {},
        tenant_id=tenant_id,
        operation_key=operation_key,
        admission_version=ADMISSION_SCHEMA_VERSION,
    )

    # Persist run record with trace metadata for worker observability.
    # The correlation_id from the HTTP request is stored so workers can
    # link their logs and spans back to the original request.
    exec_params = job.to_execution_params()
    exec_params["trace"] = {
        "correlation_id": correlation_id.get(""),
        "user_id": owner_id,
        "tenant_id": tenant_id,
        "thread_id": thread_id,
        "graph_id": assistant.graph_id,
    }

    now = datetime.now(UTC)
    run_orm = RunORM(
        run_id=run_id,
        thread_id=thread_id,
        assistant_id=resolved_assistant_id,
        status=initial_status,
        input=request.input,  # preserve None for checkpoint-only resume; matches RunExecution.input_data
        config=config,
        context=context,
        user_id=owner_id,
        tenant_id=tenant_id,
        operation_key=operation_key,
        intent_digest=digest,
        admission_version=ADMISSION_SCHEMA_VERSION,
        created_at=now,
        updated_at=now,
        output=None,
        error_message=None,
        execution_params=exec_params,
    )
    try:
        session.add(run_orm)
        await session.flush()
    except IntegrityError as exc:
        # Only the admission index means "someone else already admitted this
        # key". Any other integrity failure (FK, NOT NULL, a different unique
        # index) is a real error and must surface unchanged.
        if not _is_admission_conflict(exc):
            raise
        # Postgres aborts the transaction on a unique violation: roll back
        # BEFORE any further statement, or SQLAlchemy raises
        # PendingRollbackError. The rollback also discards this request's
        # thread metadata/status writes — deliberately NOT re-applied: the
        # winner's own request owns that state and a replay must add no side
        # effects.
        await session.rollback()
        admitted = await _find_admitted_run(session, tenant_id, owner_id, thread_id, operation_key)
        if admitted is None:
            raise
        # No side effects for a replay: the winner's own request already set
        # the thread metadata/status, and it may be terminal by now.
        run = _replay(admitted, digest or "")
        return admitted.run_id, run, await _resubmit_if_pending(admitted)
    await session.commit()

    run = Run.model_validate(run_orm)

    # Submit to executor
    await executor.submit(job)
    logger.info("Submitted run to executor", run_id=run_id)

    return run_id, run, job
