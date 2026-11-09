"""Real-Postgres behavior tests for FollowMe run admission (02-02 Task 1).

These exercise actual SQLAlchemy/asyncpg semantics — no mocks of the session,
no mocked rows — because the defects they guard against only exist in the real
driver:

* ``authorize_thread_scope`` must read a Row(owner, tenant); a ``scalar()``
  that returns only the first column would reject every protected thread or
  let a foreign tenant through.
* A unique-violation aborts the Postgres transaction; the recovery path must
  roll back *before* reading the winner, or SQLAlchemy raises
  PendingRollbackError. Two independent connections prove the losing side.
* ``_is_admission_conflict`` must match only the admission index.

Environment contract: this module never starts a database. It requires the
integrator to point ``AEGRA_TEST_DATABASE_URL`` (asyncpg URL) at an isolated
instance inside the 01-01 port profile, with the migration chain at head. If
the variable is absent the module skips with an explicit reason instead of
pretending to have verified anything.
"""

from __future__ import annotations

import os
import re
from collections.abc import AsyncIterator
from urllib.parse import urlsplit

import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from aegra_api.core.orm import Assistant as AssistantORM
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.models.auth import User
from aegra_api.services.run_preparation import (
    ADMISSION_UNIQUE_INDEX,
    _is_admission_conflict,
    authorize_thread_scope,
    bind_thread_scope,
)

pytestmark = pytest.mark.asyncio

DEFAULT_TENANT = "default"

# Payload principal tenant: "use the row's own tenant", i.e. what production
# projects into the stored principal. ``None`` is the explicit "no claim".
_ROW_TENANT = object()


def _database_url() -> str:
    url = os.environ.get("AEGRA_TEST_DATABASE_URL", "").strip()
    if not url:
        pytest.skip(
            "AEGRA_TEST_DATABASE_URL not set: real-Postgres admission checks are executed by the "
            "integrator under the 01-01 environment contract (isolated instance, migration at head)"
        )
    return url


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    url = _database_url()
    # Isolation assertion: refuse a database that is not obviously disposable.
    db_name = urlsplit(url).path.lstrip("/") or ""
    if os.environ.get("AEGRA_TEST_ALLOW_ANY_DB") != "1":
        assert "test" in db_name, f"refusing to run DB tests against non-test database {db_name!r}"

    eng = create_async_engine(url, future=True)
    try:
        async with eng.connect() as conn:
            actual = (await conn.execute(text("select current_database()"))).scalar_one()
            assert actual == db_name, f"connected to {actual!r}, expected {db_name!r}"
            # Fail loudly (not skip) when the caller's DB is not migrated: a
            # green run here must mean the migration was really applied.
            row = (
                await conn.execute(
                    text("select indexdef from pg_indexes where indexname = :n"),
                    {"n": ADMISSION_UNIQUE_INDEX},
                )
            ).first()
            assert row is not None, f"missing index {ADMISSION_UNIQUE_INDEX}: run alembic upgrade head"
            assert "UNIQUE" in row[0] and "operation_key IS NOT NULL" in row[0], row[0]
        yield eng
    finally:
        await eng.dispose()


@pytest_asyncio.fixture
async def maker(engine: AsyncEngine) -> async_sessionmaker:
    return async_sessionmaker(engine, expire_on_commit=False)


async def _seed_thread(maker: async_sessionmaker, *, thread_id: str, owner: str, tenant: str) -> None:
    """Create one thread under an explicit scope, replacing any stale row.

    The suite runs against a persistent database: a row left behind by an
    aborted run must not turn the next run into a ``thread_pkey`` duplicate.
    """
    async with maker() as session:
        await session.execute(text("delete from thread where thread_id = :t"), {"t": thread_id})
        session.add(ThreadORM(thread_id=thread_id, user_id=owner, tenant_id=tenant, status="idle"))
        await session.commit()


async def _drop_thread(maker: async_sessionmaker, thread_id: str) -> None:
    async with maker() as session:
        await session.execute(text("delete from runs where thread_id = :t"), {"t": thread_id})
        await session.execute(text("delete from thread where thread_id = :t"), {"t": thread_id})
        await session.commit()


async def _seed_assistant(maker: async_sessionmaker, *, assistant_id: str, owner: str, tenant: str) -> None:
    """Create one real assistant row, replacing any stale one.

    ``crons.assistant_id`` (and ``runs.assistant_id``) is a real foreign key, so
    the referenced row has to exist in this database before the child insert can
    be exercised at all — the FK is the point of the test, not a reason to skip.
    """
    async with maker() as session:
        await session.execute(text("delete from assistant where assistant_id = :a"), {"a": assistant_id})
        session.add(
            AssistantORM(
                assistant_id=assistant_id,
                name=assistant_id,
                graph_id="adm-seed-graph",
                config={"adm_seed": assistant_id},
                user_id=owner,
                tenant_id=tenant,
            )
        )
        await session.commit()


async def _drop_assistant(maker: async_sessionmaker, assistant_id: str) -> None:
    async with maker() as session:
        await session.execute(text("delete from assistant where assistant_id = :a"), {"a": assistant_id})
        await session.commit()


# ---------------------------------------------------------------------------
# Scope authorization against real rows
# ---------------------------------------------------------------------------
async def test_same_tenant_and_owner_is_allowed(maker: async_sessionmaker) -> None:
    thread_id, owner = "adm-ok-1", "adm-user-1"
    await _seed_thread(maker, thread_id=thread_id, owner=owner, tenant=DEFAULT_TENANT)
    principal = User(identity=owner, tenant_id=DEFAULT_TENANT)
    try:
        async with maker() as session:
            # The protected thread must NOT be rejected: this is the exact
            # regression a scalar() implementation caused.
            await authorize_thread_scope(session, thread_id, principal)
            await bind_thread_scope(session, thread_id, principal)
            await session.commit()
    finally:
        await _drop_thread(maker, thread_id)


async def test_missing_tenant_claim_is_refused(maker: async_sessionmaker) -> None:
    """A principal with no tenant claim is refused, never scoped to a default.

    The strict gate is the point of the feature: ``identity`` alone must not
    authorize a row, not even the principal's own thread.
    """
    thread_id, owner = "adm-notenant-1", "adm-user-1"
    await _seed_thread(maker, thread_id=thread_id, owner=owner, tenant=DEFAULT_TENANT)
    unscoped = User(identity=owner)
    try:
        async with maker() as session:
            with pytest.raises(HTTPException) as preflight:
                await authorize_thread_scope(session, thread_id, unscoped)
            assert preflight.value.status_code == 403
            with pytest.raises(HTTPException) as bind:
                await bind_thread_scope(session, thread_id, unscoped)
            assert bind.value.status_code == 403
            await session.rollback()
    finally:
        await _drop_thread(maker, thread_id)


async def test_cross_tenant_is_not_found(maker: async_sessionmaker) -> None:
    thread_id, owner = "adm-xtenant-1", "adm-user-1"
    await _seed_thread(maker, thread_id=thread_id, owner=owner, tenant="acme")
    foreign = User(identity=owner, tenant_id="other")
    try:
        async with maker() as session:
            with pytest.raises(HTTPException) as preflight:
                await authorize_thread_scope(session, thread_id, foreign)
            assert preflight.value.status_code == 404
            with pytest.raises(HTTPException) as bind:
                await bind_thread_scope(session, thread_id, foreign)
            assert bind.value.status_code == 404
            await session.rollback()
    finally:
        await _drop_thread(maker, thread_id)


async def test_cross_owner_is_not_found(maker: async_sessionmaker) -> None:
    thread_id = "adm-xowner-1"
    await _seed_thread(maker, thread_id=thread_id, owner="adm-user-other", tenant=DEFAULT_TENANT)
    caller = User(identity="adm-user-1", tenant_id=DEFAULT_TENANT)
    try:
        async with maker() as session:
            with pytest.raises(HTTPException) as preflight:
                await authorize_thread_scope(session, thread_id, caller)
            assert preflight.value.status_code == 404
            with pytest.raises(HTTPException) as bind:
                await bind_thread_scope(session, thread_id, caller)
            assert bind.value.status_code == 404
            await session.rollback()
    finally:
        await _drop_thread(maker, thread_id)


# ---------------------------------------------------------------------------
# Real unique-violation recovery across two connections
# ---------------------------------------------------------------------------


async def test_losing_insert_rolls_back_then_reads_winner(maker: async_sessionmaker) -> None:
    """Two connections, same (tenant, owner, thread, key).

    Proves the recovery order that a mocked session could not: the loser's
    insert waits on the winner's in-flight tuple and fails with the unique
    violation once that transaction commits; the loser's transaction is then
    aborted, so the winner must be read on a NEW transaction started by an
    explicit rollback. Reading first would raise PendingRollbackError.
    """
    import asyncio

    thread_id, owner, key = "adm-race-1", "adm-user-1", "op-race-1"
    await _seed_thread(maker, thread_id=thread_id, owner=owner, tenant=DEFAULT_TENANT)
    try:
        async with maker() as winner_session, maker() as loser_session:
            winner_session.add(
                RunORM(
                    run_id="adm-race-winner",
                    thread_id=thread_id,
                    user_id=owner,
                    tenant_id=DEFAULT_TENANT,
                    operation_key=key,
                    intent_digest="a" * 64,
                    status="pending",
                )
            )
            await winner_session.flush()  # holds the unique index entry

            loser_session.add(
                RunORM(
                    run_id="adm-race-loser",
                    thread_id=thread_id,
                    user_id=owner,
                    tenant_id=DEFAULT_TENANT,
                    operation_key=key,
                    intent_digest="a" * 64,
                    status="pending",
                )
            )

            async def _losing_insert() -> None:
                await loser_session.flush()

            loser_flush = asyncio.create_task(_losing_insert())
            # The loser is waiting on the winner's uncommitted tuple; committing
            # turns that wait into the unique violation it must recover from.
            await asyncio.sleep(0.2)
            await winner_session.commit()
            with pytest.raises(IntegrityError) as caught:
                await loser_flush
            assert _is_admission_conflict(caught.value), str(caught.value)

            # The loser's transaction is aborted: roll back before reading.
            await loser_session.rollback()

            winner = (
                await loser_session.execute(
                    text(
                        "select run_id, intent_digest from runs "
                        "where tenant_id = :t and user_id = :o and thread_id = :th and operation_key = :k"
                    ),
                    {"t": DEFAULT_TENANT, "o": owner, "th": thread_id, "k": key},
                )
            ).one_or_none()
            assert winner is not None and winner[0] == "adm-race-winner"
    finally:
        await _drop_thread(maker, thread_id)


async def test_other_integrity_violation_is_not_misclassified(maker: async_sessionmaker) -> None:
    """A foreign-key failure must not be swallowed as an admission conflict."""
    async with maker() as session:
        session.add(
            RunORM(
                run_id="adm-fk-1",
                thread_id="adm-thread-does-not-exist",
                user_id="adm-user-1",
                tenant_id=DEFAULT_TENANT,
                operation_key="op-fk-1",
                intent_digest="b" * 64,
                status="pending",
            )
        )
        with pytest.raises(IntegrityError) as caught:
            await session.flush()
        assert not _is_admission_conflict(caught.value), str(caught.value)
        await session.rollback()


async def test_operation_key_scoped_per_tenant(maker: async_sessionmaker) -> None:
    """Same owner + same key in a different tenant is a different logical run."""
    thread_a, thread_b = "adm-scope-a", "adm-scope-b"
    await _seed_thread(maker, thread_id=thread_a, owner="adm-user-1", tenant=DEFAULT_TENANT)
    await _seed_thread(maker, thread_id=thread_b, owner="adm-user-1", tenant="acme")
    try:
        async with maker() as session:
            session.add_all(
                [
                    RunORM(
                        run_id="adm-scope-run-a",
                        thread_id=thread_a,
                        user_id="adm-user-1",
                        tenant_id=DEFAULT_TENANT,
                        operation_key="op-shared",
                        intent_digest="c" * 64,
                        status="pending",
                    ),
                    RunORM(
                        run_id="adm-scope-run-b",
                        thread_id=thread_b,
                        user_id="adm-user-1",
                        tenant_id="acme",
                        operation_key="op-shared",
                        intent_digest="c" * 64,
                        status="pending",
                    ),
                ]
            )
            await session.commit()
    finally:
        await _drop_thread(maker, thread_a)
        await _drop_thread(maker, thread_b)


# ---------------------------------------------------------------------------
# FOR UPDATE bind: the lock must block a concurrent owner change
# ---------------------------------------------------------------------------


async def test_scope_lock_blocks_concurrent_owner_change(maker: async_sessionmaker) -> None:
    """bind_thread_scope holds a FOR UPDATE lock until its transaction ends.

    A plain SELECT would let connection B reassign the thread owner between the
    authorization read and the run insert (READ COMMITTED). The unlocked
    preflight deliberately takes no lock, so the lock has to come from the bind:
    with FOR UPDATE, B blocks until A ends — proven by the timeout — and the
    post-commit read shows the racing rewrite landing only after the release.

    Negative control: the same racing write against the unlocked preflight is
    NOT blocked, so the timeout above cannot pass for free.
    """
    import asyncio

    locked_id, unlocked_id, owner = "adm-lock-1", "adm-lock-2", "adm-user-1"
    await _seed_thread(maker, thread_id=locked_id, owner=owner, tenant=DEFAULT_TENANT)
    await _seed_thread(maker, thread_id=unlocked_id, owner=owner, tenant=DEFAULT_TENANT)
    principal = User(identity=owner, tenant_id=DEFAULT_TENANT)

    async def _reassign(session, thread_id: str) -> None:
        await session.execute(
            text("update thread set user_id = 'adm-other' where thread_id = :t"), {"t": thread_id}
        )
        await session.commit()

    try:
        async with maker() as holder, maker() as racer:
            await bind_thread_scope(holder, locked_id, principal)

            blocked = asyncio.create_task(_reassign(racer, locked_id))
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(blocked), timeout=0.5)
            await holder.commit()
            await asyncio.wait_for(blocked, timeout=5)

        async with maker() as holder, maker() as racer:
            await authorize_thread_scope(holder, unlocked_id, principal)
            # No row lock here, so the racing rewrite lands while A is open.
            await asyncio.wait_for(asyncio.create_task(_reassign(racer, unlocked_id)), timeout=5)
            await holder.rollback()

        async with maker() as check:
            scopes = dict(
                (
                    await check.execute(
                        text("select thread_id, user_id from thread where thread_id in (:a, :b)"),
                        {"a": locked_id, "b": unlocked_id},
                    )
                ).all()
            )
            assert scopes == {locked_id: "adm-other", unlocked_id: "adm-other"}
    finally:
        await _drop_thread(maker, locked_id)
        await _drop_thread(maker, unlocked_id)

async def test_concurrent_scope_bind_same_thread_does_not_cross_parties(maker: async_sessionmaker) -> None:
    """Two connections bind the same absent thread id under different owners.

    Creation is keyed on the primary key with ON CONFLICT DO NOTHING, so the
    loser never overwrites the winner's owner/tenant; exactly one scope wins.
    """
    import asyncio

    thread_id = "adm-bind-race-1"
    parties = [
        User(identity="adm-user-a", tenant_id="tenant-a"),
        User(identity="adm-user-b", tenant_id="tenant-b"),
    ]
    await _drop_thread(maker, thread_id)  # start from a known-absent id

    async def _bind(user: User) -> HTTPException | None:
        async with maker() as session:
            try:
                await bind_thread_scope(session, thread_id, user)
                await session.commit()
            except HTTPException as exc:
                await session.rollback()
                return exc
        return None

    try:
        outcomes = await asyncio.gather(*(_bind(user) for user in parties))
        losers = [exc for exc in outcomes if exc is not None]
        assert len(losers) == 1, outcomes
        assert losers[0].status_code == 404
        winner = parties[outcomes.index(None)]

        async with maker() as check:
            scope = (
                await check.execute(
                    text("select user_id, tenant_id from thread where thread_id = :t"), {"t": thread_id}
                )
            ).one()
        assert tuple(scope) == (winner.identity, winner.tenant_id)

        # A re-bind by the loser must not take the row over.
        async with maker() as session:
            with pytest.raises(HTTPException) as exc:
                await bind_thread_scope(session, thread_id, parties[outcomes.index(losers[0])])
            assert exc.value.status_code == 404
            await session.rollback()
    finally:
        await _drop_thread(maker, thread_id)


# ---------------------------------------------------------------------------
# /threads/prune tenant isolation (same owner, different tenant)
# ---------------------------------------------------------------------------


async def test_prune_delete_is_tenant_scoped(maker: async_sessionmaker) -> None:
    """Same owner in two tenants: a delete prune in tenant A must not touch B.

    The checkpoint side is stubbed (it is an external store); the SQL scope,
    the claim and the delete are real.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    from aegra_api.services import thread_ttl as ttl

    owner = "adm-prune-user"
    thread_a, thread_b = "adm-prune-a", "adm-prune-b"
    await _seed_thread(maker, thread_id=thread_a, owner=owner, tenant="tenant-a")
    await _seed_thread(maker, thread_id=thread_b, owner=owner, tenant="tenant-b")
    try:
        checkpointer = AsyncMock()
        async with maker() as session:
            with patch.object(ttl, "db_manager", MagicMock(get_checkpointer=MagicMock(return_value=checkpointer))):
                count = await ttl.prune_threads_by_id(
                    session,
                    thread_ids=[thread_a, thread_b],
                    strategy="delete",
                    user_id=owner,
                    tenant_id="tenant-a",
                )
        assert count == 1
        assert checkpointer.aprune.await_args.args[0] == [thread_a]

        async with maker() as check:
            remaining = {
                row[0]
                for row in (
                    await check.execute(
                        text("select thread_id from thread where thread_id in (:a, :b)"),
                        {"a": thread_a, "b": thread_b},
                    )
                ).all()
            }
        assert remaining == {thread_b}, "cross-tenant delete must not touch the other tenant"
    finally:
        await _drop_thread(maker, thread_a)
        await _drop_thread(maker, thread_b)


async def test_prune_keep_latest_is_tenant_scoped(maker: async_sessionmaker) -> None:
    """keep_latest must compact only the caller's tenant, not the sibling one."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from aegra_api.services import thread_ttl as ttl

    owner = "adm-prune-user"
    thread_a, thread_b = "adm-keep-a", "adm-keep-b"
    await _seed_thread(maker, thread_id=thread_a, owner=owner, tenant="tenant-a")
    await _seed_thread(maker, thread_id=thread_b, owner=owner, tenant="tenant-b")
    try:
        checkpointer = AsyncMock()
        async with maker() as session:
            with patch.object(ttl, "db_manager", MagicMock(get_checkpointer=MagicMock(return_value=checkpointer))):
                await ttl.prune_threads_by_id(
                    session,
                    thread_ids=[thread_a, thread_b],
                    strategy="keep_latest",
                    user_id=owner,
                    tenant_id="tenant-a",
                )
        # aprune receives a one-element list per owned thread: flatten the calls.
        pruned = [tid for call in checkpointer.aprune.await_args_list for tid in call.args[0]]
        assert pruned == [thread_a], f"keep_latest crossed tenants: {pruned}"
        assert {call.kwargs["strategy"] for call in checkpointer.aprune.await_args_list} == {"keep_latest"}
        # Compaction keeps both thread rows (only history is pruned).
        async with maker() as check:
            remaining = {
                row[0]
                for row in (
                    await check.execute(
                        text("select thread_id from thread where thread_id in (:a, :b)"),
                        {"a": thread_a, "b": thread_b},
                    )
                ).all()
            }
        assert remaining == {thread_a, thread_b}
    finally:
        await _drop_thread(maker, thread_a)
        await _drop_thread(maker, thread_b)


async def test_prune_expired_for_user_requires_tenant(maker: async_sessionmaker) -> None:
    """A caller-scoped sweep without a tenant is refused, not silently widened."""
    from aegra_api.services import thread_ttl as ttl

    async with maker() as session:
        with pytest.raises(ValueError):
            await ttl.prune_expired_threads_for_user(session, user_id="adm-prune-user", tenant_id=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Admission-version marker: new rows are strict, legacy rows stay relaxed,
# keyless new rows are NOT mistaken for legacy rows.
# ---------------------------------------------------------------------------


def _admitted_row_kwargs(**overrides):
    from aegra_api.services.run_preparation import ADMISSION_SCHEMA_VERSION

    kw = dict(
        run_id="adm-v-run",
        thread_id="adm-v-thread",
        user_id="adm-v-owner",
        tenant_id="tenant-v",
        operation_key="op-v",
        intent_digest="d" * 64,
        admission_version=ADMISSION_SCHEMA_VERSION,
        status="pending",
    )
    kw.update(overrides)
    return kw


def _payload(job_from_row: dict) -> dict:
    return job_from_row


async def _verify(maker, run_kwargs, params_mutator=None, principal_tenant=_ROW_TENANT, principal_identity=None):
    """Build one real row + payload, then run the worker verifier on it."""
    from aegra_api.models.run_job import RunJob
    from aegra_api.services.worker_executor import _verify_run_principal

    # Separate transaction: the fork's ORM declares no relationship between
    # Thread and Run, so one flush would not order the parent insert first.
    # Production binds the thread with an immediate INSERT (ON CONFLICT DO
    # NOTHING) before the admission write, i.e. before this run insert.
    await _seed_thread(
        maker,
        thread_id=run_kwargs["thread_id"],
        owner=run_kwargs["user_id"],
        tenant=run_kwargs["tenant_id"],
    )

    async with maker() as session:
        # Production projects the trusted tenant into the stored principal, so
        # the default is the row's own tenant; ``None`` means "no claim".
        tenant_claim = run_kwargs["tenant_id"] if principal_tenant is _ROW_TENANT else principal_tenant
        user_block = {"identity": principal_identity or run_kwargs["user_id"]}
        if tenant_claim is not None:
            user_block["tenant_id"] = tenant_claim
        params = {
            "graph_id": "agent",
            "user": user_block,
            "execution": {},
            "behavior": {},
            "run_metadata": {},
            "tenant_id": run_kwargs["tenant_id"],
            "operation_key": run_kwargs["operation_key"],
            "admission_version": run_kwargs["admission_version"],
        }
        if params_mutator:
            params_mutator(params)
        session.add(RunORM(**run_kwargs, execution_params=params))
        await session.commit()
        from sqlalchemy import select as _select

        run_orm = await session.scalar(_select(RunORM).where(RunORM.run_id == run_kwargs["run_id"]))
        job = RunJob.from_run_orm(run_orm)
        reason = await _verify_run_principal(session, run_orm, job)
        return reason


async def test_new_admitted_row_tampered_principal_tenant_is_refused(maker) -> None:
    async with maker() as s:
        await s.execute(text("delete from runs where run_id = 'adm-v-run'"))
        await s.execute(text("delete from thread where thread_id = 'adm-v-thread'"))
        await s.commit()
    try:
        reason = await _verify(
            maker, _admitted_row_kwargs(), principal_tenant="other-tenant"
        )
        assert reason is not None and "principal tenant" in reason, reason
    finally:
        async with maker() as s:
            await s.execute(text("delete from runs where run_id = 'adm-v-run'"))
            await s.execute(text("delete from thread where thread_id = 'adm-v-thread'"))
            await s.commit()


async def test_new_admitted_row_tampered_params_tenant_is_refused(maker) -> None:
    def _mutate(params):
        params["tenant_id"] = "evil-tenant"

    async with maker() as s:
        await s.execute(text("delete from runs where run_id = 'adm-v-run'"))
        await s.execute(text("delete from thread where thread_id = 'adm-v-thread'"))
        await s.commit()
    try:
        reason = await _verify(maker, _admitted_row_kwargs(), params_mutator=_mutate)
        assert reason is not None and "payload tenant" in reason, reason
    finally:
        async with maker() as s:
            await s.execute(text("delete from runs where run_id = 'adm-v-run'"))
            await s.execute(text("delete from thread where thread_id = 'adm-v-thread'"))
            await s.commit()


async def test_clean_admitted_row_passes(maker) -> None:
    async with maker() as s:
        await s.execute(text("delete from runs where run_id = 'adm-v-run'"))
        await s.execute(text("delete from thread where thread_id = 'adm-v-thread'"))
        await s.commit()
    try:
        reason = await _verify(maker, _admitted_row_kwargs())
        assert reason is None, reason
    finally:
        async with maker() as s:
            await s.execute(text("delete from runs where run_id = 'adm-v-run'"))
            await s.execute(text("delete from thread where thread_id = 'adm-v-thread'"))
            await s.commit()


async def test_keyless_new_row_is_strict_not_legacy(maker) -> None:
    """A keyless row written by this admission code must NOT fall into the
    legacy contract: tampering its payload tenant must still be refused."""
    from aegra_api.services.run_preparation import ADMISSION_SCHEMA_VERSION

    kw = _admitted_row_kwargs(operation_key=None, intent_digest="d" * 64)
    async with maker() as s:
        await s.execute(text("delete from runs where run_id = 'adm-v-run'"))
        await s.execute(text("delete from thread where thread_id = 'adm-v-thread'"))
        await s.commit()
    try:
        assert kw["admission_version"] == ADMISSION_SCHEMA_VERSION
        reason = await _verify(maker, kw, principal_tenant="other-tenant")
        assert reason is not None and "principal tenant" in reason, reason
    finally:
        async with maker() as s:
            await s.execute(text("delete from runs where run_id = 'adm-v-run'"))
            await s.execute(text("delete from thread where thread_id = 'adm-v-thread'"))
            await s.commit()


async def test_legacy_row_without_admission_version_stays_relaxed(maker) -> None:
    """Pre-migration row: no admission_version anywhere, payload may omit scope."""
    kw = _admitted_row_kwargs(operation_key=None, intent_digest=None, admission_version=None)
    async with maker() as s:
        await s.execute(text("delete from runs where run_id = 'adm-v-run'"))
        await s.execute(text("delete from thread where thread_id = 'adm-v-thread'"))
        await s.commit()
    try:
        reason = await _verify(
            maker, kw, principal_tenant=None
        )
        assert reason is None, reason
    finally:
        async with maker() as s:
            await s.execute(text("delete from runs where run_id = 'adm-v-run'"))
            await s.execute(text("delete from thread where thread_id = 'adm-v-thread'"))
            await s.commit()


async def test_keyless_api_contract_unchanged(maker) -> None:
    """No-key runs: two inserts, same tenant/owner/thread, no unique violation.

    The partial unique index covers only keyed rows — keyless rows never
    deduplicate (pre-existing at-least-once contract).
    """
    from aegra_api.services.run_preparation import ADMISSION_SCHEMA_VERSION

    await _seed_thread(maker, thread_id="adm-nok-thread", owner="adm-nok-owner", tenant="tenant-v")
    try:
        async with maker() as session:
            session.add_all(
                [
                    RunORM(
                        run_id="adm-nok-1",
                        thread_id="adm-nok-thread",
                        user_id="adm-nok-owner",
                        tenant_id="tenant-v",
                        operation_key=None,
                        intent_digest="e" * 64,
                        admission_version=ADMISSION_SCHEMA_VERSION,
                        status="pending",
                    ),
                    RunORM(
                        run_id="adm-nok-2",
                        thread_id="adm-nok-thread",
                        user_id="adm-nok-owner",
                        tenant_id="tenant-v",
                        operation_key=None,
                        intent_digest="e" * 64,
                        admission_version=ADMISSION_SCHEMA_VERSION,
                        status="pending",
                    ),
                ]
            )
            await session.commit()
    finally:
        async with maker() as s:
            await s.execute(text("delete from runs where thread_id = 'adm-nok-thread'"))
            await s.execute(text("delete from thread where thread_id = 'adm-nok-thread'"))
            await s.commit()


# ---------------------------------------------------------------------------
# Migration must not grant attribution to historical rows
# ---------------------------------------------------------------------------


async def test_legacy_rows_have_null_tenant_and_new_rows_do_not(maker: async_sessionmaker) -> None:
    """The migration adds NULL tenant columns (no backfill); new writes set it.

    A NULL tenant is *not* 'default': it means "unattributed", and the read
    predicate (tenant_scope) hides such rows unless the deployment is
    explicitly single-tenant.
    """
    async with maker() as session:
        cols = {
            row[0]: row[1]
            for row in (
                await session.execute(
                    text(
                        "select column_name, is_nullable from information_schema.columns "
                        "where table_name = :t and column_name = 'tenant_id'"
                    ),
                    {"t": "thread"},
                )
            ).all()
        }
        assert cols.get("tenant_id") == "YES", cols
        default = (
            await session.execute(
                text(
                    "select column_default from information_schema.columns "
                    "where table_name = :t and column_name = 'tenant_id'"
                ),
                {"t": "thread"},
            )
        ).scalar_one()
        assert default is None, f"tenant_id must not have a server default, got {default!r}"

    # A new thread row always carries the trusted tenant.
    await _seed_thread(maker, thread_id="adm-attr-1", owner="adm-attr-owner", tenant="tenant-a")
    try:
        async with maker() as session:
            value = (
                await session.execute(
                    text("select tenant_id from thread where thread_id = 'adm-attr-1'")
                )
            ).scalar_one()
        assert value == "tenant-a"
    finally:
        await _drop_thread(maker, "adm-attr-1")


async def test_cron_rows_require_tenant_attribution(maker: async_sessionmaker) -> None:
    """Cron tenant is stored per row; an unattributed row is not 'default'."""
    from aegra_api.core.orm import Cron as CronORM

    assistant_id, owner = "adm-cron-assistant", "adm-cron-owner"
    await _seed_assistant(maker, assistant_id=assistant_id, owner=owner, tenant="tenant-a")
    try:
        async with maker() as session:
            session.add(
                CronORM(
                    cron_id="adm-cron-1",
                    assistant_id=assistant_id,
                    user_id=owner,
                    tenant_id="tenant-a",
                    schedule="*/5 * * * *",
                )
            )
            # The FK is satisfied by the seeded assistant, so the row really lands.
            await session.commit()
            stored = (
                await session.execute(text("select tenant_id from crons where cron_id = 'adm-cron-1'"))
            ).scalar_one()
            assert stored == "tenant-a"
    finally:
        async with maker() as session:
            await session.execute(text("delete from crons where cron_id = 'adm-cron-1'"))
            await session.commit()
        await _drop_assistant(maker, assistant_id)

# ---------------------------------------------------------------------------
# Single-tenant config: another tenant cannot touch legacy NULL rows
# ---------------------------------------------------------------------------


async def test_configured_single_tenant_does_not_open_null_rows_to_other_tenant(
    maker: async_sessionmaker, monkeypatch
) -> None:
    """With AEGRA_SINGLE_TENANT_ID=tenant-a, a tenant-b principal is refused and
    cannot read/update/delete the unattributed (NULL tenant) legacy rows."""
    from fastapi import HTTPException

    from aegra_api.services.run_preparation import authorize_thread_scope, tenant_scope, trusted_identity

    monkeypatch.setenv("AEGRA_SINGLE_TENANT_ID", "tenant-a")

    async with maker() as session:
        # Unattributed legacy thread row (as the migration leaves it).
        session.add(ThreadORM(thread_id="adm-legacy-1", user_id="adm-legacy-owner", tenant_id=None))
        await session.commit()

    try:
        async with maker() as session:
            # tenant-b principal is refused outright
            with pytest.raises(HTTPException) as exc:
                trusted_identity(User(identity="adm-legacy-owner", tenant_id="tenant-b"))
            assert exc.value.status_code == 403

            # and the NULL branch is not reachable for tenant-b even directly
            stmt = tenant_scope(ThreadORM.tenant_id, "tenant-b")
            rows = (await session.execute(__import__("sqlalchemy").select(ThreadORM.thread_id).where(stmt))).all()
            assert rows == [], rows

            # tenant-a (the configured one) may see it
            stmt_a = tenant_scope(ThreadORM.tenant_id, "tenant-a")
            rows_a = (
                await session.execute(__import__("sqlalchemy").select(ThreadORM.thread_id).where(stmt_a))
            ).all()
            assert [r[0] for r in rows_a] == ["adm-legacy-1"]

            # write/delete attempts by tenant-b do not reach it
            await session.execute(
                text("delete from thread where thread_id = 'adm-legacy-1' and tenant_id = 'tenant-b'")
            )
            await session.commit()
        async with maker() as check:
            still_there = (
                await check.execute(text("select count(*) from thread where thread_id = 'adm-legacy-1'"))
            ).scalar_one()
            assert still_there == 1
    finally:
        async with maker() as s:
            await s.execute(text("delete from thread where thread_id = 'adm-legacy-1'"))
            await s.commit()


# ---------------------------------------------------------------------------
# Assistant uniqueness per (tenant, owner) with legacy NULL partition
# ---------------------------------------------------------------------------


async def _assistant_indexes(maker) -> dict[str, str]:
    async with maker() as session:
        rows = (
            await session.execute(
                text("select indexname, indexdef from pg_indexes where tablename = 'assistant'")
            )
        ).all()
    return {name: ddl for name, ddl in rows}

def _normalise_ddl(ddl: str) -> str:
    """Squash whitespace and the redundant parens Postgres adds around casts.

    ``md5(config::text)`` round-trips out of ``pg_indexes`` as
    ``md5((config)::text)``; both are the same index expression.
    """
    return re.sub(r"\((\w+)\)(::)", r"\1\2", ddl.replace(" ", ""))


async def test_assistant_unique_indexes_are_split_by_tenant(maker) -> None:
    indexes = await _assistant_indexes(maker)
    assert "idx_assistant_tenant_user_graph_config" in indexes
    assert "idx_assistant_user_graph_config_legacy" in indexes
    assert "idx_assistant_tenant_user_assistant" in indexes
    assert "idx_assistant_user_assistant_legacy" in indexes
    # The config expression is preserved verbatim in both partitions.
    for name in ("idx_assistant_tenant_user_graph_config", "idx_assistant_user_graph_config_legacy"):
        assert "md5(config::text)" in _normalise_ddl(indexes[name]), indexes[name]
        assert "tenant_id IS NOT NULL" in indexes[name] or "tenant_id IS NULL" in indexes[name], indexes[name]
    # The owner-wide originals are gone.
    assert "idx_assistant_user_graph_config" not in indexes
    assert "idx_assistant_user_assistant" not in indexes


async def test_same_owner_two_tenants_may_hold_same_graph_config(maker) -> None:
    """Cross-tenant same owner + same graph/config must be two distinct rows."""
    from aegra_api.core.orm import Assistant as AssistantORM

    try:
        async with maker() as session:
            session.add_all(
                [
                    AssistantORM(
                        assistant_id="adm-asst-a",
                        name="a",
                        graph_id="g",
                        config={"k": 1},
                        user_id="adm-asst-owner",
                        tenant_id="tenant-a",
                        version=1,
                    ),
                    AssistantORM(
                        assistant_id="adm-asst-b",
                        name="b",
                        graph_id="g",
                        config={"k": 1},
                        user_id="adm-asst-owner",
                        tenant_id="tenant-b",
                        version=1,
                    ),
                ]
            )
            await session.commit()
        async with maker() as check:
            count = (
                await check.execute(
                    text("select count(*) from assistant where assistant_id in ('adm-asst-a','adm-asst-b')")
                )
            ).scalar_one()
        assert count == 2
    finally:
        async with maker() as s:
            await s.execute(text("delete from assistant where assistant_id in ('adm-asst-a','adm-asst-b')"))
            await s.commit()


async def test_same_tenant_duplicate_graph_config_is_rejected(maker) -> None:
    """Within one tenant the original uniqueness still bites (real index, no mock)."""
    from aegra_api.core.orm import Assistant as AssistantORM

    try:
        async with maker() as session:
            session.add(
                AssistantORM(
                    assistant_id="adm-asst-dup-1",
                    name="a",
                    graph_id="g",
                    config={"k": 1},
                    user_id="adm-asst-owner",
                    tenant_id="tenant-a",
                    version=1,
                )
            )
            await session.commit()
            session.add(
                AssistantORM(
                    assistant_id="adm-asst-dup-2",
                    name="b",
                    graph_id="g",
                    config={"k": 1},
                    user_id="adm-asst-owner",
                    tenant_id="tenant-a",
                    version=1,
                )
            )
            with pytest.raises(IntegrityError):
                await session.commit()
            await session.rollback()
    finally:
        async with maker() as s:
            await s.execute(text("delete from assistant where assistant_id like 'adm-asst-dup-%'"))
            await s.commit()


async def test_legacy_null_tenant_assistants_stay_unique(maker) -> None:
    """Pre-attribution rows keep their old uniqueness through the NULL partition."""
    from aegra_api.core.orm import Assistant as AssistantORM

    try:
        async with maker() as session:
            session.add(
                AssistantORM(
                    assistant_id="adm-asst-legacy-1",
                    name="a",
                    graph_id="g",
                    config={"k": 2},
                    user_id="adm-asst-legacy-owner",
                    tenant_id=None,
                    version=1,
                )
            )
            await session.commit()
            session.add(
                AssistantORM(
                    assistant_id="adm-asst-legacy-2",
                    name="b",
                    graph_id="g",
                    config={"k": 2},
                    user_id="adm-asst-legacy-owner",
                    tenant_id=None,
                    version=1,
                )
            )
            with pytest.raises(IntegrityError):
                await session.commit()
            await session.rollback()
    finally:
        async with maker() as s:
            await s.execute(text("delete from assistant where assistant_id like 'adm-asst-legacy-%'"))
            await s.commit()
