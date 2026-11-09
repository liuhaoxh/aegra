"""Unit tests for run admission idempotency (operation key + intent digest).

Covers:
- same (tenant, owner, thread, operation key) + same intent -> replay of the
  admitted run, executor not called again;
- different operation keys -> distinct runs, no false dedup;
- cross-tenant / cross-owner replay attempts -> 404 thread scope rejection;
- lost-race fallback resolves concurrent duplicates to one run;
- commit->enqueue crash window recovers via replay.
"""

import hashlib
import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError

from aegra_api.models import RunCreate, User
from aegra_api.models.run_job import RunBehavior, RunExecution, RunIdentity, RunJob
from aegra_api.services import run_preparation as mod
from aegra_api.models.run_job import UnapprovedPrincipalClaim, project_principal
from aegra_api.services.run_preparation import (
    _find_admitted_run,
    _prepare_run,
    _replay,
    authorize_thread_scope,
    bind_thread_scope,
    intent_digest,
    trusted_identity,
)

MODULE = "aegra_api.services.run_preparation"


def _user(identity: str = "owner-1", tenant_id: str | None = "default", **extra: Any) -> User:
    kwargs: dict[str, Any] = dict(extra)
    if tenant_id is not None:
        kwargs["tenant_id"] = tenant_id
    return User(identity=identity, **kwargs)


def _request(**overrides: Any) -> RunCreate:
    base: dict[str, Any] = {
        "assistant_id": "agent",
        "input": {"messages": [{"role": "user", "content": "hi"}]},
        "operation_key": "op-1",
    }
    base.update(overrides)
    return RunCreate.model_validate(base)

def _admitted_run(
    *,
    run_id: str = "run-1",
    digest: str | None = None,
    tenant_id: str = "default",
    user_id: str = "owner-1",
    thread_id: str = "thread-1",
    operation_key: str = "op-1",
) -> MagicMock:
    run = MagicMock()
    run.run_id = run_id
    run.intent_digest = digest if digest is not None else intent_digest(_request())
    run.tenant_id = tenant_id
    run.user_id = user_id
    run.thread_id = thread_id
    run.operation_key = operation_key
    run.assistant_id = "agent"
    run.status = "pending"
    run.input = {"messages": [{"role": "user", "content": "hi"}]}
    run.output = None
    run.error_message = None
    run.config = {}
    run.context = {}
    run.created_at = datetime.now(UTC)
    run.updated_at = datetime.now(UTC)
    return run


def _job_for(run_id: str = "run-new") -> RunJob:
    return RunJob(
        identity=RunIdentity(run_id=run_id, thread_id="thread-1", graph_id="agent"),
        user=_user(),
        execution=RunExecution(),
        behavior=RunBehavior(),
    )


class _Result:
    """Stands in for a SQLAlchemy Result: ``one_or_none()`` yields one Row."""

    def __init__(self, row: object) -> None:
        self._row = row

    def one_or_none(self) -> object:
        return self._row


def _scope_session(row: object) -> AsyncMock:
    """Session double for authorize_thread_scope's single unlocked read."""
    session = AsyncMock()
    session.execute = AsyncMock(return_value=_Result(row))
    return session


@pytest.fixture
def _session() -> AsyncMock:
    """Session double exposing the interfaces the code actually uses.

    ``authorize_thread_scope`` must go through ``execute(...).one_or_none()``;
    if it regressed to ``scalar()`` this double raises, so the test fails.
    """
    session = AsyncMock()
    # authorize_thread_scope: create-if-absent, then FOR UPDATE read of the bound row
    session.execute.return_value = _Result(("owner-1", "default"))
    session.scalar.return_value = None  # _find_admitted_run / assistant lookup default
    nested = AsyncMock()
    nested.__aenter__ = AsyncMock(return_value=None)
    nested.__aexit__ = AsyncMock(return_value=False)
    session.begin_nested = MagicMock(return_value=nested)
    return session


# ---------------------------------------------------------------------------
# trusted_identity: tenant never comes from the request payload
# ---------------------------------------------------------------------------


def test_trusted_identity_uses_claim() -> None:
    assert trusted_identity(_user("u1", "acme")) == ("acme", "u1")


def test_trusted_identity_rejects_missing_tenant_by_default() -> None:
    """Multi-tenant default: an unscoped principal is refused, never shared."""
    with pytest.raises(HTTPException) as exc:
        trusted_identity(_user("u1", None))
    assert exc.value.status_code == 403


def test_trusted_identity_rejects_blank_tenant_by_default() -> None:
    with pytest.raises(HTTPException) as exc:
        trusted_identity(_user("u1", "   "))
    assert exc.value.status_code == 403


def test_trusted_identity_allows_explicit_single_tenant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AEGRA_SINGLE_TENANT_ID", "install-1")
    assert trusted_identity(_user("u1", None)) == ("install-1", "u1")


def test_trusted_identity_rejects_anonymous_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AEGRA_SINGLE_TENANT_ID", "install-1")  # even then
    with pytest.raises(HTTPException) as exc:
        trusted_identity(_user("anonymous", "acme"))
    assert exc.value.status_code == 401


def test_trusted_identity_rejects_unauthenticated() -> None:
    with pytest.raises(HTTPException) as exc:
        trusted_identity(_user("u1", "acme", is_authenticated=False))
    assert exc.value.status_code == 401


def test_trusted_identity_rejects_non_string_claim() -> None:
    with pytest.raises(HTTPException) as exc:
        trusted_identity(_user("u1", 123))  # non-string tenant claim
    assert exc.value.status_code in (403, 500)


# ---------------------------------------------------------------------------
# intent digest
# ---------------------------------------------------------------------------


def test_intent_digest_same_intent_same_hash() -> None:
    assert intent_digest(_request()) == intent_digest(_request())


def test_intent_digest_differing_input_differs() -> None:
    assert intent_digest(_request()) != intent_digest(_request(input={"messages": []}))


def test_intent_digest_covers_checkpoint_alias() -> None:
    a = intent_digest(_request(checkpoint={"checkpoint_id": "ea77baf6-8d18-4a31-9d54-9c1c27ecb7d5"}))
    b = intent_digest(_request(checkpoint_id="ea77baf6-8d18-4a31-9d54-9c1c27ecb7d5"))
    assert a == b


# ---------------------------------------------------------------------------
# replay semantics
# ---------------------------------------------------------------------------


async def test_replay_same_digest_returns_run(_session: AsyncMock) -> None:
    admitted = _admitted_run()
    run = _replay(admitted, intent_digest(_request()))
    assert run is not None


async def test_replay_different_digest_conflicts() -> None:
    admitted = _admitted_run(digest="0" * 64)
    with pytest.raises(HTTPException) as exc:
        _replay(admitted, intent_digest(_request()))
    assert exc.value.status_code == 409


# ---------------------------------------------------------------------------
# thread scope: cross-tenant / cross-owner undetectable
# ---------------------------------------------------------------------------


async def test_authorize_thread_scope_rejects_foreign_owner() -> None:
    session = _scope_session(("other-owner", "default"))
    with pytest.raises(HTTPException) as exc:
        await authorize_thread_scope(session, "thread-1", _user("owner-1"))
    assert exc.value.status_code == 404


async def test_authorize_thread_scope_rejects_foreign_tenant() -> None:
    session = _scope_session(("owner-1", "acme"))
    with pytest.raises(HTTPException) as exc:
        await authorize_thread_scope(session, "thread-1", _user("owner-1", "default"))
    assert exc.value.status_code == 404


async def test_authorize_thread_scope_allows_owner() -> None:
    session = _scope_session(("owner-1", "default"))
    await authorize_thread_scope(session, "thread-1", _user("owner-1", "default"))


async def test_authorize_thread_scope_passes_absent_thread_to_the_binding_step() -> None:
    """Unlocked preflight: an absent thread is not an error, binding is later."""
    session = _scope_session(None)
    await authorize_thread_scope(session, "thread-1", _user("owner-1"))
    assert session.execute.await_count == 1


async def test_bind_thread_scope_refuses_a_row_that_lost_the_insert_race() -> None:
    """Create-if-absent, then FOR UPDATE read of the bound row."""
    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[_Result(None), _Result(("other-owner", "default"))])
    with pytest.raises(HTTPException) as exc:
        await bind_thread_scope(session, "thread-1", _user("owner-1"))
    # our own concurrent insert lost the PK race to another owner -> 404, never shared
    assert exc.value.status_code == 404
    assert session.execute.await_count == 2


async def test_bind_thread_scope_allows_own_bound_row() -> None:
    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[_Result(None), _Result(("owner-1", "default"))])
    await bind_thread_scope(session, "thread-1", _user("owner-1"))
    assert session.execute.await_count == 2


# ---------------------------------------------------------------------------
# _prepare_run idempotency paths
# ---------------------------------------------------------------------------


def _patch_common(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence graph/assistant resolution and thread mutations."""
    monkeypatch.setattr(mod, "_validate_resume_command", AsyncMock())
    monkeypatch.setattr(mod, "update_thread_metadata", AsyncMock())
    monkeypatch.setattr(mod, "set_thread_status", AsyncMock())
    monkeypatch.setattr(mod, "resolve_assistant_id", lambda requested, graphs: requested)


def _assistant() -> MagicMock:
    a = MagicMock()
    a.assistant_id = "agent"
    a.graph_id = "agent"
    a.config = {}
    a.context = {}
    return a


async def test_same_key_same_intent_replays_without_enqueue(
    _session: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_common(monkeypatch)
    admitted = _admitted_run()
    monkeypatch.setattr(mod, "_find_admitted_run", AsyncMock(return_value=admitted))
    submit = AsyncMock()

    with (
        patch(f"{MODULE}.get_langgraph_service") as lg,
        patch(f"{MODULE}.executor") as exec_mod,
    ):
        lg.return_value.list_graphs.return_value = ["agent"]
        exec_mod.submit = submit

        run_id, run, job = await _prepare_run(
            _session, "thread-1", _request(), _user(), initial_status="pending"
        )

    assert run_id == "run-1"
    assert job is None
    submit.assert_not_awaited()
    mod.update_thread_metadata.assert_not_awaited()
    assert run is not None


async def test_different_key_creates_second_run(
    _session: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_common(monkeypatch)
    monkeypatch.setattr(mod, "_find_admitted_run", AsyncMock(return_value=None))
    _session.scalar = AsyncMock(return_value=_assistant())
    submit = AsyncMock()

    with (
        patch(f"{MODULE}.get_langgraph_service") as lg,
        patch(f"{MODULE}.executor") as exec_mod,
    ):
        lg.return_value.list_graphs.return_value = ["agent"]
        exec_mod.submit = submit

        run_id, _run, job = await _prepare_run(
            _session, "thread-1", _request(operation_key="op-2"), _user(), initial_status="pending"
        )

    assert run_id != "run-1"
    assert job is not None
    submit.assert_awaited_once()


async def test_lost_race_resolves_to_admitted_run(
    _session: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent duplicate: the unique index rejects the loser's insert."""
    _patch_common(monkeypatch)
    admitted = _admitted_run(run_id="winner")
    calls = {"n": 0}

    async def find(session, tenant, owner, thread, key):
        calls["n"] += 1
        return admitted if calls["n"] >= 2 else None

    monkeypatch.setattr(mod, "_find_admitted_run", find)
    _session.scalar = AsyncMock(return_value=_assistant())

    def _flush() -> None:
        raise IntegrityError("uq_runs_tenant_owner_thread_opkey", {}, Exception("dup"))

    _session.flush.side_effect = _flush
    submit = AsyncMock()

    with (
        patch(f"{MODULE}.get_langgraph_service") as lg,
        patch(f"{MODULE}.executor") as exec_mod,
    ):
        lg.return_value.list_graphs.return_value = ["agent"]
        exec_mod.submit = submit

        run_id, _run, job = await _prepare_run(
            _session, "thread-1", _request(), _user(), initial_status="pending"
        )

    assert run_id == "winner"
    assert job is None
    submit.assert_not_awaited()


async def test_unrelated_integrity_error_propagates(
    _session: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail-closed: only the admission index resolves to a replay."""
    _patch_common(monkeypatch)
    monkeypatch.setattr(mod, "_find_admitted_run", AsyncMock(return_value=None))
    _session.scalar = AsyncMock(return_value=_assistant())
    _session.flush.side_effect = IntegrityError("fk_broken", {}, Exception("fk"))

    with (
        patch(f"{MODULE}.get_langgraph_service") as lg,
        patch(f"{MODULE}.executor") as exec_mod,
    ):
        lg.return_value.list_graphs.return_value = ["agent"]
        exec_mod.submit = AsyncMock()
        with pytest.raises(IntegrityError):
            await _prepare_run(_session, "thread-1", _request(), _user(), initial_status="pending")


# ---------------------------------------------------------------------------
# _find_admitted_run query shape
# ---------------------------------------------------------------------------


async def test_find_admitted_run_scopes_all_four_columns() -> None:
    session = AsyncMock()
    session.scalar.return_value = None
    await _find_admitted_run(session, "t", "o", "th", "k")
    stmt = session.scalar.await_args.args[0]
    where = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    for col in ("tenant_id", "user_id", "thread_id", "operation_key"):
        assert col in where, f"missing scope column {col}"


# ---------------------------------------------------------------------------
# commit->enqueue window: DB run survives, replay resolves to it
# ---------------------------------------------------------------------------


async def test_crash_before_enqueue_then_replay_recovers(
    _session: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run committed, worker crash before (or losing) the enqueue: the retry
    under the same key replays the committed run instead of duplicating."""
    _patch_common(monkeypatch)
    # First attempt: admitted then process dies before submit.
    first = MagicMock()
    first.run_id = "committed"
    first.intent_digest = intent_digest(_request())
    monkeypatch.setattr(mod, "_find_admitted_run", AsyncMock(return_value=None))
    # scalar call order: authorize_thread_scope (None), then assistant lookup
    _session.scalar = AsyncMock(return_value=_assistant())
    submit = AsyncMock(side_effect=RuntimeError("worker crash before enqueue"))

    with (
        patch(f"{MODULE}.get_langgraph_service") as lg,
        patch(f"{MODULE}.executor") as exec_mod,
    ):
        lg.return_value.list_graphs.return_value = ["agent"]
        exec_mod.submit = submit
        with pytest.raises(RuntimeError):
            await _prepare_run(_session, "thread-1", _request(), _user(), initial_status="pending")

    # Retry after restart: run row exists (committed), so replay returns it.
    committed = _admitted_run(run_id="committed")
    monkeypatch.setattr(mod, "_find_admitted_run", AsyncMock(return_value=committed))
    retry_session = AsyncMock()
    retry_session.execute.return_value = _Result(None)  # thread scope: not foreign
    submit2 = AsyncMock()

    with patch(f"{MODULE}.executor") as exec_mod:
        exec_mod.submit = submit2
        run_id, _run, job = await _prepare_run(
            retry_session, "thread-1", _request(), _user(), initial_status="pending"
        )

    assert run_id == "committed"
    assert job is None
    submit2.assert_not_awaited()


# ---------------------------------------------------------------------------
# digest canonicalization guarantees
# ---------------------------------------------------------------------------


def test_digest_is_stable_sha256_of_canonical_json() -> None:
    """Frozen canonical form: every execution-affecting field is covered."""
    request = _request()
    expected = hashlib.sha256(
        json.dumps(
            {
                "assistant_id": "agent",
                "input": request.input,
                "command": None,
                "config": {},
                "context": {},
                "checkpoint": None,
                "durability": None,
                "multitask_strategy": None,
                "interrupt_before": None,
                "interrupt_after": None,
                "stream": False,
                "stream_mode": None,
                "stream_subgraphs": False,
                "on_disconnect": None,
                "on_completion": None,
                "metadata": None,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()
    assert expected == "7248fe7147ec8c306e3eab1255d5fe26a9f527623da5845068f7e35ca08acd3a"
    assert intent_digest(request) == expected

# ---------------------------------------------------------------------------
# RunJob serialization: admission scope survives, credentials never persist
# ---------------------------------------------------------------------------


def test_run_job_persists_projected_principal_and_scope() -> None:
    """execution_params carries tenant + operation key and the projected
    principal (no credentials, no unapproved extras)."""
    user = project_principal(
        User(identity="owner-1", display_name="Owner", tenant_id="acme", access_token="tok-1")
    )
    job = RunJob(
        identity=RunIdentity(run_id="run-1", thread_id="thread-1", graph_id="agent"),
        user=user,
        tenant_id="acme",
        operation_key="op-1",
    )
    params = job.to_execution_params()
    assert params["tenant_id"] == "acme"
    assert params["operation_key"] == "op-1"
    assert params["user"]["identity"] == "owner-1"
    assert params["user"]["tenant_id"] == "acme"
    assert "access_token" not in params["user"]
    assert "tok-1" not in json.dumps(params)


def test_normal_and_recovery_paths_share_one_principal() -> None:
    """Request-side execution and restart see the SAME effective User."""
    raw = User(
        identity="owner-1",
        display_name="Owner",
        tenant_id="acme",
        permissions=["runs:create"],
        scopes=["read"],
        access_token="tok-1",
        refresh_token="rt-1",
    )
    principal = project_principal(raw)
    job = RunJob(
        identity=RunIdentity(run_id="run-1", thread_id="thread-1", graph_id="agent"),
        user=principal,
        tenant_id="acme",
        operation_key="op-1",
    )

    run = MagicMock()
    run.run_id = "run-1"
    run.thread_id = "thread-1"
    run.tenant_id = "acme"
    run.operation_key = "op-1"
    run.execution_params = job.to_execution_params()

    recovered = RunJob.from_run_orm(run)
    assert recovered.user.model_dump() == job.user.model_dump() == principal.model_dump()
    assert recovered.user.permissions == ["runs:create"]
    # Restart-time authorization inputs are intact and credential-free.
    assert recovered.tenant_id == "acme"
    assert "access_token" not in json.dumps(recovered.to_execution_params())


def test_unknown_authorization_claim_is_refused_not_dropped() -> None:
    """A non-credential extra could affect authorization downstream: refuse."""
    with pytest.raises(UnapprovedPrincipalClaim) as exc:
        project_principal(User(identity="u1", tenant_id="acme", subscription_tier="pro"))
    assert "subscription_tier" in str(exc.value)


async def test_unknown_claim_refusal_reaches_the_caller_as_422(
    _session: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_common(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        await _prepare_run(
            _session,
            "thread-1",
            _request(),
            User(identity="u1", tenant_id="acme", subscription_tier="pro"),
            initial_status="pending",
        )
    assert exc.value.status_code == 422


# ---------------------------------------------------------------------------
# Principal projection: non-None claims + deployment-approved claims
# ---------------------------------------------------------------------------


def test_projection_carries_only_non_none_claims() -> None:
    """Declared-but-unset User fields are not claims.

    Regression: ``model_dump()`` emitted the ``email``/``org_id`` defaults as
    ``None``, the closed whitelist refused them, and every real run was 422
    before admission.
    """
    principal = project_principal(User(identity="owner-1"))
    dumped = principal.model_dump(exclude_none=True)

    assert dumped["identity"] == "owner-1"
    assert "email" not in dumped
    assert "org_id" not in dumped


def test_deployment_allowed_claim_is_carried_into_the_run() -> None:
    """A claim the deployment names as allowed survives the projection."""
    principal = project_principal(
        User(identity="owner-1", conversation_id="conv-1"),
        allowed_claims=("conversation_id",),
    )

    assert principal.model_dump()["conversation_id"] == "conv-1"


def test_allowed_claims_cannot_smuggle_credentials() -> None:
    """Listing a credential-shaped name as allowed does not carry it."""
    principal = project_principal(
        User(identity="owner-1", access_token="tok-1"),
        allowed_claims=("access_token",),
    )

    assert "access_token" not in principal.model_dump(exclude_none=True)


def test_unlisted_extra_claim_is_refused_with_allowed_claims_configured() -> None:
    """The allowlist is additive; anything unlisted stays fail-closed."""
    with pytest.raises(UnapprovedPrincipalClaim) as exc:
        project_principal(
            User(identity="owner-1", conversation_id="conv-1", subscription_tier="pro"),
            allowed_claims=("conversation_id",),
        )

    assert "subscription_tier" in str(exc.value)


async def test_configured_claim_reaches_admission_and_the_run_row(
    _session: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AEGRA_ALLOWED_PRINCIPAL_CLAIMS unblocks the auth backend's real claim."""
    _patch_common(monkeypatch)
    monkeypatch.setenv("AEGRA_ALLOWED_PRINCIPAL_CLAIMS", "conversation_id")
    monkeypatch.setattr(mod, "_find_admitted_run", AsyncMock(return_value=None))
    _session.scalar = AsyncMock(return_value=_assistant())
    submit = AsyncMock()

    with (
        patch(f"{MODULE}.get_langgraph_service") as lg,
        patch(f"{MODULE}.executor") as exec_mod,
    ):
        lg.return_value.list_graphs.return_value = ["agent"]
        exec_mod.submit = submit
        _run_id, _run, job = await _prepare_run(
            _session,
            "thread-1",
            _request(),
            _user(conversation_id="conv-1"),
            initial_status="pending",
        )

    assert job is not None
    assert job.user.model_dump()["conversation_id"] == "conv-1"
    assert "conv-1" in json.dumps(job.to_execution_params())


async def test_unconfigured_claim_still_422s_at_admission(
    _session: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the deployment allowlist the same claim is refused."""
    _patch_common(monkeypatch)
    monkeypatch.delenv("AEGRA_ALLOWED_PRINCIPAL_CLAIMS", raising=False)

    with pytest.raises(HTTPException) as exc:
        await _prepare_run(
            _session,
            "thread-1",
            _request(),
            _user(conversation_id="conv-1"),
            initial_status="pending",
        )

    assert exc.value.status_code == 422
    assert "conversation_id" in str(exc.value.detail)


async def test_worker_queue_payload_is_only_the_run_id() -> None:
    """Redis carries run_ids; the principal lives only in the run row."""
    from aegra_api.services import worker_executor as we

    job = RunJob(
        identity=RunIdentity(run_id="run-1", thread_id="thread-1", graph_id="agent"),
        user=project_principal(User(identity="owner-1", tenant_id="acme", access_token="tok-1")),
        tenant_id="acme",
    )
    pushed: list[object] = []

    async def _rpush(key: str, value: object) -> None:
        pushed.append((key, value))

    client = MagicMock()
    client.rpush = _rpush

    with patch.object(we.redis_manager, "get_client", return_value=client):
        await we.WorkerExecutor.submit(MagicMock(), job)

    assert len(pushed) == 1
    assert pushed[0][1] == "run-1", pushed
    assert "acme" not in str(pushed) and "tok-1" not in str(pushed)


def test_run_job_from_run_orm_restores_scope_from_the_payload() -> None:
    """Worker recovery reads scope from the persisted payload, never the row:
    reading it back off the row would make the worker's row-vs-payload
    comparison a tautology and hide a tampered payload."""
    run = MagicMock()
    run.run_id = "run-1"
    run.thread_id = "thread-1"
    run.tenant_id = "tampered-row-tenant"
    run.operation_key = "row-op"
    run.admission_version = 9
    run.execution_params = {
        "graph_id": "agent",
        "user": {"identity": "owner-1", "tenant_id": "acme"},
        "execution": {},
        "behavior": {},
        "run_metadata": {},
        "tenant_id": "acme",
        "operation_key": "op-1",
        "admission_version": 1,
    }
    job = RunJob.from_run_orm(run)
    assert job.tenant_id == "acme"
    assert job.operation_key == "op-1"
    assert job.admission_version == 1

# ---------------------------------------------------------------------------
# Digest covers every execution-affecting field; aliases stay equivalent
# ---------------------------------------------------------------------------


def test_digest_changes_with_interrupt_points() -> None:
    assert intent_digest(_request(interrupt_before=["approve"])) != intent_digest(_request())
    assert intent_digest(_request(interrupt_after=["done"])) != intent_digest(_request())


def test_digest_changes_with_durability() -> None:
    assert intent_digest(_request(durability="sync")) != intent_digest(_request(durability="async"))


def test_digest_changes_with_streaming_and_lifecycle() -> None:
    base = intent_digest(_request())
    for overrides in (
        {"stream_mode": ["values"]},
        {"stream": True},
        {"stream_subgraphs": True},
        {"on_disconnect": "continue"},
        {"on_completion": "delete"},
        {"multitask_strategy": "enqueue"},
        {"metadata": {"k": "v"}},
    ):
        assert intent_digest(_request(**overrides)) != base, overrides


def test_digest_durability_alias_is_equivalent() -> None:
    """durability='async' and the deprecated checkpoint_during=True are the
    same intent, resolved the same way the executor resolves them."""
    assert intent_digest(_request(durability="async")) == intent_digest(_request(checkpoint_during=True))
    assert intent_digest(_request(durability="exit")) == intent_digest(_request(checkpoint_during=False))


# ---------------------------------------------------------------------------
# Server-resolved identity is stamped onto the persisted principal
# ---------------------------------------------------------------------------


async def test_single_tenant_config_scope_is_stamped_on_principal(
    _session: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tenant resolved from trusted config (not from the client) must reach
    the persisted principal, or the worker's row-vs-payload check would refuse
    every run after restart."""
    _patch_common(monkeypatch)
    monkeypatch.setenv("AEGRA_SINGLE_TENANT_ID", "install-1")
    # The thread row is attributed to the same trusted tenant.
    _session.execute = AsyncMock(return_value=_Result(("owner-1", "install-1")))
    monkeypatch.setattr(mod, "_find_admitted_run", AsyncMock(return_value=None))
    _session.scalar = AsyncMock(return_value=_assistant())
    submit = AsyncMock()

    with (
        patch(f"{MODULE}.get_langgraph_service") as lg,
        patch(f"{MODULE}.executor") as exec_mod,
    ):
        lg.return_value.list_graphs.return_value = ["agent"]
        exec_mod.submit = submit
        _run_id, _run, job = await _prepare_run(
            _session, "thread-1", _request(operation_key=None), _user("owner-1", None), initial_status="pending"
        )

    assert job is not None
    assert job.tenant_id == "install-1"
    assert job.user.tenant_id == "install-1"
    assert job.user.identity == "owner-1"
    assert job.admission_version is not None


async def test_keyless_run_still_persists_a_digest(
    _session: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keyless rows are NOT legacy rows: they carry digest + admission version."""
    _patch_common(monkeypatch)
    _session.scalar = AsyncMock(return_value=_assistant())
    submit = AsyncMock()

    with (
        patch(f"{MODULE}.get_langgraph_service") as lg,
        patch(f"{MODULE}.executor") as exec_mod,
    ):
        lg.return_value.list_graphs.return_value = ["agent"]
        exec_mod.submit = submit
        await _prepare_run(
            _session, "thread-1", _request(operation_key=None), _user("owner-1"), initial_status="pending"
        )

    run_orm = _session.add.call_args.args[0]
    assert run_orm.operation_key is None
    assert run_orm.intent_digest is not None
    assert run_orm.admission_version is not None

# ---------------------------------------------------------------------------
# Tenant attribution: no server default, explicit legacy branch only
# ---------------------------------------------------------------------------


def test_tenant_scope_is_strict_by_default() -> None:
    from aegra_api.core.orm import Thread as ThreadORM
    from aegra_api.services.run_preparation import tenant_scope

    stmt = tenant_scope(ThreadORM.tenant_id, "tenant-a")
    compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "tenant-a" in compiled and "IS NULL" not in compiled


def test_tenant_scope_allows_legacy_rows_only_in_single_tenant_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    from aegra_api.core.orm import Thread as ThreadORM
    from aegra_api.services.run_preparation import tenant_scope

    monkeypatch.setenv("AEGRA_SINGLE_TENANT_ID", "install-1")
    compiled = str(tenant_scope(ThreadORM.tenant_id, "install-1").compile(compile_kwargs={"literal_binds": True}))
    assert "IS NULL" in compiled, compiled


async def test_thread_insert_carries_trusted_tenant(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /threads must write the trusted tenant, not rely on a DB default."""
    from aegra_api.api import threads as threads_api

    captured: list[Any] = []

    class _Scalars:
        def first(self):
            return None

    session = AsyncMock()
    session.scalars.return_value = _Scalars()
    session.scalar = AsyncMock(return_value=None)

    async def _scalars(stmt):
        captured.append(stmt)
        return _Scalars()

    session.scalars = _scalars
    session.add = MagicMock()

    # created is None (nothing inserted), so the endpoint ends in the 409
    # conflict branch — the INSERT values are what this test pins.
    with pytest.raises(HTTPException) as exc:
        await threads_api.create_thread(
            threads_api.ThreadCreate(), User(identity="owner-1", tenant_id="tenant-a"), session
        )
    assert exc.value.status_code == 409
    insert = next(stmt for stmt in captured if "INSERT INTO" in str(stmt))
    values = insert.compile().params
    assert "tenant-a" in values.values(), values


async def test_delete_thread_by_id_requires_tenant() -> None:
    """Owner-only cleanup is no longer callable: the tenant argument is required."""
    from aegra_api.services import run_cleanup

    with pytest.raises(TypeError):
        await run_cleanup.delete_thread_by_id("thread-1", "owner-1")  # type: ignore[call-arg]


def test_worker_rejects_unattributed_legacy_row_in_multi_tenant_mode() -> None:
    """A NULL-tenant run row cannot be executed when no single-tenant mode is set."""
    import asyncio

    from aegra_api.services.worker_executor import _verify_run_principal

    run = MagicMock()
    run.run_id = "r"
    run.thread_id = "t"
    run.user_id = "o"
    run.tenant_id = None
    run.operation_key = None
    run.intent_digest = None
    run.admission_version = None

    job = RunJob(identity=RunIdentity(run_id="r", thread_id="t", graph_id="g"), user=User(identity="o"))

    class _ThreadRow:
        def one_or_none(self):
            return ("o", None)

    session = AsyncMock()
    session.execute = AsyncMock(return_value=_ThreadRow())
    reason = asyncio.run(_verify_run_principal(session, run, job))
    assert reason is not None and "no tenant attribution" in reason

# ---------------------------------------------------------------------------
# Single-tenant config must not become a cross-tenant bypass
# ---------------------------------------------------------------------------


def test_claim_must_equal_configured_single_tenant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AEGRA_SINGLE_TENANT_ID", "tenant-a")
    with pytest.raises(HTTPException) as exc:
        trusted_identity(_user("owner-1", "tenant-b"))
    assert exc.value.status_code == 403


def test_tenant_scope_null_branch_requires_matching_configured_tenant(monkeypatch: pytest.MonkeyPatch) -> None:
    """A direct call with a different tenant must not gain the IS NULL branch."""
    from aegra_api.core.orm import Thread as ThreadORM
    from aegra_api.services.run_preparation import tenant_scope

    monkeypatch.setenv("AEGRA_SINGLE_TENANT_ID", "tenant-a")
    other = str(tenant_scope(ThreadORM.tenant_id, "tenant-b").compile(compile_kwargs={"literal_binds": True}))
    assert "IS NULL" not in other, other
    same = str(tenant_scope(ThreadORM.tenant_id, "tenant-a").compile(compile_kwargs={"literal_binds": True}))
    assert "IS NULL" in same


def test_assistant_acl_scopes_owner_by_tenant() -> None:
    """Assistant owner reads are (owner AND tenant); system assistants stay shared."""
    import inspect

    from aegra_api.services import assistant_service

    source = inspect.getsource(assistant_service.AssistantService)
    assert "tenant_scope(AssistantORM.tenant_id" in source
    assert 'AssistantORM.user_id == "system"' in source
    # No bare owner-only assistant filter may remain.
    assert "AssistantORM.user_id == self.user.identity\n" not in source.replace(
        "AssistantORM.user_id == self.user.identity,\n", ""
    )


def test_store_namespace_includes_tenant() -> None:
    from aegra_api.api.store import apply_namespace_scoping

    ns = apply_namespace_scoping(["x"], User(identity="owner-1", tenant_id="tenant-a"))
    ns_b = apply_namespace_scoping(["x"], User(identity="owner-1", tenant_id="tenant-b"))
    assert ns != ns_b
    assert "tenant-a" in " ".join(ns) and "tenant-b" in " ".join(ns_b)

def test_system_identity_is_reserved(monkeypatch: pytest.MonkeyPatch) -> None:
    """A request principal must not act as the shared system owner."""
    monkeypatch.setenv("AEGRA_SINGLE_TENANT_ID", "install-1")
    with pytest.raises(HTTPException) as exc:
        trusted_identity(_user("system", "install-1"))
    assert exc.value.status_code == 403
    # There is no environment bypass: system rows are written through the ORM
    # by server-side registration code, never by a request principal.
    with pytest.raises(HTTPException):
        trusted_identity(_user("system", "install-1"))

def test_wildcard_identity_and_tenant_claims_are_refused() -> None:
    with pytest.raises(HTTPException) as id_exc:
        trusted_identity(_user("own*r", "tenant-a"))
    assert id_exc.value.status_code == 403
    with pytest.raises(HTTPException) as tenant_exc:
        trusted_identity(_user("owner-1", "tenant-*"))
    assert tenant_exc.value.status_code == 403
