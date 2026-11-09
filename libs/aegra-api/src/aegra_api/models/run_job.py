"""Job specification for graph run execution.

Groups the 15+ parameters of execute_run into three logical sub-objects
so every function in the execution chain takes at most three arguments.

Uses Pydantic models (consistent with the rest of the codebase) with
frozen=True for immutability. Serialization uses model_dump/model_validate
— no manual to_dict/from_dict needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

from aegra_api.models.auth import User
from aegra_api.models.runs import Durability

logger = structlog.getLogger(__name__)

if TYPE_CHECKING:
    from aegra_api.core.orm import Run as RunORM


#: Claims of the authenticated principal that are carried into a run. Each is a
#: field the real consumers in this fork actually read:
#:
#: * ``identity``          -> run ownership, graph access context, thread owner
#: * ``tenant_id``         -> admission scope, worker cross-check
#: * ``is_authenticated``  -> fail-closed identity checks
#: * ``display_name``      -> traces / graph access context
#: * ``permissions``       -> ``with_auth_ctx`` and ``require_permission``
#: * ``scopes``            -> auth middleware surface read by the same handlers
#:
#: This is a closed list. There is deliberately NO operator-extensible key list:
#: an env var would let ``access_token``/``password`` back into the persisted
#: principal, which is exactly what must not happen.
PERSISTED_USER_FIELDS: tuple[str, ...] = (
    "identity",
    "tenant_id",
    "is_authenticated",
    "display_name",
    "permissions",
    "scopes",
)

#: Field-name markers that identify a credential. Such extras are dropped (never
#: persisted, never logged with values). Any OTHER extra field is not silently
#: discarded — the caller must either drop it at the auth boundary or the
#: admission is refused, because an unknown claim could change authorization
#: somewhere downstream of us.
CREDENTIAL_FIELD_MARKERS: tuple[str, ...] = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "private_key",
    "cookie",
)


class UnapprovedPrincipalClaim(ValueError):
    """Raised when the authenticated principal carries a non-credential extra.

    Fails closed at admission: we cannot prove an unknown claim is not part of
    authorization downstream, and dropping it would silently change behavior
    between the request-side identity and the worker-side identity.
    """

    def __init__(self, claims: list[str]) -> None:
        self.claims = claims
        super().__init__(
            "Authenticated principal carries unapproved claims that would not survive "
            f"into the run: {', '.join(sorted(claims))}. Drop them at the auth boundary "
            "or add a reviewed field to PERSISTED_USER_FIELDS."
        )


def _is_credential_field(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in CREDENTIAL_FIELD_MARKERS)


def project_principal(user: User) -> User:
    """The single authenticated-principal projection used by every run path.

    Both the request-side execution (this object is what the executor gets) and
    the restart path (``from_run_orm`` re-validates the persisted copy) use this
    exact projection, so a recovered run executes as the same principal as the
    original one.

    * credential-looking extras are dropped: they must never reach the DB,
      the worker, or a log;
    * any other extra claim is refused with :class:`UnapprovedPrincipalClaim`
      (nothing is silently changed);
    * business ``input``/``context`` payloads are NOT inspected or filtered —
      no content-security claim is made for them.
    """
    data = user.model_dump()
    extras = [key for key in data if key not in PERSISTED_USER_FIELDS]
    unknown = [key for key in extras if not _is_credential_field(key)]
    if unknown:
        raise UnapprovedPrincipalClaim(unknown)
    if extras:
        logger.info("Dropped credential-bearing claims from run principal", fields=sorted(extras))
    return User.model_validate({key: data[key] for key in PERSISTED_USER_FIELDS if key in data})


class RunIdentity(BaseModel):
    """Locates a run within the system."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    thread_id: str
    graph_id: str


class RunExecution(BaseModel):
    """What to execute and how to execute it."""

    model_config = ConfigDict(frozen=True)

    input_data: dict[str, Any] | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    stream_mode: str | list[str] | None = None
    checkpoint: dict[str, Any] | None = None
    command: dict[str, Any] | None = None
    # Resolved at creation (run field, then server default); None keeps LangGraph's default.
    durability: Durability | None = None
    # When true, stream via the native v3 protocol producer for Agent Protocol v2.
    event_streaming_v2: bool = False


class RunBehavior(BaseModel):
    """Behavioral modifiers for the run."""

    model_config = ConfigDict(frozen=True)

    interrupt_before: str | list[str] | None = None
    interrupt_after: str | list[str] | None = None
    multitask_strategy: str | None = None
    subgraphs: bool = False


class RunJob(BaseModel):
    """Complete specification for executing a graph run.

    Serialization for DB storage and Redis transport uses Pydantic's
    built-in model_dump/model_validate — no manual methods needed.
    """

    model_config = ConfigDict(frozen=True)

    identity: RunIdentity
    user: User
    execution: RunExecution = RunExecution()
    behavior: RunBehavior = RunBehavior()
    # Admission scope, carried so DB storage and Redis round-trips cannot lose
    # the tenant or the client-stable operation key the run was admitted under.
    # ``None`` means "the persisted payload does not declare it" — the worker
    # treats that as fail-closed for admitted rows and as a legacy row only when
    # the row itself proves it predates admission.
    tenant_id: str | None = None
    operation_key: str | None = None
    #: Persisted admission schema version. Present and >= 1 on every row this
    #: admission code writes; absent on rows written before the migration. This
    #: is the verifiable marker that separates "legacy row" from "new row"
    #: without guessing from missing keys.
    admission_version: int | None = None
    # JSONB-persisted; deliberately ``dict[str, Any]`` so legacy/drifted rows stay readable via
    # ``from_run_orm``. OTEL boundary in ``merge_run_metadata`` filters at emit. Do not narrow.
    run_metadata: dict[str, Any] = Field(default_factory=dict)

    def to_execution_params(self) -> dict[str, Any]:
        """Serialize for the ``execution_params`` JSONB column.

        Stores everything a worker needs to reconstruct the RunJob from
        the database. Identity fields (run_id, thread_id) are already
        columns on the runs table, but graph_id is not — so we include it.
        ``RunJob.user`` is already the projected principal (see
        :func:`project_principal`), so the stored block and the in-process
        execution see the same identity. Credentials never reach this dict.
        Business ``input``/``context`` are stored unchanged and carry no
        content-security guarantee.
        """
        return {
            "graph_id": self.identity.graph_id,
            "user": self.user.model_dump(),
            "execution": self.execution.model_dump(),
            "behavior": self.behavior.model_dump(),
            "run_metadata": self.run_metadata,
            "tenant_id": self.tenant_id,
            "operation_key": self.operation_key,
            "admission_version": self.admission_version,
        }

    @classmethod
    def from_run_orm(cls, run_orm: RunORM) -> RunJob:
        """Reconstruct a RunJob from a Run ORM row with execution_params."""
        params = run_orm.execution_params
        if params is None:
            raise ValueError(f"Run {run_orm.run_id} has no execution_params")
        return cls(
            identity=RunIdentity(
                run_id=run_orm.run_id,
                thread_id=run_orm.thread_id,
                graph_id=params["graph_id"],
            ),
            user=User.model_validate(params["user"]),
            execution=RunExecution.model_validate(params["execution"]),
            behavior=RunBehavior.model_validate(params["behavior"]),
            run_metadata=params.get("run_metadata") or {},
            # Admission scope comes ONLY from the persisted payload. Reading it
            # back off the row would make the worker's row-vs-job comparison a
            # tautology and hide a tampered payload.
            tenant_id=params.get("tenant_id"),
            operation_key=params.get("operation_key"),
            admission_version=params.get("admission_version"),
        )
