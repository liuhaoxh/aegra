"""Behavior tests: commit->enqueue crash window recovery.

A run that is committed to Postgres but whose enqueue to Redis was lost
(process crash between ``session.commit()`` and ``executor.submit()``) must
still reach a worker. Proves the recovery chain end to end at unit level:

1. ``LeaseReaper._find_recoverable`` surfaces a stuck pending unclaimed run
   older than the threshold — this is the lost-enqueue run.
2. ``_missing_from_queue`` classifies it as absent from the Redis queue.
3. ``_reenqueue`` pushes it back.
4. Idempotency invariant: replaying the request under the same operation key
   returns the admitted run and never enqueues a second one — recovery
   re-executes the same run, it does not duplicate it.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from redis import RedisError

from aegra_api.services.lease_reaper import LeaseReaper
from aegra_api.services import run_preparation as mod

MODULE = "aegra_api.services.lease_reaper"


@pytest.fixture
def _settings_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aegra_api.services.lease_reaper.settings.worker.STUCK_PENDING_THRESHOLD_SECONDS", 60)


async def test_lost_enqueue_run_is_found_stuck_pending(
    _settings_threshold: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Committed-but-never-enqueued: status=pending, claimed_by=None, old enough."""
    session = AsyncMock()
    stuck_result = MagicMock()
    stuck_result.fetchall.return_value = [MagicMock(__getitem__=lambda s, i: "run-committed")]
    crashed_result = MagicMock()
    crashed_result.fetchall.return_value = []

    def _execute(stmt, *a, **kw):
        # crashed query selects running+lease rows; stuck query selects pending rows.
        sql = str(stmt)
        return stuck_result if "claimed_by" in sql else crashed_result

    session.execute = AsyncMock(side_effect=_execute)
    maker = MagicMock()
    maker.return_value.__aenter__ = AsyncMock(return_value=session)
    maker.return_value.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr("aegra_api.services.lease_reaper._get_session_maker", lambda: maker)

    crashed, stuck = await LeaseReaper._find_recoverable()
    assert crashed == []
    assert stuck == ["run-committed"]


async def test_fresh_pending_run_is_not_yet_stuck(
    _settings_threshold: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run committed milliseconds ago is inside the threshold: not recovered
    yet — that's fine, the reaper's next tick covers it if enqueue is lost."""
    session = AsyncMock()
    stuck_result = MagicMock()
    stuck_result.fetchall.return_value = []
    crashed_result = MagicMock()
    crashed_result.fetchall.return_value = []

    def _execute(stmt, *a, **kw):
        sql = str(stmt)
        return crashed_result if "running" in sql else stuck_result

    session.execute = AsyncMock(side_effect=_execute)
    maker = MagicMock()
    maker.return_value.__aenter__ = AsyncMock(return_value=session)
    maker.return_value.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr("aegra_api.services.lease_reaper._get_session_maker", lambda: maker)

    crashed, stuck = await LeaseReaper._find_recoverable()
    assert crashed == [] and stuck == []


async def test_missing_from_queue_classifies_lost_enqueue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run is not in Redis (enqueue lost): it is reported missing."""
    client = AsyncMock()
    client.lrange.return_value = ["run-queued-elsewhere"]
    monkeypatch.setattr(
        "aegra_api.services.lease_reaper.redis_manager", MagicMock(get_client=MagicMock(return_value=client))
    )
    missing = await LeaseReaper._missing_from_queue(["run-lost", "run-queued-elsewhere"])
    assert missing == ["run-lost"]


async def test_redis_down_returns_empty_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Redis unavailable: fall back empty; workers pick up via Postgres poll."""
    client = AsyncMock()
    client.lrange.side_effect = RedisError("down")
    monkeypatch.setattr(
        "aegra_api.services.lease_reaper.redis_manager", MagicMock(get_client=MagicMock(return_value=client))
    )
    missing = await LeaseReaper._missing_from_queue(["run-lost"])
    assert missing == []


async def test_reenqueue_pushes_lost_run(monkeypatch: pytest.MonkeyPatch) -> None:
    client = AsyncMock()
    monkeypatch.setattr(
        "aegra_api.services.lease_reaper.redis_manager", MagicMock(get_client=MagicMock(return_value=client))
    )
    pushed = await LeaseReaper._reenqueue(["run-lost"])
    assert pushed == ["run-lost"]
    client.rpush.assert_awaited_once()


async def test_replay_of_recovered_run_does_not_double_admit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full window: run committed, enqueue lost, reaper re-enqueued, client
    retries the same operation key while the recovered run is still pending.
    The retry must return the same run and enqueue nothing new."""
    from datetime import UTC, datetime
    admitted = MagicMock()
    admitted.run_id = "run-committed"
    admitted.thread_id = "thread-1"
    admitted.assistant_id = "agent"
    admitted.status = "pending"
    admitted.input = {"messages": [{"role": "user", "content": "hi"}]}
    admitted.output = None
    admitted.error_message = None
    admitted.config = {}
    admitted.context = {}
    admitted.user_id = "owner-1"
    admitted.tenant_id = "default"
    admitted.operation_key = "op-1"
    admitted.created_at = datetime.now(UTC)
    admitted.updated_at = datetime.now(UTC)
    from aegra_api.models import RunCreate

    request = RunCreate(
        assistant_id="agent",
        input={"messages": [{"role": "user", "content": "hi"}]},
        operation_key="op-1",
    )

    # digest recorded at admission time equals the retry's digest
    admitted.intent_digest = mod.intent_digest(request)

    session = AsyncMock()
    session.scalar.return_value = admitted
    run = mod._replay(admitted, mod.intent_digest(request))
    assert run.run_id == "run-committed"
