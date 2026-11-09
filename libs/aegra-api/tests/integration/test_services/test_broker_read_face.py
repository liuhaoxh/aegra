"""Real-Redis tests for the broker read face (V11P-03 03-01).

These run against the isolation Redis (``AEGRA_TEST_REDIS_URL``); nothing is
mocked, so gap detection is proven on the same keys and commands production
uses. The five classes under test: complete history, resume tail, retention
gap, unreadable history, terminal tail.
"""

import asyncio
import os
import socket
import uuid

import pytest
import redis.asyncio as aioredis

from aegra_api.services import redis_broker as redis_broker_module
from aegra_api.services.base_broker import EventStreamState
from aegra_api.services.redis_broker import RedisRunBroker
from aegra_api.utils import generate_event_id

REDIS_URL = os.environ.get("AEGRA_TEST_REDIS_URL", "")

pytestmark = pytest.mark.skipif(
    not REDIS_URL,
    reason="AEGRA_TEST_REDIS_URL is required for real-Redis broker tests",
)


class _ClientStub:
    """The module-level ``redis_manager`` seam, backed by a real client."""

    def __init__(self, client: aioredis.Redis) -> None:
        self._client = client

    def get_client(self) -> aioredis.Redis:
        return self._client


@pytest.fixture
async def client(monkeypatch):
    real = aioredis.from_url(REDIS_URL, socket_connect_timeout=2, decode_responses=True)
    monkeypatch.setattr(redis_broker_module, "redis_manager", _ClientStub(real))
    yield real
    await real.aclose()


@pytest.fixture
def run_id() -> str:
    return f"readface-{uuid.uuid4().hex[:8]}"


def _broker(run_id: str) -> RedisRunBroker:
    return RedisRunBroker(
        run_id,
        f"aegra:run:events:{run_id}",
        f"aegra:run:cache:{run_id}",
        f"aegra:run:counter:{run_id}",
    )


async def _publish(broker: RedisRunBroker, client: aioredis.Redis, run_id: str, payload) -> str:
    """Allocate a sequence exactly like production, then store the event."""
    seq = await client.incr(broker._counter_key)
    event_id = generate_event_id(run_id, int(seq))
    await broker.put(event_id, payload)
    return event_id


async def _seed(broker: RedisRunBroker, client: aioredis.Redis, run_id: str, count: int) -> list[str]:
    return [
        await _publish(broker, client, run_id, ("values", {"index": index}))
        for index in range(count)
    ]


def _unused_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def test_complete_history_replays_from_the_start_and_from_a_cursor(client, run_id):
    broker = _broker(run_id)
    event_ids = await _seed(broker, client, run_id, 3)

    full = await broker.read_events(0)
    assert full.state is EventStreamState.OPEN
    assert [entry.event_id for entry in full.events] == event_ids
    assert [entry.seq for entry in full.events] == [1, 2, 3]
    assert full.events[0].payload == ("values", {"index": 0}), "payload 原样回放"

    tail = await broker.read_events(2)
    assert tail.state is EventStreamState.OPEN
    assert [entry.seq for entry in tail.events] == [3]
    assert tail.after_seq == 2

    nothing_new = await broker.read_events(3)
    assert nothing_new.state is EventStreamState.OPEN
    assert nothing_new.events == ()


async def test_retention_gap_is_reported_with_the_lost_range(client, run_id):
    broker = _broker(run_id)
    await _seed(broker, client, run_id, 5)
    # Simulate LTRIM dropping the first three events while the counter keeps
    # the watermark: the retained history no longer reaches a cursor at 0.
    await client.ltrim(broker._cache_key, -2, -1)

    read = await broker.read_events(0)
    assert read.state is EventStreamState.GAP
    assert (read.lost_from_seq, read.lost_to_seq) == (1, 3)
    assert [entry.seq for entry in read.events] == [4, 5], "已保留的事件仍要给出"

    resumed = await broker.read_events(3)
    assert resumed.state is EventStreamState.OPEN
    assert [entry.seq for entry in resumed.events] == [4, 5], "游标在保留区之后 ⇒ 不是缺口"


async def test_unreadable_history_is_unavailable_never_empty(client, run_id, monkeypatch):
    unreachable = aioredis.from_url(
        f"redis://127.0.0.1:{_unused_port()}/0",
        socket_connect_timeout=0.3,
        socket_timeout=0.3,
    )
    monkeypatch.setattr(redis_broker_module, "redis_manager", _ClientStub(unreachable))
    try:
        read = await _broker(run_id).read_events(0)
    finally:
        await unreachable.aclose()

    assert read.state is EventStreamState.UNAVAILABLE, "读不到 ≠ 没有历史"
    assert read.events == ()
    assert read.detail, "不可用必须带可诊断原因"


async def test_terminal_tail_reports_end_of_stream(client, run_id):
    broker = _broker(run_id)
    await _publish(broker, client, run_id, ("values", {"index": 0}))
    await _publish(broker, client, run_id, ("end", {"status": "success"}))

    from_start = await broker.read_events(0)
    assert from_start.state is EventStreamState.END_OF_STREAM
    assert [entry.seq for entry in from_start.events] == [1, 2]

    at_end = await broker.read_events(2)
    assert at_end.state is EventStreamState.END_OF_STREAM, "游标停在终态事件上仍要知道流已结束"
    assert at_end.events == ()


async def test_terminal_claim_lands_at_most_once_across_instances(client, run_id):
    first, second = _broker(run_id), _broker(run_id)
    await _publish(first, client, run_id, ("values", {"index": 0}))

    results = await asyncio.gather(
        first.put_terminal(("end", {"status": "success"})),
        second.put_terminal(("end", {"status": "success"})),
    )
    assert sorted(result is not None for result in results) == [False, True], (
        "两个实例只能有一个落终态"
    )

    stored = await first.stored_events()
    terminals = [entry for entry in stored if entry.payload[0] == "end"]
    assert len(terminals) == 1
    assert terminals[0].seq == 2, "失败的一方不烧序号，事件 id 连续"
    assert [entry.seq for entry in stored] == [1, 2]


async def test_watermark_is_not_double_incremented_by_a_cache_write(client, run_id):
    broker = _broker(run_id)
    await _seed(broker, client, run_id, 2)

    assert await broker.high_water_seq() == 2, (
        "put() 不得再自己 INCR：否则计数器是事件数的两倍、事件 id 出现空洞"
    )
