"""Streaming service replay face (V11P-03 03-01).

The typed read result must reach the wire unchanged: normal frames keep their
exact bytes, and an incomplete history adds exactly one explicit
``stream_state`` frame before any replayed event.
"""

import json

import pytest

from aegra_api.services import streaming_service as streaming_service_module
from aegra_api.services.base_broker import BrokerEvent, EventRead, EventStreamState
from aegra_api.services.streaming_service import StreamingService


class _FakeBroker:
    def __init__(self, read: EventRead) -> None:
        self._read = read
        self.cursors: list[int] = []

    async def read_events(self, after_seq: int | None = None) -> EventRead:
        self.cursors.append(0 if after_seq is None else after_seq)
        return self._read


def _entry(seq: int, payload) -> BrokerEvent:
    return BrokerEvent(
        event_id=f"run-1_event_{seq}",
        seq=seq,
        dedupe_key=f"dedupe-{seq}",
        payload=payload,
    )


@pytest.fixture
def service(monkeypatch):
    instance = StreamingService()

    def _install(read: EventRead) -> _FakeBroker:
        broker = _FakeBroker(read)
        monkeypatch.setattr(
            streaming_service_module.broker_manager,
            "get_or_create_broker",
            lambda run_id: broker,
        )
        return broker

    instance.install = _install  # type: ignore[attr-defined]
    return instance


async def _collect(service: StreamingService, last_event_id: str | None) -> list[tuple[str, str]]:
    return [frame async for frame in service._replay_stored_events("run-1", last_event_id)]


def _state_frames(frames: list[tuple[str, str]]) -> list[str]:
    return [frame for _, frame in frames if "event: stream_state" in frame]


def _data(frame: str) -> dict:
    line = next(row for row in frame.split("\n") if row.startswith("data: "))
    return json.loads(line[len("data: ") :])


async def test_complete_history_replays_without_a_state_frame(service):
    broker = service.install(
        EventRead(
            events=(_entry(1, ("values", {"index": 1})), _entry(2, ("values", {"index": 2}))),
            state=EventStreamState.OPEN,
            after_seq=0,
        )
    )

    frames = await _collect(service, None)

    assert _state_frames(frames) == []
    assert [event_id for event_id, _ in frames] == ["run-1_event_1", "run-1_event_2"]
    assert frames[0][1] == (
        'event: values\ndata: {"index":1}\nid: run-1_event_1\n\n'
    ), "正常帧字节不变（event/data/id 布局与既有实现一致）"
    assert _data(frames[0][1]) == {"index": 1}
    assert broker.cursors == [0]


async def test_resume_cursor_is_extracted_from_the_last_event_id(service):
    service.install(
        EventRead(
            events=(_entry(4, ("values", {"index": 4})),),
            state=EventStreamState.OPEN,
            after_seq=3,
        )
    )

    frames = await _collect(service, "run-1_event_3")

    assert [event_id for event_id, _ in frames] == ["run-1_event_4"]


async def test_gap_yields_one_explicit_state_frame_before_the_tail(service):
    service.install(
        EventRead(
            events=(_entry(6, ("values", {"index": 6})),),
            state=EventStreamState.GAP,
            after_seq=2,
            lost_from_seq=3,
            lost_to_seq=5,
            detail="retained history does not reach the reader cursor",
        )
    )

    frames = await _collect(service, "run-1_event_2")

    (state_frame,) = _state_frames(frames)
    assert _data(state_frame) == {
        "state": "gap",
        "after_seq": 2,
        "lost_from_seq": 3,
        "lost_to_seq": 5,
        "reason": "retained history does not reach the reader cursor",
    }
    assert frames[0][0] == "", "状态帧不冒领事件 id（不推进 live 去重游标）"
    assert frames[-1][0] == "run-1_event_6", "缺口之后仍回放已保留的事件"


async def test_unavailable_yields_only_the_explicit_state_frame(service):
    service.install(
        EventRead(
            events=(),
            state=EventStreamState.UNAVAILABLE,
            after_seq=0,
            detail="replay buffer unreadable: connection refused",
        )
    )

    frames = await _collect(service, None)

    (state_frame,) = _state_frames(frames)
    payload = _data(state_frame)
    assert payload["state"] == "unavailable"
    assert "connection refused" in payload["reason"], "原因必须到达客户端（可诊断）"
    assert len(frames) == 1, "读不到历史时不能伪装成「没有事件」的完整空流"
