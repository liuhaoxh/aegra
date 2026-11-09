"""Pure tests for the event completeness classifier (V11P-03 03-01).

Every state is decided from two facts only: the retained entries and the
durable watermark. No Redis here — the Redis read face is covered separately
against a real server.
"""

import pytest

from aegra_api.services.base_broker import (
    BrokerEvent,
    EventStreamState,
    classify_events,
    dedupe_key_for,
    is_terminal_payload,
)


def _event(seq: int, payload=None, *, dedupe: str | None = None) -> BrokerEvent:
    body = payload if payload is not None else ("values", {"seq": seq})
    return BrokerEvent(
        event_id=f"run-1_event_{seq}",
        seq=seq,
        dedupe_key=dedupe or dedupe_key_for("run-1", body),
        payload=body,
    )


def test_empty_run_from_the_start_is_open():
    read = classify_events((), after_seq=0, high_water_seq=0)

    assert read.state is EventStreamState.OPEN
    assert read.events == ()


def test_complete_history_from_the_start_is_open():
    read = classify_events((_event(1), _event(2)), after_seq=0, high_water_seq=2)

    assert read.state is EventStreamState.OPEN
    assert [entry.seq for entry in read.events] == [1, 2]


def test_page_after_the_cursor_keeps_only_the_tail():
    read = classify_events((_event(1), _event(2), _event(3)), after_seq=2, high_water_seq=3)

    assert read.state is EventStreamState.OPEN
    assert [entry.seq for entry in read.events] == [3]


def test_cursor_past_a_retained_tail_event_is_end_of_stream():
    read = classify_events((_event(1), _event(2, ("end", {"status": "success"}))), after_seq=2, high_water_seq=2)

    assert read.state is EventStreamState.END_OF_STREAM
    assert read.events == ()


def test_fresh_terminal_tail_is_end_of_stream():
    read = classify_events((_event(1), _event(2, ("end", {"status": "success"}))), after_seq=0, high_water_seq=2)

    assert read.state is EventStreamState.END_OF_STREAM
    assert [entry.seq for entry in read.events] == [1, 2]


def test_hole_inside_the_retained_history_is_a_gap():
    read = classify_events((_event(1), _event(3)), after_seq=0, high_water_seq=3)

    assert read.state is EventStreamState.GAP
    assert (read.lost_from_seq, read.lost_to_seq) == (2, 2)
    assert read.detail == "sequence hole in stored history"
    assert [entry.seq for entry in read.events] == [1, 3], "已保留的事件仍要给出"


def test_cursor_before_the_retained_window_is_a_gap():
    read = classify_events((_event(4), _event(5)), after_seq=0, high_water_seq=5)

    assert read.state is EventStreamState.GAP
    assert (read.lost_from_seq, read.lost_to_seq) == (1, 3)
    assert read.detail == "reader cursor predates retained history"


def test_wiped_buffer_is_a_gap_not_an_empty_complete_history():
    read = classify_events((), after_seq=0, high_water_seq=7)

    assert read.state is EventStreamState.GAP
    assert (read.lost_from_seq, read.lost_to_seq) == (1, 7)
    assert read.detail == "retained history does not reach the reader cursor"


def test_cursor_ahead_of_the_watermark_is_a_gap_without_a_range():
    read = classify_events((), after_seq=9, high_water_seq=7)

    assert read.state is EventStreamState.GAP
    assert read.lost_from_seq is None and read.lost_to_seq is None
    assert read.detail == "reader cursor is ahead of the source watermark"


def test_redelivered_event_is_collapsed_but_a_conflict_is_a_gap():
    delivery = _event(2)
    collapsed = classify_events((_event(1), delivery, delivery), after_seq=0, high_water_seq=2)
    assert collapsed.state is EventStreamState.OPEN
    assert [entry.seq for entry in collapsed.events] == [1, 2]

    conflicting = classify_events(
        (_event(1), _event(2), _event(2, dedupe="different-content")),
        after_seq=0,
        high_water_seq=2,
    )
    assert conflicting.state is EventStreamState.GAP
    assert conflicting.detail == "stored sequence is not strictly increasing"


@pytest.mark.parametrize(
    "payload,terminal",
    [(("end", {"status": "success"}), True), (("values", {}), False), ("plain", False)],
)
def test_terminal_payload_detection(payload, terminal):
    assert is_terminal_payload(payload) is terminal
