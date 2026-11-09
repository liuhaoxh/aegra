"""Abstract base classes for the broker system"""

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, NamedTuple

from aegra_api.models.enums import RunCancellationAction
from aegra_api.utils import extract_event_sequence, generate_event_id

# How long a finished run's events stay replayable, both backends. Reconnects
# happen in seconds; this is the documented window in docs/guides/streaming.mdx.
REPLAY_RETENTION_SECONDS = 600


class EventStreamState(StrEnum):
    """What a read of a run's stored events implies about completeness.

    Consumers must take this from the broker instead of inferring it from
    timing, list length or a missing ``Last-Event-ID``: only the source knows
    whether history is complete, truncated or unreadable.
    """

    END_OF_STREAM = "end_of_stream"
    """Contiguous history whose tail is the run's terminal ``end`` event."""

    OPEN = "open"
    """Contiguous history so far; the run may still publish more events."""

    GAP = "gap"
    """History is truncated or its sequence is not contiguous: events are lost."""

    UNAVAILABLE = "unavailable"
    """The event source could not be read, so its contents are unknown."""


class BrokerUnavailable(RuntimeError):
    """The event source is unreachable, so nothing can be claimed about its contents."""


class BrokerEvent(NamedTuple):
    """A stored event together with its stable, replayable identity."""

    event_id: str
    """``{run_id}_event_{seq}``: stable across re-reads and re-deliveries."""

    seq: int
    """Run-scoped monotonic sequence number allocated once per event."""

    dedupe_key: str
    """Content discriminator: identical payloads share it across sequence numbers."""

    payload: Any
    """The event payload as published (``("values", {...})``, ``("end", ...)``...)."""


@dataclass(frozen=True, slots=True)
class EventRead:
    """Typed result of reading a run's stored events after a cursor."""

    events: tuple[BrokerEvent, ...]
    """Retained events with ``seq > after_seq``, in stored order."""

    state: EventStreamState
    """Completeness of what was read; never a guess."""

    after_seq: int = 0
    """The cursor the read was issued for (``0`` = from the start)."""

    lost_from_seq: int | None = None
    """First sequence number known to be missing, when ``state`` is ``GAP``."""

    lost_to_seq: int | None = None
    """Last sequence number known to be missing, when that bound is known."""

    detail: str | None = None
    """Human-readable reason for a non-normal state."""


def dedupe_key_for(run_id: str, payload: Any) -> str:
    """Return the stable idempotency discriminator for ``payload`` within ``run_id``.

    Two publishes of the same logical event (a producer that re-allocated an id
    after a restart, two instances signalling the same terminal state) get the
    same key, so a consumer dedupes on content instead of time heuristics.
    """
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(f"{run_id}\x1f{body}".encode()).hexdigest()[:32]


def is_terminal_payload(payload: Any) -> bool:
    """True when ``payload`` is the run's terminal ``("end", {...})`` event."""
    return isinstance(payload, tuple) and bool(payload) and payload[0] == "end"


def classify_events(
    entries: Sequence[BrokerEvent],
    *,
    after_seq: int,
    high_water_seq: int,
) -> EventRead:
    """Decide the completeness state of ``entries`` read after ``after_seq``.

    ``high_water_seq`` is the highest sequence the source durably stored; it is
    the only evidence used when nothing is retained, so a wiped buffer is
    reported as a gap instead of an empty stream.
    """
    ordered, disorder = _order_after_cursor(entries, after_seq)
    if disorder is not None:
        return EventRead(
            events=tuple(ordered),
            state=EventStreamState.GAP,
            after_seq=after_seq,
            detail=disorder,
        )

    lost = _lost_range(ordered, after_seq=after_seq, high_water_seq=high_water_seq)
    if lost is not None:
        lost_from, lost_to, detail = lost
        return EventRead(
            events=tuple(ordered),
            state=EventStreamState.GAP,
            after_seq=after_seq,
            lost_from_seq=lost_from,
            lost_to_seq=lost_to,
            detail=detail,
        )

    tail = ordered[-1] if ordered else _last_at_or_before(entries, after_seq)
    terminal = tail is not None and is_terminal_payload(tail.payload)
    return EventRead(
        events=tuple(ordered),
        state=EventStreamState.END_OF_STREAM if terminal else EventStreamState.OPEN,
        after_seq=after_seq,
    )


def _order_after_cursor(
    entries: Sequence[BrokerEvent], after_seq: int
) -> tuple[list[BrokerEvent], str | None]:
    """Drop already-seen events; return ``(ordered, disorder detail)``.

    A re-delivered event (same seq and same dedupe key) is collapsed. Anything
    else out of order means the stored sequence cannot be trusted, so the read
    is reported as a gap rather than silently reordered.
    """
    ordered: list[BrokerEvent] = []
    for entry in entries:
        if entry.seq <= after_seq:
            continue
        if ordered and entry.seq <= ordered[-1].seq:
            if entry.seq == ordered[-1].seq and entry.dedupe_key == ordered[-1].dedupe_key:
                continue
            return ordered, "stored sequence is not strictly increasing"
        ordered.append(entry)
    return ordered, None


def _lost_range(
    ordered: Sequence[BrokerEvent], *, after_seq: int, high_water_seq: int
) -> tuple[int | None, int | None, str] | None:
    """First missing range the retained history cannot cover, if any."""
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if current.seq > previous.seq + 1:
            return previous.seq + 1, current.seq - 1, "sequence hole in stored history"
    if ordered and ordered[0].seq > after_seq + 1:
        return after_seq + 1, ordered[0].seq - 1, "reader cursor predates retained history"
    if not ordered and after_seq > high_water_seq:
        return None, None, "reader cursor is ahead of the source watermark"
    if not ordered and high_water_seq > after_seq:
        return after_seq + 1, high_water_seq, "retained history does not reach the reader cursor"
    return None


def _last_at_or_before(entries: Sequence[BrokerEvent], after_seq: int) -> BrokerEvent | None:
    """The newest retained event the cursor has already consumed."""
    return max(
        (entry for entry in entries if entry.seq <= after_seq),
        key=lambda entry: entry.seq,
        default=None,
    )


class BaseRunBroker(ABC):
    """Abstract base class for a run-specific event broker.

    Handles both live event broadcast and replay storage for SSE reconnection.
    """

    run_id: str

    @abstractmethod
    async def put(self, event_id: str, payload: Any, *, resumable: bool = True) -> None:
        """Publish an event to live subscribers and optionally store for replay.

        ``event_id`` must be allocated once per event and reused across retries,
        so a re-delivery keeps the same identity.
        """

    @abstractmethod
    def aiter(self) -> AsyncIterator[tuple[str, Any]]:
        """Async iterator yielding (event_id, payload) pairs."""
        ...

    @abstractmethod
    async def replay(self, last_event_id: str | None) -> list[tuple[str, Any]]:
        """Return stored events for replay on reconnect.

        Args:
            last_event_id: Last event ID the client received, or None for all.
                Only events strictly after this one are returned.

        Returns:
            List of ``(event_id, payload)`` tuples in publish order.
        """

    @abstractmethod
    def mark_finished(self) -> None:
        """Mark this broker as finished."""

    @abstractmethod
    def is_finished(self) -> bool:
        """Check if this broker is finished."""

    async def stored_events(self) -> list[BrokerEvent]:
        """Return every retained event with its identity.

        Backends that persist the identity (Redis) override this so a re-read
        returns the stored values instead of recomputing them. The default
        derives identity from the event id and the payload.
        """
        return [
            BrokerEvent(
                event_id=event_id,
                seq=extract_event_sequence(event_id),
                dedupe_key=dedupe_key_for(self.run_id, payload),
                payload=payload,
            )
            for event_id, payload in await self.replay(None)
        ]

    async def high_water_seq(self) -> int:
        """Highest sequence the source durably stored for this run.

        Overridden by backends that keep a watermark, so a wiped buffer is still
        recognisable as lost history. Defaults to the newest retained event.
        """
        return max((entry.seq for entry in await self.stored_events()), default=0)

    async def read_events(self, after_seq: int | None = None) -> EventRead:
        """Read retained events after ``after_seq`` with an explicit completeness state.

        This is the read face consumers should use: it returns the missing tail
        and says plainly whether history is complete, gapped or unreadable,
        rather than shortening the answer silently.
        """
        cursor = after_seq or 0
        try:
            entries = await self.stored_events()
            retained = max((entry.seq for entry in entries), default=0)
            high_water = max(await self.high_water_seq(), retained)
        except BrokerUnavailable as exc:
            return EventRead(
                events=(),
                state=EventStreamState.UNAVAILABLE,
                after_seq=cursor,
                detail=str(exc),
            )
        return classify_events(entries, after_seq=cursor, high_water_seq=high_water)

    async def allocate_event_id(self, run_id: str) -> str:
        """Allocate the next event id for this run.

        Default is ``get_event_sequence() + 1`` — not atomic; backends with a
        shared source override it with an atomic allocation.
        """
        return generate_event_id(run_id, await self.get_event_sequence(run_id) + 1)

    async def get_event_sequence(self, run_id: str) -> int:
        """Return the highest stored sequence number for ``run_id``."""
        return await self.high_water_seq()

    async def put_terminal(self, payload: Any) -> str | None:
        """Publish the run's terminal event at most once.

        Returns the published event id, or ``None`` when the run already has a
        terminal event, so duplicate terminal signalling cannot land twice in a
        consumer's ledger.

        The event id is allocated *after* the backend claim succeeds, so losing
        the claim never burns a sequence number and leaves a hole behind.
        """
        if not is_terminal_payload(payload):
            raise ValueError("put_terminal() requires a terminal ('end', ...) payload")
        if self.is_finished():
            return None
        event_id = await self.allocate_event_id(self.run_id)
        await self.put(event_id, payload)
        return event_id


class BaseBrokerManager(ABC):
    """Abstract base class for managing multiple RunBroker instances"""

    @abstractmethod
    def get_or_create_broker(self, run_id: str) -> BaseRunBroker:
        """Get or create a broker for a run"""

    @abstractmethod
    def get_broker(self, run_id: str) -> BaseRunBroker | None:
        """Get an existing broker or None"""

    @abstractmethod
    def cleanup_broker(self, run_id: str) -> None:
        """Clean up a broker for a run"""

    @abstractmethod
    def remove_broker(self, run_id: str) -> None:
        """Remove a broker completely"""

    @abstractmethod
    async def start(self) -> None:
        """Start background tasks (cleanup loop, cancel listener, etc.)"""

    @abstractmethod
    async def stop(self) -> None:
        """Stop background tasks and release resources"""

    @abstractmethod
    async def request_cancel(
        self,
        run_id: str,
        action: RunCancellationAction = "cancel",
        *,
        emit_end_event: bool = True,
    ) -> None:
        """Request cancellation of a run.

        In single-instance mode, cancels the local asyncio task directly.
        In multi-instance mode, broadcasts via Redis pub/sub so the owning
        instance can act on it. ``emit_end_event`` is disabled when the API
        has already reconciled the database and owns terminal signaling.
        """

    @abstractmethod
    async def get_event_sequence(self, run_id: str) -> int:
        """Return the current (highest) event sequence number for a run.

        Used by signal methods (cancel, error) to generate the next event ID
        when they need to inject events into a run's stream.
        """

    async def allocate_event_id(self, run_id: str) -> str:
        """Atomically allocate the next event sequence number and return event_id.

        Default implementation uses get_event_sequence + 1 (not atomic).
        Redis backend overrides with atomic INCR.
        """
        counter = await self.get_event_sequence(run_id) + 1
        return generate_event_id(run_id, counter)
