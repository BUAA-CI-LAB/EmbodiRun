"""Bounded retention and non-blocking subscriptions for observations."""

from __future__ import annotations

import os
import socket
import threading
import time
from collections import deque
from queue import Empty

from .values import (
    ObservationError,
    ObservationExpiredError,
    ObservationSnapshot,
    ObservationSubscriptionClosed,
    ObservationUnavailableError,
)


def _default_service_instance_id() -> str:
    try:
        host = socket.gethostname() or "unknown-host"
    except OSError:
        host = "unknown-host"
    return f"{host}:{os.getpid()}:{time.time_ns()}"


class ObservationSubscription:
    """A bounded non-blocking producer queue with visible drops."""

    def __init__(self, store: ObservationStore, max_queue: int) -> None:
        self._store = store
        self._max_queue = max_queue
        self._queue: deque[ObservationSnapshot] = deque()
        self._condition = threading.Condition()
        self._dropped_count = 0
        self._closed = False

    @property
    def max_queue(self) -> int:
        return self._max_queue

    @property
    def dropped_count(self) -> int:
        with self._condition:
            return self._dropped_count

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def _offer(self, snapshot: ObservationSnapshot) -> None:
        with self._condition:
            if self._closed:
                return
            if len(self._queue) >= self._max_queue:
                self._queue.popleft()
                self._dropped_count += 1
            self._queue.append(snapshot)
            self._condition.notify()

    def get(self, timeout: float | None = None) -> ObservationSnapshot:
        """Read the next snapshot; a slow reader never blocks publication."""

        if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float))):
            raise TypeError("timeout must be a number or None")
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._condition:
            while not self._queue:
                if self._closed:
                    raise ObservationSubscriptionClosed("observation subscription is closed")
                if deadline is None:
                    self._condition.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise Empty
                    self._condition.wait(remaining)
            return self._queue.popleft()

    def get_nowait(self) -> ObservationSnapshot:
        return self.get(timeout=0)

    def close(self) -> None:
        self._store.unsubscribe(self)

    def __iter__(self):
        while True:
            try:
                yield self.get()
            except ObservationSubscriptionClosed:
                return


class ObservationStore:
    """Retain a bounded latest/history window and fan out bounded queues."""

    def __init__(
        self,
        *,
        max_retained: int = 32,
        service_instance_id: str | None = None,
    ) -> None:
        if isinstance(max_retained, bool) or not isinstance(max_retained, int):
            raise TypeError("max_retained must be an integer")
        if max_retained <= 0:
            raise ValueError("max_retained must be positive")
        self.max_retained = max_retained
        self.service_instance_id = service_instance_id or _default_service_instance_id()
        if not isinstance(self.service_instance_id, str) or not self.service_instance_id.strip():
            raise ValueError("service_instance_id must not be empty")
        self._lock = threading.RLock()
        self._retained: deque[ObservationSnapshot] = deque(maxlen=max_retained)
        self._by_id: dict[str, ObservationSnapshot] = {}
        self._retired_ids: deque[str] = deque(maxlen=max_retained * 2)
        self._subscriptions: set[ObservationSubscription] = set()
        self._latest: ObservationSnapshot | None = None
        self._generation = 0
        self._next_sequence = 0
        self._closed = False

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def next_observation_id(self) -> tuple[str, int, int]:
        with self._lock:
            if self._closed:
                raise ObservationError("observation store is closed")
            self._next_sequence += 1
            sequence = self._next_sequence
            observation_id = f"{self.service_instance_id}:g{self._generation}:o{sequence}"
            return observation_id, self._generation, sequence

    def publish(self, snapshot: ObservationSnapshot) -> ObservationSnapshot:
        if not isinstance(snapshot, ObservationSnapshot):
            raise TypeError("snapshot must be an ObservationSnapshot")
        with self._lock:
            if self._closed:
                raise ObservationError("observation store is closed")
            if snapshot.service_instance_id != self.service_instance_id:
                raise ValueError("snapshot belongs to a different service instance")
            if snapshot.generation != self._generation:
                raise ObservationUnavailableError(
                    snapshot.observation_id,
                    "snapshot belongs to an expired reconnect generation",
                )
            if snapshot.observation_id in self._by_id or snapshot.observation_id in self._retired_ids:
                raise ValueError("observation_id cannot be published more than once")
            if self._retained and snapshot.sequence <= self._retained[-1].sequence:
                raise ValueError("observation sequence must increase")
            if self._retained and len(self._retained) == self.max_retained:
                retired = self._retained[0]
                self._by_id.pop(retired.observation_id, None)
                self._retired_ids.append(retired.observation_id)
            self._retained.append(snapshot)
            self._by_id[snapshot.observation_id] = snapshot
            self._latest = snapshot
            subscribers = tuple(self._subscriptions)
        for subscription in subscribers:
            # ``_offer`` is bounded and never waits for the consumer.
            subscription._offer(snapshot)
        return snapshot

    def latest(self) -> ObservationSnapshot | None:
        with self._lock:
            return self._latest

    def get(self, observation_id: str) -> ObservationSnapshot:
        if not isinstance(observation_id, str) or not observation_id.strip():
            raise ValueError("observation_id must not be empty")
        with self._lock:
            snapshot = self._by_id.get(observation_id)
            if snapshot is not None:
                return snapshot
            reason = (
                "observation expired from bounded retention"
                if observation_id in self._retired_ids
                else "observation ID is unknown or belongs to another generation"
            )
        if "expired" in reason:
            raise ObservationExpiredError(observation_id, reason)
        raise ObservationUnavailableError(observation_id, reason)

    def try_get(self, observation_id: str) -> ObservationSnapshot | None:
        try:
            return self.get(observation_id)
        except ObservationUnavailableError:
            return None

    def subscribe(
        self,
        max_queue: int = 8,
        *,
        queue_size: int | None = None,
        replay_latest: bool = False,
    ) -> ObservationSubscription:
        if queue_size is not None:
            max_queue = queue_size
        if isinstance(max_queue, bool) or not isinstance(max_queue, int):
            raise TypeError("max_queue must be an integer")
        if max_queue <= 0:
            raise ValueError("max_queue must be positive")
        subscription = ObservationSubscription(self, max_queue)
        with self._lock:
            if self._closed:
                raise ObservationError("observation store is closed")
            self._subscriptions.add(subscription)
            latest = self._latest if replay_latest else None
        if latest is not None:
            subscription._offer(latest)
        return subscription

    def unsubscribe(self, subscription: ObservationSubscription) -> None:
        with self._lock:
            self._subscriptions.discard(subscription)
        with subscription._condition:
            subscription._closed = True
            subscription._condition.notify_all()

    def begin_generation(self) -> int:
        """Invalidate old IDs after a reconnect or owner restart."""

        with self._lock:
            if self._closed:
                raise ObservationError("observation store is closed")
            for snapshot in self._retained:
                self._retired_ids.append(snapshot.observation_id)
            self._retained.clear()
            self._by_id.clear()
            self._latest = None
            self._generation += 1
            return self._generation

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            subscriptions = tuple(self._subscriptions)
            self._subscriptions.clear()
            self._retained.clear()
            self._by_id.clear()
            self._latest = None
        for subscription in subscriptions:
            with subscription._condition:
                subscription._closed = True
                subscription._condition.notify_all()


__all__ = ["ObservationStore", "ObservationSubscription"]
