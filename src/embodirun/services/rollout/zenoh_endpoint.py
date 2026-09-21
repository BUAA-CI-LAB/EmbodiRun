"""Opt-in Zenoh transport with bounded per-route queues and recovery telemetry.

Payload coding is supplied by the caller. StreamSession remains responsible for
session/route/sequence validation. Recovery caches are bounded and volatile;
neither liveliness nor a completed put acknowledges application consumption.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import threading
import time
import uuid
from collections import deque

from .stream import StreamError


def _key(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


class ZenohEndpoint:
    """Return concurrent futures, as WirelessEndpoint does.

    Each peer owns one sending thread so a congested peer cannot occupy another
    peer's sender. A put deadline does not cancel an in-progress native put:
    that route is failed and its trajectory must be reset by the caller.
    """

    def __init__(
        self,
        local,
        peers,
        *,
        session_id,
        encode,
        decode,
        timeout_s=30,
        bind_host="0.0.0.0",
        connect_to=None,
        recovery=True,
        cache_samples=8,
        history_samples=0,
        queue_samples=32,
        queue_bytes=64 * 1024**2,
        max_message_bytes=16 * 1024**2,
        heartbeat_s=0.2,
        priority="DATA_HIGH",
        congestion_control="BLOCK",
    ):
        if (
            not session_id
            or timeout_s <= 0
            or heartbeat_s <= 0
            or not 0 <= history_samples <= cache_samples
            or min(cache_samples, queue_samples, queue_bytes, max_message_bytes) <= 0
        ):
            raise ValueError("session and positive transport limits are required")
        import zenoh

        self._zenoh = zenoh
        self.timeout_s, self.encode, self.decode = timeout_s, encode, decode
        self._local = local["node_id"]
        self._peers = {peer["node_id"]: peer for peer in peers}
        if self._local in self._peers or len(self._peers) != len(peers):
            raise ValueError("peer IDs must be distinct from each other and local ID")
        connectors = [peer for peer in self._peers if peer < self._local] if connect_to is None else list(connect_to)
        if set(connectors) - self._peers.keys():
            raise ValueError("connect_to contains an unknown peer")
        self._peer_keys = {_key(peer): peer for peer in self._peers}
        self._prefix = "embodirun/" + _key(session_id)
        self._recovery, self._cache_samples = recovery, cache_samples
        self._heartbeat = heartbeat_s
        self._priority = getattr(zenoh.Priority, priority)
        self._congestion = getattr(zenoh.CongestionControl, congestion_control)
        self._queue_samples, self._queue_bytes = queue_samples, queue_bytes
        self._max_message_bytes = max_message_bytes
        self._lock = threading.Lock()
        self._messages, self._sizes, self._errors = {}, {}, {}
        self._events, self._publishers, self._send_locks = {}, {}, {}
        self._live_tokens = set()
        self.events = deque(maxlen=4096)
        self.counters = {"received": 0, "overflow": 0, "missed": 0, "published": 0}
        self._closed = False
        self._session = None
        self._executors = {peer: concurrent.futures.ThreadPoolExecutor(max_workers=1) for peer in self._peers}
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        try:
            cfg = zenoh.Config()
            settings = {
                "mode": "peer",
                "listen/endpoints": [f"tcp/{bind_host}:{local['port']}"],
                "connect/endpoints": [
                    f"tcp/{self._peers[peer]['host']}:{self._peers[peer]['port']}" for peer in connectors
                ],
                "connect/exit_on_failure": False,
                "connect/timeout_ms": -1,
                "connect/retry": {
                    "period_init_ms": 100,
                    "period_max_ms": 1000,
                    "period_increase_factor": 2,
                },
                "open/return_conditions/connect_scouted": False,
                "scouting/multicast/enabled": False,
                "scouting/gossip/enabled": False,
                # This endpoint benchmarks the LAN path. Automatic local SHM
                # would change the tested transport and its memory requirements.
                "transport/shared_memory/enabled": False,
            }
            for key, value in settings.items():
                cfg.insert_json5(key, json.dumps(value))
            self._session = zenoh.open(cfg)
            subscription = f"{self._prefix}/msg/*/{_key(self._local)}/*"
            if recovery:
                self._subscriber = zenoh.ext.declare_advanced_subscriber(
                    self._session,
                    subscription,
                    self._receive_sample,
                    history=(
                        zenoh.ext.HistoryConfig(detect_late_publishers=True, max_samples=history_samples)
                        if history_samples
                        else None
                    ),
                    recovery=zenoh.ext.RecoveryConfig(periodic_queries=None, heartbeat=True),
                )
                self._miss_listener = self._subscriber.sample_miss_listener(self._miss)
            else:
                self._subscriber = self._session.declare_subscriber(subscription, self._receive_sample)
            self._liveliness = self._session.liveliness().declare_subscriber(
                f"{self._prefix}/live/*/*", self._live, history=True
            )
            # Published last: this endpoint has already declared its subscriber.
            self._token = self._session.liveliness().declare_token(
                f"{self._prefix}/live/{_key(self._local)}/{uuid.uuid4().hex}"
            )
        except BaseException:
            self.close()
            raise

    def _wake(self, key):
        event = self._events.get(key)
        if event is not None:
            event.set()

    def _live(self, sample):
        token = str(sample.key_expr)
        peer = self._peer_keys.get(token.split("/")[-2])
        if peer is None:
            return
        with self._lock:
            if sample.kind == self._zenoh.SampleKind.PUT:
                self._live_tokens.add(token)
            else:
                self._live_tokens.discard(token)
            self.events.append(
                {
                    "event": "liveliness",
                    "peer": peer,
                    "kind": str(sample.kind),
                    "monotonic_s": time.monotonic(),
                }
            )
        if not self._closed:
            self._loop.call_soon_threadsafe(self._wake, ("live", peer))

    def _miss(self, miss):
        with self._lock:
            self.counters["missed"] += miss.nb
            self.events.append(
                {
                    "event": "missed",
                    "count": miss.nb,
                    "source": str(miss.source),
                    "monotonic_s": time.monotonic(),
                }
            )

    def _receive_sample(self, sample):
        parts = str(sample.key_expr).split("/")
        peer = self._peer_keys.get(parts[-3])
        if peer is None or self._closed:
            return
        key = (peer, parts[-1])
        raw = sample.payload.to_bytes()
        with self._lock:
            if key in self._errors:
                return
            queue = self._messages.setdefault(key, deque())
            size = self._sizes.get(key, 0)
            if (
                len(raw) > self._max_message_bytes
                or len(queue) >= self._queue_samples
                or size + len(raw) > self._queue_bytes
            ):
                self._errors[key] = StreamError("Zenoh receive queue overflow; reset trajectory")
                self.counters["overflow"] += 1
                queue.clear()
                self._sizes[key] = 0
            else:
                queue.append(raw)
                self._sizes[key] = size + len(raw)
                self.counters["received"] += 1
        self._loop.call_soon_threadsafe(self._wake, key)

    def _route(self, peer_id, route):
        if peer_id not in self._peers:
            raise ValueError(f"unknown Zenoh peer {peer_id}")
        tag = hashlib.sha256(json.dumps(route, separators=(",", ":")).encode()).hexdigest()
        return peer_id, tag

    def _submit(self, coroutine):
        if self._closed:
            coroutine.close()
            raise RuntimeError("Zenoh endpoint is closed")
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    def recv(self, peer_id, route):
        return self._submit(self._recv(self._route(peer_id, route)))

    async def _recv(self, key):
        deadline = self._loop.time() + self.timeout_s
        event = self._events.setdefault(key, asyncio.Event())
        while True:
            event.clear()
            with self._lock:
                if key in self._errors:
                    raise self._errors[key]
                queue = self._messages.get(key)
                if queue:
                    raw = queue.popleft()
                    self._sizes[key] -= len(raw)
                    break
            remaining = deadline - self._loop.time()
            if remaining <= 0:
                raise TimeoutError("Zenoh receive deadline exceeded")
            await asyncio.wait_for(event.wait(), remaining)
        return self.decode(raw)

    def send(self, peer_id, route, payload):
        return self._submit(self._send(self._route(peer_id, route), payload))

    async def _send(self, key, payload):
        peer, tag = key
        deadline = self._loop.time() + self.timeout_s
        # Serialize each peer's puts without blocking another peer's event loop.
        lock = self._send_locks.setdefault(peer, asyncio.Lock())
        await asyncio.wait_for(lock.acquire(), self.timeout_s)
        try:
            with self._lock:
                if key in self._errors:
                    raise self._errors[key]
            event = self._events.setdefault(("live", peer), asyncio.Event())
            while True:
                event.clear()
                with self._lock:
                    alive = any(f"/live/{_key(peer)}/" in token for token in self._live_tokens)
                if alive:
                    break
                remaining = deadline - self._loop.time()
                if remaining <= 0:
                    raise TimeoutError(f"Zenoh peer {peer} has no live subscriber")
                await asyncio.wait_for(event.wait(), remaining)
            raw = self.encode(payload)
            if len(raw) > self._max_message_bytes:
                raise ValueError("Zenoh message exceeds configured byte limit")
            if key not in self._publishers:
                topic = f"{self._prefix}/msg/{_key(self._local)}/{_key(peer)}/{tag}"
                options = {
                    "priority": self._priority,
                    "congestion_control": self._congestion,
                    "reliability": self._zenoh.Reliability.RELIABLE,
                    "express": True,
                }
                if self._recovery:
                    self._publishers[key] = self._zenoh.ext.declare_advanced_publisher(
                        self._session,
                        topic,
                        **options,
                        cache=self._zenoh.ext.CacheConfig(max_samples=self._cache_samples),
                        sample_miss_detection=self._zenoh.ext.MissDetectionConfig(
                            heartbeat=self._heartbeat, sporadic_heartbeat=None
                        ),
                        publisher_detection=True,
                    )
                else:
                    self._publishers[key] = self._session.declare_publisher(topic, **options)
            remaining = deadline - self._loop.time()
            if remaining <= 0:
                raise TimeoutError("Zenoh send preparation exceeded deadline")
            future = self._loop.run_in_executor(self._executors[peer], self._publishers[key].put, raw)
            await asyncio.wait_for(future, remaining)
            with self._lock:
                self.counters["published"] += 1
            return {"message_bytes": len(raw), "completion": "local_publish"}
        except (TimeoutError, asyncio.CancelledError):
            with self._lock:
                self._errors[key] = StreamError("Zenoh send interrupted; reset trajectory")
            raise
        finally:
            lock.release()

    async def _shutdown(self):
        pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._session is not None:
                self._session.close()
            asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop).result(self.timeout_s)
        finally:
            for executor in self._executors.values():
                executor.shutdown(wait=True, cancel_futures=True)
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(self.timeout_s)
            if not self._thread.is_alive():
                self._loop.close()
