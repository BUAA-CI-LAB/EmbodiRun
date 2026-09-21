import socket
import time
import uuid
from dataclasses import dataclass

import pytest

from embodirun.services.rollout.segmented_codec import SegmentedCodec
from embodirun.services.rollout.stream import StreamError, StreamSession
from embodirun.services.rollout.zenoh_endpoint import ZenohEndpoint

pytest.importorskip("zenoh")
pytest.importorskip("wireless_comm")


def address(name):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return {"node_id": name, "host": "127.0.0.1", "port": sock.getsockname()[1]}


def endpoints(count=2, **kwargs):
    peers = [address(chr(ord("a") + i)) for i in range(count)]
    codec = SegmentedCodec()
    session = uuid.uuid4().hex
    result = []
    try:
        for local in peers:
            result.append(
                ZenohEndpoint(
                    local,
                    [peer for peer in peers if peer != local],
                    session_id=session,
                    encode=codec.encode,
                    decode=codec.decode,
                    bind_host="127.0.0.1",
                    timeout_s=2,
                    **kwargs,
                )
            )
        return result
    except BaseException:
        for endpoint in result:
            endpoint.close()
        raise


def wait_until(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError("test condition did not become true")
        time.sleep(0.01)


def test_segmented_codec_preserves_registered_tensor_bytes_and_rejects_truncation():
    import torch

    @dataclass
    class Sample:
        tensor: object
        state: tuple

    codec = SegmentedCodec(lambda registry: registry.register_dataclass(Sample, type_id="sample"))
    value = Sample(torch.tensor([0.0, -0.0, 1.5], dtype=torch.bfloat16), (1, b"jpeg"))
    raw = codec.encode(value)
    restored = codec.decode(raw)
    assert type(restored) is Sample
    assert restored.state == value.state
    assert torch.equal(restored.tensor.view(torch.uint8), value.tensor.view(torch.uint8))
    for bad in (raw[:-1], raw + b"extra", b"badmagic" + raw[8:]):
        with pytest.raises(ValueError):
            codec.decode(bad)


@pytest.mark.parametrize("recovery", [False, True])
def test_zenoh_round_trip_preserves_contiguous_stream_and_pending_cancel(recovery):
    a, b = endpoints(recovery=recovery)
    sender, receiver = StreamSession("trial"), StreamSession("trial")
    route = ("observations", 0)
    try:
        for i in range(3):
            pending = b.recv("a", route)
            message = sender.pack(route, {"index": i, "image": bytes([i]) * 65536})
            a.send("b", route, message).result(4)
            assert receiver.unpack(route, pending.result(4)) == message["payload"]
        waiting = b.recv("a", route)
        assert waiting.cancel()
        pending = b.recv("a", route)
        a.send("b", route, sender.pack(route, "fresh")).result(4)
        assert receiver.unpack(route, pending.result(4)) == "fresh"
        b.send("a", ("response",), b"memory-only-action").result(4)
        assert a.recv("b", ("response",)).result(4) == b"memory-only-action"
    finally:
        a.close()
        b.close()


def test_slow_consumer_queue_overflow_is_explicit_and_other_body_continues():
    a, b, c = endpoints(3, queue_samples=2)
    try:
        for i in range(4):
            a.send("b", ("slow",), i).result(4)
        wait_until(lambda: b.counters["overflow"] == 1)
        with pytest.raises(StreamError, match="overflow"):
            b.recv("a", ("slow",)).result(4)
        healthy = c.recv("a", ("healthy",))
        a.send("c", ("healthy",), "new request").result(4)
        assert healthy.result(4) == "new request"
        b.close()
        wait_until(lambda: any("DELETE" in event["kind"] for event in a.events if event["event"] == "liveliness"))
        healthy = c.recv("a", ("healthy",))
        a.send("c", ("healthy",), "after other body closed").result(4)
        assert healthy.result(4) == "after other body closed"
    finally:
        for endpoint in (a, b, c):
            endpoint.close()


def test_recovery_history_cannot_silently_restart_a_consumed_trajectory():
    import zenoh

    a, b = endpoints(cache_samples=4)
    sender, receiver = StreamSession("trial"), StreamSession("trial")
    route = ("actions",)
    try:
        pending = b.recv("a", route)
        a.send("b", route, sender.pack(route, "consumed")).result(4)
        assert receiver.unpack(route, pending.result(4)) == "consumed"
        # Exercise the real Zenoh history cache with a late subscriber. This is
        # cache retrieval, explicitly not a physical/network recovery benchmark.
        b._subscriber.undeclare()
        a.send("b", route, sender.pack(route, "during subscriber absence")).result(4)
        b._subscriber = zenoh.ext.declare_advanced_subscriber(
            b._session,
            f"{b._prefix}/msg/*/*/*",
            b._receive_sample,
            history=zenoh.ext.HistoryConfig(detect_late_publishers=True, max_samples=4),
            recovery=zenoh.ext.RecoveryConfig(periodic_queries=None, heartbeat=True),
        )
        old = b.recv("a", route).result(4)
        with pytest.raises(StreamError, match="expected sequence 1"):
            receiver.unpack(route, old)
        with pytest.raises(StreamError, match="foreign session"):
            StreamSession("new-trial").unpack(route, old)
    finally:
        a.close()
        b.close()
