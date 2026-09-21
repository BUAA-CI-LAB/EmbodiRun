import socket

import pytest

from embodirun.services.rollout.replay import ReplaySource
from embodirun.services.rollout.stream import (
    StreamError,
    StreamSession,
    WirelessEndpoint,
)


def test_stream_rejects_old_actions_foreign_sessions_and_missing_messages():
    route = ("robot", "rollout", 0)
    sender, receiver = StreamSession("run1"), StreamSession("run1")
    first = sender.pack(route, {"action": [1, 2]})
    assert receiver.unpack(route, first) == {"action": [1, 2]}
    with pytest.raises(StreamError):
        receiver.unpack(route, first)
    with pytest.raises(StreamError):
        StreamSession("run2").unpack(route, first)
    with pytest.raises(StreamError):
        receiver.unpack(("other-robot",), sender.pack(route, {}))
    with pytest.raises(StreamError):
        receiver.unpack(route, sender.pack(route, {}))


def test_replay_source_has_no_actuation_boundary():
    source = ReplaySource(state_dim=6, image_size=16)
    assert source.observation(0) == source.observation(100)
    assert len(source.observation(0)["state"]) == 6
    assert len(source.observation(0)["images"]["front"]) == 16 * 16 * 3
    assert not hasattr(source, "execute")
    assert not hasattr(source, "connect")
    assert source.observation(0)["images"]["front"] != source.observation(0)["images"]["wrist"]


def test_replay_fingerprint_includes_decoded_images_and_state():
    a = ReplaySource(state_dim=6, image_size=16)
    b = ReplaySource(state_dim=6, image_size=16)
    assert a.fingerprint() == b.fingerprint()
    b.frames[0]["images"]["wrist"] = bytes(16 * 16 * 3)
    assert a.fingerprint() != b.fingerprint()
    b = ReplaySource(state_dim=6, image_size=16)
    b.frames[0]["state"][0] = 1.0
    assert a.fingerprint() != b.fingerprint()


def free_peer(name):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return {"node_id": name, "host": "127.0.0.1", "port": sock.getsockname()[1]}


def test_wireless_peer_outage_does_not_block_other_peer_and_restart_recovers():
    pytest.importorskip("wireless_comm")
    from wireless_comm import CommError

    a, b, c = free_peer("a"), free_peer("b"), free_peer("c")
    sender = WirelessEndpoint(a, [b, c], timeout_s=1)
    healthy = WirelessEndpoint(c, [a, b], timeout_s=1)
    restarted = WirelessEndpoint(b, [a, c], timeout_s=1)
    try:
        receive = restarted.recv("a", ("before-disconnect",))
        sender.send("b", ("before-disconnect",), b"connected").result(3)
        assert receive.result(3) == b"connected"
        restarted.close()
        # Peer b disconnects while c continues serving real TCP messages.
        failed = sender.send("b", ("action",), b"unavailable")
        receive = healthy.recv("a", ("action",))
        sender.send("c", ("action",), b"healthy").result(3)
        assert receive.result(3) == b"healthy"
        with pytest.raises(CommError):
            failed.result(3)
        restarted = WirelessEndpoint(b, [a, c], timeout_s=1)
        receive = restarted.recv("a", ("new-session",))
        sender.send("b", ("new-session",), b"recovered").result(3)
        assert receive.result(3) == b"recovered"
    finally:
        sender.close()
        healthy.close()
        if restarted:
            restarted.close()


@pytest.mark.parametrize("retry", [False, True])
def test_receive_rearm_preserves_sequence_and_other_peer_service(retry):
    pytest.importorskip("wireless_comm")
    from wireless_comm import ConnectionClosedError

    a, b, c = free_peer("a"), free_peer("b"), free_peer("c")
    sender = WirelessEndpoint(a, [b, c], timeout_s=2)
    receiver = WirelessEndpoint(b, [a, c], timeout_s=2, retry_disconnected_receive=retry)
    healthy = WirelessEndpoint(c, [a, b], timeout_s=2)
    outgoing, incoming = StreamSession("same-session"), StreamSession("same-session")
    route = ("observations",)
    try:
        pending = receiver.recv("a", route)
        sender.send("b", route, outgoing.pack(route, "first")).result(3)
        assert incoming.unpack(route, pending.result(3)) == "first"
        pending = receiver.recv("a", route)
        sender.close()
        # The other peer remains usable during the disconnection.
        good = receiver.recv("c", ("healthy",))
        healthy.send("b", ("healthy",), "ok").result(3)
        assert good.result(3) == "ok"
        if not retry:
            with pytest.raises(ConnectionClosedError):
                pending.result(3)
            return
        sender = WirelessEndpoint(a, [b, c], timeout_s=2)
        sender.send("b", route, outgoing.pack(route, "second")).result(3)
        assert incoming.unpack(route, pending.result(3)) == "second"
        assert receiver.receive_disconnect_retries >= 1
        outgoing.pack(route, "lost")
        pending = receiver.recv("a", route)
        sender.send("b", route, outgoing.pack(route, "after-loss")).result(3)
        with pytest.raises(StreamError, match="expected sequence 2"):
            incoming.unpack(route, pending.result(3))
    finally:
        sender.close()
        receiver.close()
        healthy.close()


def test_receive_rearm_keeps_original_timeout():
    pytest.importorskip("wireless_comm")
    from wireless_comm import OperationTimeoutError

    a, b = free_peer("a"), free_peer("b")
    sender = WirelessEndpoint(a, [b], timeout_s=1)
    receiver = WirelessEndpoint(b, [a], timeout_s=0.3, retry_disconnected_receive=True)
    try:
        first = receiver.recv("a", ("data",))
        sender.send("b", ("data",), "first").result(2)
        assert first.result(2) == "first"
        pending = receiver.recv("a", ("data",))
        sender.close()
        with pytest.raises(OperationTimeoutError):
            pending.result(1)
        assert receiver.receive_disconnect_retries >= 1
    finally:
        sender.close()
        receiver.close()
