from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from embodirun.robots.sensors.cameras import CameraFrame
from embodirun.services.control.observations import (
    ObservationExpiredError,
    ObservationProducer,
)


class FakeSource:
    def __init__(self, frame_name: str, *, timestamp_ns: int | None = 100) -> None:
        self.frame_name = frame_name
        self.timestamp_ns = timestamp_ns
        self.calls = 0
        self.failure: Exception | None = None

    def capture(self) -> tuple[CameraFrame, ...]:
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        return (
            CameraFrame(
                self.frame_name,
                "image/jpeg",
                f"frame-{self.calls}".encode(),
                captured_timestamp_ns=self.timestamp_ns,
                received_timestamp_ns=101,
                clock_domain="host_monotonic_ns" if self.timestamp_ns is not None else None,
                profile={"width": 2, "height": 1},
            ),
        )


class BlockingSource:
    def __init__(self, frame_name: str, release: threading.Event) -> None:
        self.frame_name = frame_name
        self.release = release
        self.entered = threading.Event()
        self.calls = 0

    def capture(self) -> tuple[CameraFrame, ...]:
        self.calls += 1
        self.entered.set()
        self.release.wait()
        return (
            CameraFrame(
                self.frame_name,
                "image/jpeg",
                b"blocked",
                captured_timestamp_ns=100,
                received_timestamp_ns=101,
                clock_domain="host_monotonic_ns",
            ),
        )


class LateSource:
    def __init__(self, release: threading.Event) -> None:
        self.release = release
        self.finished = threading.Event()
        self.calls = 0

    def capture(self) -> tuple[CameraFrame, ...]:
        self.calls += 1
        if self.calls == 1:
            self.release.wait()
            payload = b"late-old"
        else:
            payload = b"fresh-new"
        self.finished.set()
        return (
            CameraFrame(
                "front",
                "image/jpeg",
                payload,
                captured_timestamp_ns=100,
                received_timestamp_ns=101,
                clock_domain="host_monotonic_ns",
            ),
        )


def test_one_producer_shares_immutable_snapshot_with_multiple_consumers() -> None:
    mutable_state = {"joints": [1.0, 2.0]}
    source = FakeSource("front")
    producer = ObservationProducer(
        lambda: SimpleNamespace(
            values=mutable_state,
            timestamp_s=0.0000001,
            metadata={"clock_domain": "host_monotonic_ns"},
        ),
        camera_sources={"front-camera": source},
        clock_ns=lambda: 200,
        service_instance_id="service-a",
    )
    subscriptions = [producer.subscribe(max_queue=1) for _ in range(4)]

    snapshot = producer.publish_once()
    assert source.calls == 1
    assert [subscription.get(timeout=0).observation_id for subscription in subscriptions] == [
        snapshot.observation_id
    ] * 4
    assert snapshot.state["joints"] == (1.0, 2.0)
    with pytest.raises(TypeError):
        snapshot.state["joints"] = (3.0,)  # type: ignore[index]
    mutable_state["joints"].append(3.0)
    assert snapshot.state["joints"] == (1.0, 2.0)
    assert snapshot.cameras[0].profile["width"] == 2  # type: ignore[index]
    with pytest.raises(TypeError):
        snapshot.cameras[0].profile["width"] = 4  # type: ignore[index]


def test_subscription_is_bounded_and_reports_dropped_snapshots() -> None:
    source = FakeSource("front")
    producer = ObservationProducer(
        camera_sources={"front-camera": source},
        clock_ns=lambda: 200,
        service_instance_id="service-b",
    )
    subscription = producer.subscribe(max_queue=1)

    first = producer.publish_once()
    second = producer.publish_once()
    third = producer.publish_once()

    assert subscription.dropped_count == 2
    assert subscription.get(timeout=0).observation_id == third.observation_id
    assert first.observation_id != third.observation_id
    assert second.observation_id != third.observation_id


def test_source_failure_is_explicit_and_other_source_keeps_publishing() -> None:
    good = FakeSource("front")
    bad = FakeSource("wrist")
    bad.failure = RuntimeError("camera disconnected")
    producer = ObservationProducer(
        camera_sources={"good": good, "bad": bad},
        clock_ns=lambda: 200,
        service_instance_id="service-c",
        max_retained=2,
    )

    failed = producer.publish_once()
    assert [frame.name for frame in failed.cameras] == ["front"]
    assert failed.available is True
    assert failed.stale is True
    assert "bad" in failed.errors
    assert producer.source_status("bad").available is False
    assert producer.source_status("bad").last_captured_timestamp_ns is None

    old_id = failed.observation_id
    producer.reconnect("bad")
    with pytest.raises(ObservationExpiredError):
        producer.get(old_id)
    bad.failure = None
    recovered = producer.publish_once()
    assert recovered.generation == failed.generation + 1
    assert {frame.name for frame in recovered.cameras} == {"front", "wrist"}
    assert recovered.observation_id != old_id


def test_reconnect_drops_an_inflight_capture_before_new_generation() -> None:
    release = threading.Event()
    source = BlockingSource("front", release)
    producer = ObservationProducer(
        camera_sources={"front": source},
        source_timeout_s=0.05,
        clock_ns=lambda: 200,
        service_instance_id="service-reconnect-inflight",
    )
    result: list[object] = []

    def publish() -> None:
        try:
            result.append(producer.publish_once())
        except Exception as error:  # the reconnect boundary is the assertion
            result.append(error)

    thread = threading.Thread(target=publish)
    thread.start()
    assert source.entered.wait(1.0)
    generation = producer.reconnect("front")
    release.set()
    thread.join(1.0)

    assert len(result) == 1
    assert isinstance(result[0], Exception)
    assert producer.publish_once().generation == generation
    assert source.calls == 2
    producer.close()


def test_late_timeout_result_is_dropped_before_a_new_capture() -> None:
    release = threading.Event()
    source = LateSource(release)
    producer = ObservationProducer(
        camera_sources={"front": source},
        source_timeout_s=0.01,
        clock_ns=lambda: 200,
        service_instance_id="service-late-timeout",
    )

    timed_out = producer.publish_once()
    assert timed_out.cameras == ()
    assert "front" in timed_out.errors
    release.set()
    assert source.finished.wait(1.0)

    fresh = producer.publish_once()

    assert source.calls == 2
    assert [frame.data for frame in fresh.cameras] == [b"fresh-new"]
    assert fresh.errors == {}
    assert fresh.stale is False
    producer.close()


def test_blocked_camera_is_bounded_and_does_not_hold_up_healthy_camera() -> None:
    release = threading.Event()
    blocked = BlockingSource("blocked", release)
    healthy = FakeSource("healthy")
    producer = ObservationProducer(
        camera_sources={"blocked": blocked, "healthy": healthy},
        source_timeout_s=0.02,
        clock_ns=lambda: 200,
        service_instance_id="service-blocked",
    )
    try:
        started = time.monotonic()
        first = producer.publish_once()
        elapsed = time.monotonic() - started
        assert elapsed < 0.5
        assert blocked.entered.wait(0.1)
        assert [frame.name for frame in first.cameras] == ["healthy"]
        assert "blocked" in first.errors
        assert blocked.calls == 1
        assert healthy.calls == 1

        second = producer.publish_once()
        assert [frame.name for frame in second.cameras] == ["healthy"]
        assert blocked.calls == 1  # no second in-flight capture was queued
        assert healthy.calls == 2
    finally:
        release.set()
        assert producer.close(timeout_s=1.0)


def test_single_foreign_clock_source_is_stale_and_never_fresh() -> None:
    def source() -> tuple[CameraFrame, ...]:
        return (
            CameraFrame(
                "foreign",
                "image/jpeg",
                b"foreign",
                captured_timestamp_ns=100,
                received_timestamp_ns=101,
                clock_domain="other_node_monotonic_ns",
            ),
        )

    producer = ObservationProducer(
        camera_sources={"foreign": source},
        clock_ns=lambda: 200,
        clock_domain="host_monotonic_ns",
        service_instance_id="service-foreign",
    )

    snapshot = producer.publish_once()

    assert snapshot.stale is True
    assert "foreign" in snapshot.errors
    assert snapshot.clock_domains["foreign"] == "other_node_monotonic_ns"
    assert (
        snapshot.is_fresh(
            now_ns=200,
            now_clock_domain="host_monotonic_ns",
            max_age_ns=1_000,
        )
        is False
    )


def test_freshness_requires_callers_clock_domain_to_match_snapshot() -> None:
    producer = ObservationProducer(
        camera_sources={"front": FakeSource("front")},
        clock_ns=lambda: 200,
        clock_domain="host_monotonic_ns",
        service_instance_id="service-clock-check",
    )
    snapshot = producer.publish_once()

    assert snapshot.stale is False
    assert (
        snapshot.is_fresh(
            now_ns=200,
            now_clock_domain="host_monotonic_ns",
            max_age_ns=1_000,
        )
        is True
    )
    assert (
        snapshot.is_fresh(
            now_ns=200,
            now_clock_domain="other_node_monotonic_ns",
            max_age_ns=1_000,
        )
        is False
    )


def test_state_freshness_uses_explicit_capture_metadata_and_keeps_legacy_time() -> None:
    raw_timestamp_s = 1_700_000_000.25
    producer = ObservationProducer(
        state_reader=lambda: SimpleNamespace(
            values={"joint": 1.0},
            timestamp_s=raw_timestamp_s,
            metadata={
                "captured_timestamp_ns": 100,
                "clock_domain": "host_monotonic_ns",
            },
        ),
        clock_ns=lambda: 200,
        service_instance_id="service-state-time",
    )

    snapshot = producer.publish_once()

    assert snapshot.source_timestamps_ns["state"] == 100
    assert snapshot.metadata["state_metadata"]["timestamp_s"] == raw_timestamp_s
    assert snapshot.stale is False
    assert (
        snapshot.is_fresh(
            now_ns=200,
            now_clock_domain="host_monotonic_ns",
            max_age_ns=1_000,
        )
        is True
    )


def test_state_received_timestamp_is_sampled_after_reader_returns() -> None:
    clock_value = 100

    def clock_ns() -> int:
        return clock_value

    def state_reader() -> SimpleNamespace:
        nonlocal clock_value
        clock_value = 140
        captured_timestamp_ns = clock_value
        clock_value = 150
        return SimpleNamespace(
            values={"joint": 1.0},
            timestamp_s=1.0,
            metadata={
                "captured_timestamp_ns": captured_timestamp_ns,
                "clock_domain": "host_monotonic_ns",
            },
        )

    producer = ObservationProducer(
        state_reader=state_reader,
        clock_ns=clock_ns,
        service_instance_id="service-state-receipt-order",
    )
    try:
        snapshot = producer.publish_once()
    finally:
        producer.close()

    assert snapshot.source_timestamps_ns["state"] == 140
    assert snapshot.source_received_timestamps_ns["state"] == 150
    assert snapshot.source_received_timestamps_ns["state"] > snapshot.source_timestamps_ns["state"]
    assert snapshot.clock_domains["state"] == "host_monotonic_ns"


def test_missing_timestamp_is_unknown_and_never_fresh() -> None:
    source = FakeSource("legacy", timestamp_ns=None)
    producer = ObservationProducer(
        camera_sources={"legacy": source},
        clock_ns=lambda: 200,
        service_instance_id="service-d",
    )

    snapshot = producer.publish_once()
    assert snapshot.captured_timestamp_ns is None
    assert snapshot.skew_ns is None
    assert snapshot.stale is True
    assert (
        snapshot.is_fresh(
            now_ns=200,
            now_clock_domain="host_monotonic_ns",
            max_age_ns=1_000,
        )
        is False
    )


def test_cross_domain_timestamps_keep_skew_unknown() -> None:
    left = FakeSource("left")

    def right_capture() -> tuple[CameraFrame, ...]:
        return (
            CameraFrame(
                "right",
                "image/jpeg",
                b"right",
                captured_timestamp_ns=100,
                received_timestamp_ns=101,
                clock_domain="other_node_monotonic_ns",
            ),
        )

    producer = ObservationProducer(
        camera_sources={"left": left, "right": right_capture},
        clock_ns=lambda: 200,
        service_instance_id="service-e",
    )
    snapshot = producer.publish_once()
    assert snapshot.skew_ns is None
    assert (
        snapshot.is_fresh(
            now_ns=200,
            now_clock_domain="host_monotonic_ns",
            max_age_ns=1_000,
        )
        is False
    )


@pytest.mark.parametrize(
    ("configured", "expected"),
    [(None, 0.03), ("0.05", 0.05), ("0", 0.03), ("-1", 0.03), ("nan", 0.03), ("bad", 0.03)],
)
def test_producer_poll_interval_does_not_cap_the_stream_at_ten_hertz(monkeypatch, configured, expected) -> None:
    """A fixed 0.1 s poll starves a 20 Hz control loop; the default must be lower."""
    if configured is None:
        monkeypatch.delenv("RLINF_DEPLOY_OBSERVATION_INTERVAL_S", raising=False)
    else:
        monkeypatch.setenv("RLINF_DEPLOY_OBSERVATION_INTERVAL_S", configured)

    producer = ObservationProducer(lambda: None)

    assert producer.interval_s == pytest.approx(expected)
