from __future__ import annotations

import errno
import json
import threading
import time
from types import SimpleNamespace

import pytest

from embodirun.robots.sensors.cameras import CameraFrame
from embodirun.services.control.observations import (
    ActionEvent,
    ObservationProducer,
    ObservationRecorder,
    RecordingError,
)


class Source:
    def __init__(self) -> None:
        self.calls = 0

    def capture(self) -> tuple[CameraFrame, ...]:
        self.calls += 1
        return (
            CameraFrame(
                "front",
                "image/jpeg",
                f"frame-{self.calls}".encode(),
                captured_timestamp_ns=100,
                received_timestamp_ns=101,
                clock_domain="host_monotonic_ns",
                profile={"width": 2, "height": 1, "fps": 30},
            ),
        )


def wait_until(predicate, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate()


def test_recorder_writes_raw_media_timestamps_and_redacts_credentials(tmp_path) -> None:
    source = Source()
    producer = ObservationProducer(
        lambda: SimpleNamespace(
            values={"joints": [1.0], "password": "do-not-store"},
            timestamp_s=0.0000001,
            metadata={
                "clock_domain": "host_monotonic_ns",
                "captured_timestamp_ns": 100,
                "api_key": "do-not-store",
            },
        ),
        camera_sources={"front": source},
        clock_ns=lambda: 200,
        service_instance_id="recording-service",
    )
    recorder = ObservationRecorder(
        producer.store,
        tmp_path,
        "take-1",
        expected_frame_names=("front",),
        clock_ns=lambda: 200,
    )
    recorder.start()
    recorder.record_action(
        ActionEvent(
            "proposal-1",
            source="agent",
            stage="proposal",
            executed=False,
            observation_id="recording-service:g0:o1",
            payload={"token": "do-not-store"},
        )
    )
    recorder.record_action(
        ActionEvent(
            "execution-1",
            source="control",
            stage="executed",
            executed=True,
            observation_id="recording-service:g0:o1",
            outcome="accepted",
        )
    )
    snapshot = producer.publish_once()
    wait_until(lambda: recorder.status().observation_count == 1)
    assert recorder.stop(timeout_s=1.0)

    status = recorder.status()
    assert status.state == "stopped"
    assert status.incomplete is False
    record = recorder.get_record(snapshot.observation_id)
    assert record is not None
    assert record["state"]["password"] == "<redacted>"
    assert record["metadata"]["state_metadata"]["api_key"] == "<redacted>"
    assert record["timestamps"]["captured_timestamp_ns"] == 100
    assert record["timestamps"]["recorded_timestamp_ns"] == 200
    assert record["timestamps"]["capture_to_record_ns"] == 100
    media = record["cameras"][0]
    assert media["encoding"] == "source_encoded_bytes"
    assert media["encoding_timestamp_ns"] is None
    assert (recorder.path / media["path"]).read_bytes() == b"frame-1"
    assert media["profile"] == {"fps": 30, "height": 1, "width": 2}

    actions = list(recorder.iter_actions())
    assert [event["executed"] for event in actions] == [False, True]
    assert actions[0]["payload"]["token"] == "<redacted>"
    assert actions[0]["observation_id"] == snapshot.observation_id

    manifest = json.loads((recorder.path / "manifest.json").read_text())
    assert manifest["state"] == "stopped"
    assert manifest["incomplete"] is False
    # The recorder owns only its subscription; the producer remains usable.
    producer.publish_once()
    assert source.calls == 2
    producer.close()


def test_recorder_writes_all_frames_in_one_snapshot_directory(tmp_path) -> None:
    class ThreeSource:
        def capture(self) -> tuple[CameraFrame, ...]:
            return tuple(
                CameraFrame(
                    name,
                    mime,
                    data,
                    captured_timestamp_ns=100,
                    received_timestamp_ns=101,
                    clock_domain="host_monotonic_ns",
                )
                for name, mime, data in (
                    ("front", "image/jpeg", b"front"),
                    ("wrist", "image/jpeg", b"wrist"),
                    ("depth", "image/png", b"depth"),
                )
            )

    producer = ObservationProducer(
        camera_sources={"cameras": ThreeSource()},
        clock_ns=lambda: 200,
        service_instance_id="three-frame-service",
    )
    recorder = ObservationRecorder(
        producer.store,
        tmp_path,
        "three-frames",
        expected_frame_names=("front", "wrist", "depth"),
        clock_ns=lambda: 200,
    )
    recorder.start()
    snapshot = producer.publish_once()
    wait_until(lambda: recorder.status().observation_count == 1)
    assert recorder.stop(timeout_s=1.0)

    record = recorder.get_record(snapshot.observation_id)
    assert record is not None
    assert [frame["name"] for frame in record["cameras"]] == [
        "front",
        "wrist",
        "depth",
    ]
    assert [(recorder.path / frame["path"]).read_bytes() for frame in record["cameras"]] == [
        b"front",
        b"wrist",
        b"depth",
    ]
    assert recorder.status().state == "stopped"
    producer.close()


def test_recorder_media_paths_survive_store_reconnect_generation(tmp_path) -> None:
    source = Source()
    producer = ObservationProducer(
        camera_sources={"front": source},
        clock_ns=lambda: 200,
        service_instance_id="generation-service",
    )
    recorder = ObservationRecorder(
        producer.store,
        tmp_path,
        "generations",
        expected_frame_names=("front",),
        clock_ns=lambda: 200,
    )
    recorder.start()
    first = producer.publish_once()
    wait_until(lambda: recorder.status().observation_count == 1)
    producer.reconnect("front")
    second = producer.publish_once()
    wait_until(lambda: recorder.status().observation_count == 2)
    assert recorder.stop(timeout_s=1.0)

    first_record = recorder.get_record(first.observation_id)
    second_record = recorder.get_record(second.observation_id)
    assert first_record is not None and second_record is not None
    first_path = first_record["cameras"][0]["path"]
    second_path = second_record["cameras"][0]["path"]
    assert first_record["generation"] != second_record["generation"]
    assert first_path != second_path
    assert (recorder.path / first_path).read_bytes() == b"frame-1"
    assert (recorder.path / second_path).read_bytes() == b"frame-2"
    producer.close()


def test_recorder_exposes_subscription_overflow_and_missing_frames(tmp_path) -> None:
    source = Source()
    producer = ObservationProducer(
        camera_sources={"front": source},
        clock_ns=lambda: 200,
        service_instance_id="overflow-service",
    )
    recorder = ObservationRecorder(
        producer.store,
        tmp_path,
        "overflow",
        queue_size=1,
        expected_frame_names=("front", "wrist"),
    )
    entered = threading.Event()
    release = threading.Event()
    original_write = recorder._write_snapshot

    def slow_write(snapshot) -> None:
        entered.set()
        release.wait()
        original_write(snapshot)

    recorder._write_snapshot = slow_write
    recorder.start()
    producer.publish_once()
    assert entered.wait(1.0)
    producer.publish_once()
    producer.publish_once()
    release.set()
    wait_until(lambda: recorder.status().observation_count >= 1)
    assert recorder.stop(timeout_s=1.0)

    status = recorder.status()
    assert status.dropped_observations >= 1
    assert status.missing_frames >= 1
    assert status.incomplete is True
    assert any(record["missing_frames"] == ["wrist"] for record in recorder.iter_records())
    producer.close()


def test_recorder_storage_limit_fails_without_blocking_producer(tmp_path) -> None:
    source = Source()
    producer = ObservationProducer(
        camera_sources={"front": source},
        clock_ns=lambda: 200,
        service_instance_id="storage-service",
    )
    recorder = ObservationRecorder(
        producer.store,
        tmp_path,
        "full",
        max_bytes=1,
    )
    recorder.start()
    producer.publish_once()
    wait_until(lambda: recorder.status().state == "failed")

    status = recorder.status()
    assert status.incomplete is True
    assert "max_bytes" in (status.error or "")
    assert recorder.stop(timeout_s=1.0)
    producer.publish_once()
    assert source.calls == 2
    producer.close()


def test_recorder_rejects_path_escape_and_declares_lerobot_unsupported(
    tmp_path,
) -> None:
    producer = ObservationProducer(
        camera_sources={"front": Source()},
        clock_ns=lambda: 200,
        service_instance_id="path-service",
    )
    for recording_id in ("../escape", "/absolute", "a\\b", ""):
        with pytest.raises(ValueError):
            ObservationRecorder(producer.store, tmp_path, recording_id)

    recorder = ObservationRecorder(producer.store, tmp_path, "raw")
    with pytest.raises(NotImplementedError, match="LeRobot conversion is unsupported"):
        recorder.export_lerobot(tmp_path / "lerobot")

    root_file = tmp_path / "not-a-directory"
    root_file.write_text("occupied")
    broken = ObservationRecorder(producer.store, root_file, "broken")
    with pytest.raises(RecordingError, match="recording directory is unavailable"):
        broken.start()
    assert broken.status().state == "failed"

    missing_mount = ObservationRecorder(
        producer.store,
        tmp_path / "fallback-data",
        "missing-mount",
        expected_mount=tmp_path / "unmounted-data",
    )
    with pytest.raises(RecordingError, match="expected recording mount is unavailable"):
        missing_mount.start()
    assert missing_mount.status().state == "failed"
    producer.close()


def test_recorder_storage_write_error_is_failed_and_incomplete(tmp_path) -> None:
    source = Source()
    producer = ObservationProducer(
        camera_sources={"front": source},
        clock_ns=lambda: 200,
        service_instance_id="enospc-service",
    )
    recorder = ObservationRecorder(producer.store, tmp_path, "enospc")

    def no_space(path, data) -> None:
        del path, data
        raise OSError(errno.ENOSPC, "No space left on device")

    recorder._write_media = no_space
    recorder.start()
    producer.publish_once()
    wait_until(lambda: recorder.status().state == "failed")
    assert recorder.status().incomplete is True
    assert "No space left on device" in (recorder.status().error or "")
    assert recorder.stop(timeout_s=1.0)
    producer.close()


def test_recorder_stop_is_bounded_and_does_not_close_store(tmp_path) -> None:
    source = Source()
    producer = ObservationProducer(
        camera_sources={"front": source},
        clock_ns=lambda: 200,
        service_instance_id="stop-service",
    )
    recorder = ObservationRecorder(producer.store, tmp_path, "bounded-stop")
    entered = threading.Event()
    release = threading.Event()

    def blocked_write(snapshot) -> None:
        del snapshot
        entered.set()
        release.wait()

    recorder._write_snapshot = blocked_write
    recorder.start()
    producer.publish_once()
    assert entered.wait(1.0)
    assert recorder.stop(timeout_s=0.01) is False
    assert recorder.status().state == "stopping"
    release.set()
    assert recorder.stop(timeout_s=1.0)
    assert recorder.status().state == "stopped"
    producer.publish_once()
    producer.close()
