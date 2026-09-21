import hashlib
import io
import multiprocessing
import threading
import time
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from embodirun.robots.sensors.cameras import RawCameraFrame
from embodirun.services.rollout.camera_capture import (
    ObservationStore,
    make_handler,
    observation_packet,
)
from embodirun.services.rollout.camera_shm import SharedCameraStore
from embodirun.services.rollout.camera_source import CameraObservationSource


def packet(index=0, *, image=None):
    if image is None:
        image = bytes([index % 256]) * 65536
    return observation_packet(
        [SimpleNamespace(name="front", mime_type="image/jpeg", data=image)],
        index=index,
        state=[0] * 6,
        instruction="observe",
        started=time.monotonic(),
        finished=time.monotonic(),
        binary=True,
    )


def test_readers_reject_devices_and_do_not_own_lifetime(tmp_path):
    with pytest.raises(ValueError, match="regular file"):
        SharedCameraStore("/dev/null")
    path = tmp_path / "camera"
    writer = SharedCameraStore(path, create=True, record_capacity=1, slot_bytes=100000)
    writer.publish(packet(), {"state": "running", "motor_access": False}, record=True)
    reader = SharedCameraStore(path)
    original = reader.get("/frame/0")
    with pytest.raises(RuntimeError, match="publishing"):
        reader.publish(packet(), {})
    with pytest.raises(KeyError):
        reader.get("/action")
    reader.close()
    assert path.exists()
    writer.close()
    writer.close()
    other = SharedCameraStore(path)
    assert other.get("/health")["state"] == "stopped"
    assert other.get("/frame/10") == original
    with pytest.raises(LookupError, match="not running"):
        other.get("/latest")
    other.close()


def test_overflow_preserves_previous_snapshot(tmp_path):
    path = tmp_path / "camera"
    writer = SharedCameraStore(path, create=True, record_capacity=1, slot_bytes=100000)
    writer.publish(packet(), {"state": "running", "motor_access": False}, record=True)
    try:
        with pytest.raises(ValueError, match="capacity exhausted"):
            writer.publish(packet(1), {"state": "running"}, record=True)
        assert writer.get("/latest")["index"] == 0
        with pytest.raises(ValueError, match="header capacity"):
            writer.publish(packet(3), {"state": "running", "oversized": "x" * 8192})
        assert writer.get("/latest")["index"] == 0
        assert writer.get("/health")["recorded"] == 1
        with pytest.raises(ValueError, match="slot capacity"):
            writer.publish(packet(2, image=b"x" * 100000), {"state": "running"})
        assert writer.get("/latest")["index"] == 0
    finally:
        writer.close()


@pytest.mark.parametrize("encoding", ["jpeg", "bgr8", "rgb8"])
def test_http_and_shm_decode_identical_recorded_generations(tmp_path, encoding):
    Image = pytest.importorskip("PIL.Image")
    path = tmp_path / "camera"
    writer = SharedCameraStore(path, create=True, record_capacity=2, slot_bytes=100000)
    http = ObservationStore()
    http.status = {
        "state": "running",
        "motor_access": False,
        "recorded": 2,
        "record_count_target": 2,
    }
    for index, color in enumerate(["red", "green"]):
        image = io.BytesIO()
        pixels = Image.new("RGB", (16, 16), color)
        if encoding == "jpeg":
            pixels.save(image, format="JPEG")
            p = packet(index, image=image.getvalue())
        else:
            raw = pixels.tobytes("raw", "BGR" if encoding == "bgr8" else "RGB")
            p = observation_packet(
                [RawCameraFrame("front", 16, 16, raw, pixel_format=encoding)],
                index=index,
                state=[0] * 6,
                instruction="observe",
                started=time.monotonic(),
                finished=time.monotonic(),
                binary=True,
            )
        writer.publish(p, http.status, record=True)
        http.recorded.append(p)
    http.latest = http.recorded[-1]
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(http, {"127.0.0.1"}))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    a = b = None
    try:
        a = CameraObservationSource([f"http://127.0.0.1:{server.server_port}"])
        b = CameraObservationSource(["shm://" + str(path)])
        assert a.fingerprint() == b.fingerprint()
        for index in range(6):
            assert a.observation(index) == b.observation(index)
            assert a.last_metadata["frame_index"] == b.last_metadata["frame_index"]
            assert a.last_metadata["image_payload_bytes"] == b.last_metadata["image_payload_bytes"]
            if encoding != "jpeg":
                expected = Image.new("RGB", (224, 224), ["red", "green"][index % 2]).tobytes()
                assert b.observation(index)["images"]["front"] == expected
                assert b.last_metadata["image_payload_bytes"] == 16 * 16 * 3
    finally:
        if a:
            a.close()
        if b:
            b.close()
        writer.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def read_generations(path, output):
    reader = SharedCameraStore(path)
    try:
        seen = []
        for _ in range(100):
            p = reader.get("/latest")
            frame = p["images"]["front"]
            data = frame["data"]
            assert data == bytes([p["index"] % 256]) * len(data)
            assert hashlib.sha256(data).hexdigest() == frame["sha256"]
            seen.append(p["index"])
        output.put({"count": len(seen), "valid": True})
    finally:
        reader.close()


def test_cross_process_readers_never_observe_torn_frames(tmp_path):
    path = tmp_path / "camera"
    writer = SharedCameraStore(path, create=True, record_capacity=1, slot_bytes=100000)
    writer.publish(packet(), {"state": "running", "motor_access": False})
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    process = context.Process(target=read_generations, args=(path, output))
    process.start()
    try:
        for index in range(1, 300):
            writer.publish(packet(index), {"state": "running", "motor_access": False})
        result = output.get(timeout=15)
        process.join(timeout=5)
        assert process.exitcode == 0
        assert result == {"count": 100, "valid": True}
        assert path.exists()
        assert writer.get("/latest")["index"] == 299
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        writer.close()
        output.close()
