import base64
import copy
import hashlib
import io
import json
import threading
import time
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import pytest

from embodirun.robots.sensors.cameras import RawCameraFrame
from embodirun.services.rollout.camera_capture import (
    ObservationStore,
    make_handler,
    observation_packet,
    video_device,
)
from embodirun.services.rollout.camera_source import CameraObservationSource


def test_capture_rejects_non_video_device_before_opening():
    with pytest.raises(ValueError, match="/dev/video"):
        video_device("/dev/null")


def test_camera_packets_preserve_images_and_label_fixture_state():
    frame = SimpleNamespace(name="front", mime_type="image/jpeg", data=b"jpeg bytes")
    packet = observation_packet(
        [frame],
        index=3,
        state=[0] * 6,
        instruction="observe",
        started=10.0,
        finished=10.1,
    )
    assert base64.b64decode(packet["images"]["front"]["base64"]) == frame.data
    assert packet["state_source"] == "fixed-fixture-no-robot-read"
    assert packet["capture_finished_monotonic_s"] == 10.1


def test_camera_server_has_no_action_api_and_fails_when_camera_stops():
    store = ObservationStore()
    store.latest = {"index": 1, "images": {}}
    store.recorded = [store.latest]
    store.status = {"state": "running"}
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store, {"127.0.0.1"}))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    address = f"http://127.0.0.1:{server.server_port}"
    urlopen = build_opener(ProxyHandler({})).open
    try:
        with urlopen(address + "/latest", timeout=2) as response:
            assert json.load(response)["index"] == 1
        with pytest.raises(HTTPError) as error:
            urlopen(Request(address + "/action", data=b"{}"), timeout=2)
        assert error.value.code == 501
        with store.lock:
            store.status["state"] = "stopped"
        with pytest.raises(HTTPError) as error:
            urlopen(address + "/latest", timeout=2)
        assert error.value.code == 503
        with urlopen(address + "/frame/5", timeout=2) as response:
            assert json.load(response)["index"] == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_recorded_source_detects_changed_inputs_and_live_source_checks_freshness():
    Image = pytest.importorskip("PIL.Image")

    image = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(image, format="JPEG")
    frame = SimpleNamespace(name="front", mime_type="image/jpeg", data=image.getvalue())
    packet = observation_packet(
        [frame],
        index=0,
        state=[0] * 6,
        instruction="observe",
        started=1.0,
        finished=2.0,
    )
    store = ObservationStore()
    store.recorded = [packet]
    store.latest = packet
    store.status = {
        "state": "running",
        "motor_access": False,
        "recorded": 1,
        "record_count_target": 1,
    }
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store, {"127.0.0.1"}))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    address = f"http://127.0.0.1:{server.server_port}"
    try:
        source = CameraObservationSource([address], image_size=8)
        assert len(source.observation(10)["images"]["front"]) == 8 * 8 * 3
        assert source.fingerprint() != "live-inputs-not-identical"
        live = CameraObservationSource([address], image_size=8, mode="live")
        assert live.fingerprint() == "live-inputs-not-identical"
        with pytest.raises(ValueError, match="stale"):
            live.observation(0)
        with store.lock:
            # Freshly returned bytes can still originate from an old pipeline frame.
            store.latest = dict(
                packet,
                capture_finished_monotonic_s=time.monotonic(),
                acquisition={
                    "backend": "gstreamer-cpu",
                    "frames": {"front": {"sample_time_monotonic_estimate_s": time.monotonic() - 5}},
                },
            )
        with pytest.raises(ValueError, match="stale pipeline timestamp"):
            live.observation(0)
        with store.lock:
            store.recorded[0] = dict(packet, state=[1] * 6)
        with pytest.raises(ValueError, match="changed"):
            source.observation(0)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_raw_and_png_decode_to_identical_pixels_and_reject_bad_metadata():
    Image = pytest.importorskip("PIL.Image")
    # All three channels and both spatial axes differ, catching BGR/stride errors.
    rgb = bytes((index * 37) % 256 for index in range(7 * 5 * 3))
    pixels = Image.frombytes("RGB", (7, 5), rgb)
    raw = RawCameraFrame("front", 7, 5, pixels.tobytes("raw", "BGR"))
    common = dict(
        index=0,
        state=[0] * 6,
        instruction="observe",
        started=1.0,
        finished=2.0,
        binary=True,
    )
    packet = observation_packet([raw], **common)
    encoded = io.BytesIO()
    pixels.save(encoded, format="PNG")
    png = observation_packet(
        [SimpleNamespace(name="front", mime_type="image/png", data=encoded.getvalue())],
        **common,
    )
    source = object.__new__(CameraObservationSource)
    source.state_dim, source.image_size = 6, 11
    assert source._decode(packet) == source._decode(png)
    assert source._decode(packet)["images"]["front"] == pixels.resize((11, 11)).tobytes()
    for change in (
        {"width": 8},
        {"height": True},
        {"pixel_format": "nv12"},
        {"row_stride_bytes": 24},
    ):
        damaged = copy.deepcopy(packet)
        damaged["images"]["front"].update(change)
        with pytest.raises(ValueError, match="raw camera image layout"):
            source._decode(damaged)
    damaged = copy.deepcopy(packet)
    damaged["images"]["front"].update(data=b"short", sha256=hashlib.sha256(b"short").hexdigest())
    with pytest.raises(ValueError, match="raw camera image layout"):
        source._decode(damaged)


@pytest.mark.parametrize("backend", ["opencv-v4l2", "gstreamer-cpu"])
def test_raw_capture_cli_records_raw_files_and_cleans_up_without_robot(tmp_path, monkeypatch, backend):
    from embodirun.robots.sensors.cameras import gstreamer
    from embodirun.robots.sensors.cameras.v4l2 import camera
    from embodirun.services.rollout import camera_capture
    from embodirun.services.rollout.camera_shm import SharedCameraStore

    raw = RawCameraFrame("front", 4, 4, bytes(range(48)))
    closed = []

    class FakeSource:
        def __init__(self, _configs, **_settings):
            assert _configs[0].input_format == "yuy2"

        def capture_raw(self):
            return (raw,)

        def capture(self):
            raise AssertionError("raw CLI must not request JPEG capture")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(camera, "V4L2CameraSource", FakeSource)
    monkeypatch.setattr(gstreamer, "GStreamerCameraSource", FakeSource)
    monkeypatch.setattr(camera_capture, "video_device", lambda value: value)
    monkeypatch.setattr(camera_capture.signal, "signal", lambda *_args: None)
    args = SimpleNamespace(
        camera=[("front", "/dev/video0")],
        output=tmp_path / "recording",
        shm_path=tmp_path / "shared",
        frame_format="raw-bgr8",
        capture_backend=backend,
        input_format="yuy2",
        measure_stages=True,
        width=4,
        height=4,
        fps=100,
        record_count=1,
        record_hz=100,
        duration_s=0.05,
        bind="127.0.0.1",
        port=0,
        allow_host=["127.0.0.1"],
        state=[0] * 6,
        instruction="observe",
    )
    camera_capture.run(args)
    assert closed == [True]
    assert (args.output / "00000-front.bgr8").read_bytes() == raw.data
    metadata = json.loads((args.output / "process.json").read_text())
    assert metadata["motor_access"] is False
    assert metadata["frame_format"] == "raw-bgr8"
    assert metadata["acquisition_backend"] == backend
    stages = [json.loads(row) for row in (args.output / "capture-stages.jsonl").read_text().splitlines()]
    assert stages and stages[0]["image_payload_bytes"] == 48
    assert stages[0]["backend"] == backend
    assert stages[0]["capture_s"] >= 0 and stages[0]["shm_publish_s"] >= 0
    row = json.loads((args.output / "observations.jsonl").read_text())
    assert row["image_metadata"]["front"]["row_stride_bytes"] == 12
    reader = SharedCameraStore(args.shm_path)
    try:
        assert reader.get("/health")["state"] == "stopped"
        assert reader.get("/frame/0")["images"]["front"]["data"] == raw.data
    finally:
        reader.close()

    args.output, args.shm_path = tmp_path / "failed", tmp_path / "failed-shared"

    def fail_bind(*_args):
        raise OSError("bind failed")

    monkeypatch.setattr(camera_capture, "ThreadingHTTPServer", fail_bind)
    with pytest.raises(OSError, match="bind failed"):
        camera_capture.run(args)
    reader = SharedCameraStore(args.shm_path)
    try:
        assert reader.get("/health")["state"] == "stopped"
    finally:
        reader.close()
