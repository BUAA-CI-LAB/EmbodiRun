"""Pipeline tests use appsrc bytes only; no test opens a camera device."""

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodirun.robots.sensors.cameras.gstreamer import (
    GStreamerCameraConfig,
    GStreamerCameraSource,
    GStreamerFrameStream,
    load_gstreamer,
)
from embodirun.robots.sensors.cameras.v4l2.camera import CameraError


def test_camera_profiles_are_explicit_and_reject_non_video_paths():
    base = dict(name="front", device="/dev/video0", width=640, height=480, fps=30)
    cpu, required = GStreamerCameraConfig(**base, output_width=224, output_height=224).pipeline()
    assert "format=BGR,width=224,height=224" in cpu
    assert "add-borders=false" in cpu and "max-buffers=1" in cpu
    assert "jpegdec" in required and "nvv4l2decoder" not in required
    jetson, required = GStreamerCameraConfig(**base, conversion="jetson").pipeline()
    assert "nvv4l2decoder mjpeg=true" in jetson and "nvvidconv" in required
    assert "format=BGRx" in jetson and "jpegdec" not in required
    raw, required = GStreamerCameraConfig(**base, input_format="yuy2").pipeline()
    assert "format=YUY2" in raw and "jpegdec" not in required
    for changes in (
        {"device": "/dev/ttyUSB0"},
        {"device": "/dev/video0 ! fakesrc"},
        {"input_format": "h264"},
        {"conversion": "auto"},
        {"output_width": 10},
        {"output_width": 0, "output_height": 10},
    ):
        with pytest.raises(ValueError):
            GStreamerCameraConfig(**(base | changes))


def test_missing_plugins_fail_before_any_pipeline_is_created(monkeypatch):
    from embodirun.robots.sensors.cameras import gstreamer

    fake = SimpleNamespace(ElementFactory=SimpleNamespace(find=lambda _name: None))
    monkeypatch.setattr(gstreamer, "load_gstreamer", lambda: (fake, None))
    monkeypatch.setattr(Path, "resolve", lambda self, **_kwargs: self)
    monkeypatch.setattr(Path, "exists", lambda _self: True)
    camera = GStreamerCameraConfig("front", "/dev/video0", 640, 480, 30, conversion="jetson")
    with pytest.raises(CameraError, match="missing GStreamer plugins.*nvv4l2decoder"):
        GStreamerCameraSource([camera])  # fake has no parse_launch: cannot open anything.


@pytest.fixture
def gst():
    pytest.importorskip("gi")
    try:
        Gst, Video = load_gstreamer()
    except CameraError as error:
        pytest.skip(str(error))
    if any(Gst.ElementFactory.find(name) is None for name in ("appsrc", "appsink")):
        pytest.skip("GStreamer app plugin unavailable")
    return Gst, Video


def app_stream(gst, *, width=7, height=5, **settings):
    Gst, _ = gst
    pipeline = Gst.parse_launch(
        f'appsrc name=camera is-live=true format=time caps="video/x-raw,format=BGR,width={width},height={height},framerate=30/1" '
        "! appsink name=frames sync=false async=false"
    )
    stream = GStreamerFrameStream(pipeline, "front", width, height, **settings)
    return pipeline, stream


def push(gst, pipeline, data, *, pts=0, layout=None):
    Gst, Video = gst
    buffer = Gst.Buffer.new_allocate(None, len(data), None)
    buffer.fill(0, data)
    buffer.pts = pts
    if layout is not None:
        offset, stride, width, height = layout
        Video.buffer_add_video_meta_full(
            buffer,
            Video.VideoFrameFlags.NONE,
            Video.VideoFormat.BGR,
            width,
            height,
            1,
            [offset, 0, 0, 0],
            [stride, 0, 0, 0],
        )
    assert pipeline.get_by_name("camera").emit("push-buffer", buffer) == Gst.FlowReturn.OK
    return buffer


@pytest.mark.parametrize("layout", [None, (8, 28, 7, 5)])
def test_native_mapping_handles_padding_offsets_and_owned_bytes(gst, layout):
    pipeline, stream = app_stream(gst)
    offset, stride = (0, 24) if layout is None else layout[:2]
    data = bytes((i * 37) % 256 for i in range(offset + stride * 5))
    expected = b"".join(data[offset + row * stride : offset + row * stride + 21] for row in range(5))
    try:
        push(gst, pipeline, data, layout=layout)
        first = stream.read()
        assert first.data == expected
        assert stream.last_metrics["source_row_stride_bytes"] == stride
        assert stream.last_metrics["source_plane_offset_bytes"] == offset
        push(gst, pipeline, bytes(len(data)), pts=1, layout=layout)
        assert stream.read().data == bytes(105)
        assert first.data == expected
    finally:
        stream.close()
    assert pipeline.get_state(0).state == gst[0].State.NULL


def test_slow_consumer_gets_latest_frame_instead_of_an_unbounded_backlog(gst):
    pipeline, stream = app_stream(gst, measure_stages=True)
    complete, count = threading.Event(), []
    stream.sink.set_property("emit-signals", True)

    def arrived(_sink):
        count.append(1)
        if len(count) == 100:
            complete.set()
        return gst[0].FlowReturn.OK

    stream.sink.connect("new-sample", arrived)
    try:
        for index in range(100):
            push(gst, pipeline, bytes([index]) * 120, pts=index)
        assert complete.wait(2), "native pipeline did not consume the offered frames"
        assert stream.read().data == bytes([99]) * 105
        assert stream.last_metrics["native_pipeline_s"] >= 0
        assert stream.last_metrics["stage_probe_counts"]["source"] == 100
        assert len(stream.timings) <= 64
        if "appsink_dropped" in stream.last_metrics:
            assert stream.last_metrics["appsink_dropped"] == 99
    finally:
        stream.close()


def test_missing_sample_times_out_and_closes_the_pipeline(gst):
    pipeline, stream = app_stream(gst, timeout_s=0.03)
    with pytest.raises(CameraError, match="timeout"):
        stream.read()
    assert stream.closed and pipeline.get_state(0).state == gst[0].State.NULL
    with pytest.raises(CameraError, match="closed"):
        stream.read()


@pytest.mark.parametrize("failure", ["short", "duplicate", "stale", "future", "no-pts"])
def test_bad_layout_or_timestamp_is_rejected_and_pipeline_stops(gst, failure):
    pipeline, stream = app_stream(gst, max_age_s=0.02 if failure == "stale" else 1)
    try:
        pts = 0
        if failure in {"duplicate", "no-pts"}:
            # Appsrc assigns zero to an untimestamped first buffer. Exercise
            # missing PTS after the stream is already established instead.
            push(gst, pipeline, bytes(120))
            stream.read()
        if failure == "future":
            pts = 10 * gst[0].SECOND
        elif failure == "no-pts":
            pts = gst[0].CLOCK_TIME_NONE
        push(gst, pipeline, bytes(1 if failure == "short" else 120), pts=pts)
        if failure == "stale":
            time.sleep(0.04)
        with pytest.raises(CameraError):
            stream.read()
        assert stream.closed
    finally:
        stream.close()
