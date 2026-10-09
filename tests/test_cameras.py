from types import SimpleNamespace

import pytest

from embodirun.robots.sensors.cameras import (
    RealSenseCameraConfig,
    RealSenseCameraSource,
    V4L2CameraConfig,
    V4L2CameraSource,
)


def test_viewer_reader_stops_with_an_open_or_partial_input_pipe() -> None:
    import os
    from queue import Queue
    from threading import Event, Thread

    from embodirun.simulators.viewer import _read_frames

    for partial in (b"", b"\x00\x00", b"\x00\x00\x00\x04xy"):
        read_fd, write_fd = os.pipe()
        with (
            os.fdopen(read_fd, "rb") as stream,
            os.fdopen(write_fd, "wb", buffering=0) as writer,
        ):
            frames = Queue(maxsize=1)
            frames.put(b"previous-frame")
            stopped = Event()
            reader = Thread(target=_read_frames, args=(stream, frames, stopped))
            reader.start()
            writer.write(partial)
            stopped.set()
            reader.join(timeout=2)
            assert not reader.is_alive()
            assert frames.get_nowait() is None


class FakeCapture:
    def __init__(self) -> None:
        self.settings = []
        self.grabs = 0
        self.released = False

    def isOpened(self):
        return True

    def set(self, name, value):
        self.settings.append((name, value))

    def grab(self):
        self.grabs += 1
        return True

    def retrieve(self):
        return True, SimpleNamespace(shape=(480, 640, 3))

    def release(self):
        self.released = True


class FakeCv2:
    CAP_V4L2 = 1
    CAP_PROP_FRAME_WIDTH = 2
    CAP_PROP_FRAME_HEIGHT = 3
    CAP_PROP_FPS = 4
    CAP_PROP_BUFFERSIZE = 5
    IMWRITE_JPEG_QUALITY = 6

    def __init__(self) -> None:
        self.capture = FakeCapture()

    def VideoCapture(self, device, backend):
        assert device == "/dev/video0"
        assert backend == self.CAP_V4L2
        return self.capture

    def imencode(self, extension, frame, options):
        assert extension == ".jpg"
        assert frame.shape == (480, 640, 3)
        assert options == [self.IMWRITE_JPEG_QUALITY, 90]
        return True, SimpleNamespace(tobytes=lambda: b"jpeg")


class ProfiledCapture(FakeCapture):
    def get(self, property_id):
        return {
            FakeCv2.CAP_PROP_FRAME_WIDTH: 640.0,
            FakeCv2.CAP_PROP_FRAME_HEIGHT: 480.0,
            # A zero means that this backend did not report an actual FPS.
            FakeCv2.CAP_PROP_FPS: 0.0,
        }[property_id]


class ProfiledCv2(FakeCv2):
    def __init__(self) -> None:
        self.capture = ProfiledCapture()


@pytest.mark.parametrize(
    ("configured", "expected"),
    [(None, 2), ("4", 4), ("1", 1), ("0", 2), ("not-a-number", 2)],
)
def test_v4l2_camera_requests_two_buffers_by_default(monkeypatch, configured, expected) -> None:
    """One V4L2 buffer drops every other frame; the default must be two."""
    if configured is None:
        monkeypatch.delenv("EMBODIRUN_CAMERA_BUFFERSIZE", raising=False)
    else:
        monkeypatch.setenv("EMBODIRUN_CAMERA_BUFFERSIZE", configured)
    cv2 = FakeCv2()

    source = V4L2CameraSource(
        (
            V4L2CameraConfig(
                name="observation.images.front",
                device="/dev/video0",
                width=640,
                height=480,
                fps=30.0,
            ),
        ),
        cv2_module=cv2,
        clock_ns=lambda: 1,
    )
    source.capture()
    source.close()

    requested = [value for name, value in cv2.capture.settings if name == FakeCv2.CAP_PROP_BUFFERSIZE]
    assert requested == [expected]


def test_v4l2_camera_produces_neutral_frame_and_releases_device() -> None:
    cv2 = FakeCv2()
    clock_value = 0

    def clock_ns() -> int:
        nonlocal clock_value
        clock_value += 1
        return clock_value

    source = V4L2CameraSource(
        (
            V4L2CameraConfig(
                name="observation.images.front",
                device="/dev/video0",
                width=640,
                height=480,
                fps=30.0,
            ),
        ),
        cv2_module=cv2,
        clock_ns=clock_ns,
    )

    frames = source.capture()
    source.close()

    assert len(frames) == 1
    assert frames[0].name == "observation.images.front"
    assert frames[0].mime_type == "image/jpeg"
    assert frames[0].data == b"jpeg"
    assert frames[0].captured_timestamp_ns is not None
    assert frames[0].received_timestamp_ns is not None
    assert frames[0].captured_timestamp_ns < frames[0].received_timestamp_ns
    assert frames[0].clock_domain == "host_monotonic_ns"
    assert frames[0].profile is None
    assert cv2.capture.grabs == 4
    assert cv2.capture.released is True


def test_v4l2_profile_omits_fps_when_driver_does_not_report_it() -> None:
    cv2 = ProfiledCv2()
    source = V4L2CameraSource(
        (
            V4L2CameraConfig(
                name="front",
                device="/dev/video0",
                width=640,
                height=480,
                fps=30.0,
            ),
        ),
        cv2_module=cv2,
        clock_ns=lambda: 10,
    )
    frame = source.capture()[0]
    source.close()
    assert frame.profile == {"width": 640, "height": 480}
    assert "fps" not in frame.profile


class FakeVideoProfile:
    def width(self):
        return 640

    def height(self):
        return 480

    def fps(self):
        return 30


class FakePipelineProfile:
    def get_stream(self, stream):
        assert stream == FakeRs.stream.color
        return self

    def as_video_stream_profile(self):
        return FakeVideoProfile()


class FakeColorFrame:
    def get_data(self):
        return b"rgb"


class FakeFrames:
    def get_color_frame(self):
        return FakeColorFrame()


class FakePipeline:
    def start(self, config):
        return FakePipelineProfile()

    def wait_for_frames(self, timeout_ms):
        assert timeout_ms == 2000
        return FakeFrames()

    def stop(self):
        pass


class FakeConfig:
    def enable_device(self, serial):
        assert serial == "serial-1"

    def enable_stream(self, *args):
        assert args == (FakeRs.stream.color, 640, 480, FakeRs.format.rgb8, 30)


class FakeRs:
    class stream:
        color = object()

    class format:
        rgb8 = object()

    def pipeline(self):
        return FakePipeline()

    def config(self):
        return FakeConfig()


def test_realsense_timestamps_host_read_before_encoding_and_reports_profile() -> None:
    clock_value = 0

    def clock_ns() -> int:
        nonlocal clock_value
        clock_value += 1
        return clock_value

    encoder_calls = []

    def encoder(raw, width, height, quality):
        encoder_calls.append((raw, width, height, quality))
        return b"jpeg"

    source = RealSenseCameraSource(
        (RealSenseCameraConfig("front", "serial-1", width=640, height=480),),
        rs_module=FakeRs(),
        jpeg_encoder=encoder,
        clock_ns=clock_ns,
    )
    frame = source.capture()[0]
    source.close()
    assert encoder_calls == [(b"rgb", 640, 480, 90)]
    assert frame.data == b"jpeg"
    assert frame.captured_timestamp_ns == 1
    assert frame.received_timestamp_ns == 2
    assert frame.clock_domain == "host_monotonic_ns"
    assert frame.profile == {"width": 640, "height": 480, "fps": 30.0}
