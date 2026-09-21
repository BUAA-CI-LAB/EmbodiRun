"""Optional camera-only GStreamer acquisition with bounded, owned raw frames.

PyGObject and the platform's GStreamer plugins are loaded only on construction.
Jetson conversion is explicit and never silently replaced with CPU conversion.
"""

from __future__ import annotations

import math
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from .camera import RawCameraFrame
from .v4l2.camera import CameraError, V4L2CameraConfig


def load_gstreamer():
    """Load the platform ABI rather than installing a second multimedia stack."""
    try:
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstVideo", "1.0")
        from gi.repository import Gst, GstVideo
    except (ImportError, ValueError) as error:
        raise CameraError(
            "GStreamer capture requires matching PyGObject, Gst and GstVideo "
            "typelibs in this Python environment; see transport-experiments.md"
        ) from error
    Gst.init(None)
    if Gst.version()[:2] < (1, 20):
        raise CameraError("GStreamer camera capture requires version 1.20 or newer")
    return Gst, GstVideo


@dataclass(frozen=True, slots=True)
class GStreamerCameraConfig:
    name: str
    device: str
    width: int
    height: int
    fps: float
    input_format: str = "mjpeg"
    conversion: str = "cpu"
    output_width: int | None = None
    output_height: int | None = None

    def __post_init__(self):
        V4L2CameraConfig(self.name, self.device, self.width, self.height, self.fps)
        if not re.fullmatch(r"/dev/video[0-9]+", self.device):
            raise ValueError("GStreamer camera device must be a resolved /dev/videoN")
        if self.input_format not in {"mjpeg", "yuy2"}:
            raise ValueError("GStreamer input format must be mjpeg or yuy2")
        if self.conversion not in {"cpu", "jetson"}:
            raise ValueError("GStreamer conversion must be cpu or jetson")
        if (self.output_width is None) != (self.output_height is None):
            raise ValueError("set both GStreamer output dimensions or neither")
        for value in (self.width, self.height, *self.output_size):
            if type(value) is not int or not 0 < value <= 16384:
                raise ValueError("GStreamer dimensions must be integers in 1..16384")

    @property
    def output_size(self):
        return (
            self.width if self.output_width is None else self.output_width,
            self.height if self.output_height is None else self.output_height,
        )

    def pipeline(self):
        """Build only the declared camera path; no arbitrary user pipeline text."""
        rate = Fraction(str(self.fps)).limit_denominator(1001)
        width, height = self.output_size
        caps = "image/jpeg" if self.input_format == "mjpeg" else "video/x-raw,format=YUY2"
        parts = [
            f"v4l2src name=camera device={self.device} do-timestamp=true",
            f"{caps},width={self.width},height={self.height},framerate={rate.numerator}/{rate.denominator}",
            "queue name=pending max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream",
        ]
        required = ["v4l2src", "queue", "videoconvert", "appsink"]
        if self.input_format == "mjpeg":
            if self.conversion == "jetson":
                parts += ["jpegparse", "nvv4l2decoder mjpeg=true"]
                required += ["jpegparse", "nvv4l2decoder"]
            else:
                parts += ["jpegdec"]
                required += ["jpegdec"]
        if self.conversion == "jetson":
            parts += [
                "nvvidconv",
                f"video/x-raw,format=BGRx,width={width},height={height},pixel-aspect-ratio=1/1",
                "videoconvert",
            ]
            required += ["nvvidconv"]
        else:
            parts += ["videoconvert", "videoscale method=1 add-borders=false"]
            required += ["videoscale"]
        parts += [
            f"video/x-raw,format=BGR,width={width},height={height},pixel-aspect-ratio=1/1",
            "appsink name=frames max-buffers=1 drop=true sync=false emit-signals=false wait-on-eos=false enable-last-sample=false",
        ]
        return " ! ".join(parts), required


class GStreamerFrameStream:
    """Own one already constructed pipeline and copy samples before releasing them.

    This low-level reader also accepts an appsrc pipeline for file-only tests.
    Camera creation belongs to GStreamerCameraSource. Reads are serialized.
    """

    def __init__(
        self,
        pipeline,
        name,
        width,
        height,
        *,
        timeout_s=2,
        max_age_s=1,
        measure_stages=False,
    ):
        if any(not math.isfinite(x) or x <= 0 for x in (timeout_s, max_age_s)):
            raise ValueError("GStreamer timeout and maximum frame age must be positive")
        self.Gst, self.GstVideo = load_gstreamer()
        self.pipeline, self.name = pipeline, name
        self.width, self.height = width, height
        self.timeout_s, self.max_age_s = timeout_s, max_age_s
        self.sink = pipeline.get_by_name("frames")
        if self.sink is None or self.sink.get_factory().get_name() != "appsink":
            raise CameraError("GStreamer pipeline requires an appsink named frames")
        self.sink.set_property("max-buffers", 1)
        self.sink.set_property("drop", True)  # Also supported by Jetson Gst 1.20.
        self.sink.set_property("wait-on-eos", False)
        self.sink.set_property("enable-last-sample", False)
        self.bus = pipeline.get_bus()
        self.closed = False
        self.last_pts = None
        self.last_metrics = {}
        self.probes = []
        self.timings = OrderedDict()
        self.timing_lock = threading.Lock()
        self.probe_counts = {"source": 0, "sink": 0, "evicted": 0}
        if measure_stages:
            camera = pipeline.get_by_name("camera")
            if camera is None:
                raise CameraError("stage measurement requires a source named camera")
            for pad, label in (
                (camera.get_static_pad("src"), "source"),
                (self.sink.get_static_pad("sink"), "sink"),
            ):
                identifier = pad.add_probe(self.Gst.PadProbeType.BUFFER, self._probe, label)
                self.probes.append((pad, identifier))
        if pipeline.set_state(self.Gst.State.PLAYING) == self.Gst.StateChangeReturn.FAILURE:
            self.close()
            raise CameraError("GStreamer pipeline failed to enter PLAYING")

    def _probe(self, _pad, probe, label):
        buffer = probe.get_buffer()
        if buffer is not None and buffer.pts != self.Gst.CLOCK_TIME_NONE:
            with self.timing_lock:
                self.probe_counts[label] += 1
                self.timings.setdefault(buffer.pts, {})[label] = time.monotonic()
                while len(self.timings) > 64:
                    self.timings.popitem(last=False)
                    self.probe_counts["evicted"] += 1
        return self.Gst.PadProbeReturn.OK

    def _error(self):
        message = self.bus.pop_filtered(self.Gst.MessageType.ERROR)
        if message is not None:
            error, debug = message.parse_error()
            raise CameraError(f"GStreamer {self.name}: {error}; {debug}")

    def read(self):
        if self.closed:
            raise CameraError("GStreamer frame stream is closed")
        try:
            return self._read()
        except BaseException:
            self.close()
            raise

    def _read(self):
        self._error()
        before = time.monotonic()
        sample = self.sink.emit("try-pull-sample", int(self.timeout_s * self.Gst.SECOND))
        pulled = time.monotonic()
        self._error()
        if sample is None:
            state = "end of stream" if self.sink.get_property("eos") else "sample timeout"
            raise CameraError(f"GStreamer {self.name}: {state}")
        buffer, caps = sample.get_buffer(), sample.get_caps()
        info = self.GstVideo.VideoInfo.new_from_caps(caps)
        if (
            info is None
            or info.finfo.format != self.GstVideo.VideoFormat.BGR
            or (info.width, info.height) != (self.width, self.height)
        ):
            raise CameraError("GStreamer sample differs from configured BGR dimensions")
        if buffer.pts == self.Gst.CLOCK_TIME_NONE:
            raise CameraError("GStreamer sample has no presentation timestamp")
        if self.last_pts is not None and buffer.pts <= self.last_pts:
            raise CameraError("GStreamer sample timestamp repeated or regressed")
        running = sample.get_segment().to_running_time(self.Gst.Format.TIME, buffer.pts)
        clock = self.pipeline.get_clock()
        if running == self.Gst.CLOCK_TIME_NONE or clock is None:
            raise CameraError("GStreamer sample has no valid running-time clock")
        age = (clock.get_time() - self.pipeline.get_base_time() - running) / self.Gst.SECOND
        if not 0 <= age <= self.max_age_s:
            raise CameraError(f"GStreamer sample is stale or in the future: age={age}")
        video_meta = self.GstVideo.buffer_get_video_meta(buffer)
        if video_meta is not None:
            if (
                video_meta.format != self.GstVideo.VideoFormat.BGR
                or video_meta.n_planes != 1
                or (video_meta.width, video_meta.height) != (self.width, self.height)
            ):
                raise CameraError("GStreamer video metadata differs from sample caps")
            stride, offset = video_meta.stride[0], video_meta.offset[0]
        else:
            stride, offset = info.stride[0], info.offset[0]
        ok, mapped = buffer.map(self.Gst.MapFlags.READ)
        if not ok:
            raise CameraError("GStreamer sample could not be mapped into CPU memory")
        try:
            data = memoryview(mapped.data)
            row_bytes = self.width * 3
            if stride < row_bytes or offset < 0 or offset + (self.height - 1) * stride + row_bytes > len(data):
                raise CameraError("GStreamer mapped sample has an invalid row layout")
            packed = b"".join(
                data[offset + row * stride : offset + row * stride + row_bytes] for row in range(self.height)
            )
            del data
        finally:
            buffer.unmap(mapped)
        copied = time.monotonic()
        self.last_pts = buffer.pts
        self.last_metrics = {
            "pull_wait_s": pulled - before,
            "sample_map_pack_s": copied - pulled,
            "sample_pts_ns": buffer.pts,
            "sample_running_time_ns": running,
            "sample_age_at_pull_s": age,
            "sample_time_monotonic_estimate_s": pulled - age,
            "timestamp_semantics": "pipeline presentation time; not sensor exposure time",
            "source_row_stride_bytes": stride,
            "source_plane_offset_bytes": offset,
            "packed_bytes": len(packed),
            "negotiated_caps": caps.to_string(),
        }
        if self.sink.find_property("dropped") is not None:
            self.last_metrics["appsink_dropped"] = self.sink.get_property("dropped")
        if self.probes:
            with self.timing_lock:
                timing = self.timings.pop(buffer.pts, {})
                counts = dict(self.probe_counts)
            self.last_metrics.update(
                native_pipeline_s=timing["sink"] - timing["source"] if set(timing) == {"source", "sink"} else None,
                native_pipeline_semantics="source pad to appsink pad; includes queue/decode/convert/resize and probe overhead",
                stage_probe_counts=counts,
            )
        return RawCameraFrame(self.name, self.width, self.height, packed)

    def close(self):
        if not self.closed:
            self.pipeline.set_state(self.Gst.State.NULL)
            for pad, identifier in self.probes:
                pad.remove_probe(identifier)
            self.probes.clear()
            self.closed = True


class GStreamerCameraSource:
    """Independent camera pipelines; no motor adapter and no implicit fallback."""

    def __init__(self, cameras, *, timeout_s=2, max_age_s=1, measure_stages=False):
        self.streams = []
        self.last_metrics = {}
        cameras = tuple(cameras)
        if not cameras or len({camera.name for camera in cameras}) != len(cameras):
            raise ValueError("GStreamer cameras must have unique names and not be empty")
        Gst, _ = load_gstreamer()
        plans = []
        for camera in cameras:
            resolved = Path(camera.device).resolve(strict=True)
            if str(resolved) != camera.device or not (Path("/sys/class/video4linux") / resolved.name).exists():
                raise CameraError("GStreamer capture requires a registered resolved V4L2 device")
            description, required = camera.pipeline()
            missing = [name for name in required if Gst.ElementFactory.find(name) is None]
            if missing:
                raise CameraError(f"missing GStreamer plugins: {', '.join(missing)}")
            plans.append((camera, description, required))
        self.metadata = {
            "gstreamer_version": Gst.version_string(),
            "pipelines": {camera.name: description for camera, description, _ in plans},
            "camera_format_negotiation": "explicit mjpeg/yuy2; no fallback",
        }
        try:
            for camera, description, _ in plans:
                pipeline = Gst.parse_launch(description)
                try:
                    stream = GStreamerFrameStream(
                        pipeline,
                        camera.name,
                        *camera.output_size,
                        timeout_s=timeout_s,
                        max_age_s=max_age_s,
                        measure_stages=measure_stages,
                    )
                except BaseException:
                    pipeline.set_state(Gst.State.NULL)
                    raise
                self.streams.append(stream)
            for _ in range(3):
                self.capture_raw()
        except BaseException:
            self.close()
            raise

    def capture_raw(self):
        try:
            frames = tuple(stream.read() for stream in self.streams)
            if not frames:
                raise CameraError("GStreamer camera source is closed")
            self.last_metrics = {stream.name: dict(stream.last_metrics) for stream in self.streams}
            return frames
        except BaseException:
            self.close()
            raise

    def close(self):
        for stream in self.streams:
            stream.close()
        self.streams.clear()
