"""Read camera-only publishers without importing any robot driver."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import socket
import time
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener


def image_payload_bytes(packet):
    """Count image bytes, excluding HTTP base64 expansion and envelope metadata."""
    total = 0
    for image in packet["images"].values():
        if "data" in image:
            total += len(image["data"])
        else:
            encoded = image["base64"]
            padding = len(encoded) - len(encoded.rstrip("="))
            total += len(encoded) // 4 * 3 - padding
    return total


class CameraObservationSource:
    """Fetch recorded or live images while physical camera acquisition runs.

    Recorded mode checks every fetched observation against the initial recording.
    Live mode has no identical-input fingerprint and must not be used as such.
    Capture monotonic timestamps may only be compared on the camera's own host.
    """

    def __init__(self, urls, *, image_size=224, state_dim=6, mode="recorded", timeout_s=5):
        if mode not in {"recorded", "live"} or not urls:
            raise ValueError("camera source requires URLs and recorded/live mode")
        if image_size <= 0 or state_dim <= 0 or timeout_s <= 0:
            raise ValueError("camera source dimensions and timeout must be positive")
        self.urls = [url.rstrip("/") for url in urls]
        if any(not url.startswith(("http://", "shm:///")) for url in self.urls):
            raise ValueError("camera publishers require HTTP or absolute shm:/// URLs")
        self.mode, self.image_size, self.state_dim = mode, image_size, state_dim
        self.timeout_s = timeout_s
        self.opener = build_opener(ProxyHandler({}))
        self.local_clocks = []
        self.recorded = []
        self.last_metadata = {}
        self.shared_stores = {}
        try:
            for url in self.urls:
                parsed = urlsplit(url)
                if parsed.scheme == "shm":
                    from .camera_shm import SharedCameraStore

                    self.shared_stores[url] = SharedCameraStore(parsed.path)
                    self.local_clocks.append(True)
                else:
                    address = socket.gethostbyname(parsed.hostname)
                    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                        # UDP connect only selects a route; no packet is transmitted.
                        probe.connect((address, parsed.port or 80))
                        self.local_clocks.append(address == probe.getsockname()[0])
                health = self._get(url + "/health")
                if parsed.scheme == "shm" and health.get("clock_identity") != self.shared_stores[url].clock_identity():
                    raise ValueError("Shared camera timestamps belong to another host or time namespace")
                if health.get("state") != "running" or health.get("motor_access") is not False:
                    raise ValueError("camera-only publisher must be running")
                if mode == "recorded":
                    count = health.get("recorded", 0)
                    if not count or count != health.get("record_count_target"):
                        raise ValueError("camera recording must finish before paired trials")
                    self.recorded.append([self._decode(self._get(url + f"/frame/{index}")) for index in range(count)])
        except BaseException:
            self.close()
            raise

    def _get(self, url):
        for prefix, store in self.shared_stores.items():
            if url.startswith(prefix + "/"):
                return store.get(url[len(prefix) :])
        with self.opener.open(url, timeout=self.timeout_s) as response:
            return json.load(response)

    def close(self):
        for store in self.shared_stores.values():
            store.close()
        self.shared_stores.clear()

    def _decode(self, packet):
        from PIL import Image

        if packet.get("schema") != "embodirun.camera-only.v1":
            raise ValueError("unexpected camera observation schema")
        if packet.get("state_source") != "fixed-fixture-no-robot-read":
            raise ValueError("camera experiment requires explicitly labelled fixture state")
        state = packet["state"]
        if len(state) != self.state_dim or not all(math.isfinite(x) for x in state):
            raise ValueError("invalid camera fixture state")
        images = {}
        for name, value in packet["images"].items():
            data = value.get("data")
            if data is None:
                data = base64.b64decode(value["base64"], validate=True)
            if not isinstance(data, bytes):
                raise TypeError("camera image payload must contain bytes")
            if hashlib.sha256(data).hexdigest() != value["sha256"]:
                raise ValueError("camera image checksum mismatch")
            if value["mime_type"] == "application/x-embodirun-raw-image":
                width, height = value.get("width"), value.get("height")
                if (
                    any(type(n) is not int or not 0 < n <= 16384 for n in (width, height))
                    or value.get("pixel_format") not in {"bgr8", "rgb8"}
                    or value.get("row_stride_bytes") != width * 3
                    or len(data) != width * height * 3
                ):
                    raise ValueError("invalid packed raw camera image layout")
                image = Image.frombytes(
                    "RGB",
                    (width, height),
                    data,
                    "raw",
                    "BGR" if value["pixel_format"] == "bgr8" else "RGB",
                )
            elif value["mime_type"] in {"image/jpeg", "image/png"}:
                image = Image.open(io.BytesIO(data))
            else:
                raise ValueError("unsupported camera image MIME type")
            with image:
                images[name] = image.convert("RGB").resize((self.image_size, self.image_size)).tobytes()
        if not images:
            raise ValueError("camera packet has no images")
        return {"state": state, "instruction": packet["instruction"], "images": images}

    def observation(self, index):
        source = index % len(self.urls)
        frame = index // len(self.urls)
        suffix = f"/frame/{frame}" if self.mode == "recorded" else "/latest"
        started = time.perf_counter()
        packet = self._get(self.urls[source] + suffix)
        value = self._decode(packet)
        capture_age = None
        pipeline_age = None
        acquisition = packet.get("acquisition", {})
        if self.local_clocks[source]:
            capture_age = time.monotonic() - packet["capture_finished_monotonic_s"]
            if self.mode == "live" and not 0 <= capture_age <= 1.0:
                raise ValueError("live camera frame is stale or its clock is invalid")
            if acquisition.get("backend", "").startswith("gstreamer-"):
                frame_times = acquisition.get("frames", {})
                if set(frame_times) != set(packet["images"]):
                    raise ValueError("GStreamer timestamps do not cover every camera")
                timestamps = [frame["sample_time_monotonic_estimate_s"] for frame in frame_times.values()]
                if not all(isinstance(t, (int, float)) and math.isfinite(t) for t in timestamps):
                    raise ValueError("invalid GStreamer pipeline timestamp")
                pipeline_age = time.monotonic() - min(timestamps)
                if self.mode == "live" and not 0 <= pipeline_age <= 1.0:
                    raise ValueError("live GStreamer frame has a stale pipeline timestamp")
        if self.mode == "recorded":
            expected = self.recorded[source][frame % len(self.recorded[source])]
            if value != expected:
                raise ValueError("recorded camera observation changed during paired trial")
        self.last_metadata = {
            "source": self.urls[source],
            "mode": self.mode,
            "frame_index": packet["index"],
            "image_encodings": {name: image["mime_type"] for name, image in packet["images"].items()},
            "image_payload_bytes": image_payload_bytes(packet),
            "read_decode_s": time.perf_counter() - started,
            "capture_started_monotonic_s": packet["capture_started_monotonic_s"],
            "capture_finished_monotonic_s": packet["capture_finished_monotonic_s"],
            "capture_unix_s": packet["capture_unix_s"],
            "local_clock": self.local_clocks[source],
            "capture_age_s": capture_age,
            "pipeline_age_s": pipeline_age,
            "acquisition": acquisition,
        }
        return value

    def fingerprint(self):
        if self.mode == "live":
            return "live-inputs-not-identical"
        digest = hashlib.sha256()
        digest.update(json.dumps([self.state_dim, self.image_size]).encode())
        for recording in self.recorded:
            for frame in recording:
                value = {
                    **frame,
                    "images": {
                        name: hashlib.sha256(data).hexdigest() for name, data in sorted(frame["images"].items())
                    },
                }
                digest.update(json.dumps(value, sort_keys=True).encode())
        return digest.hexdigest()
