"""Read original DualSense HID reports; this module never accesses the robot.

Report layouts are documented by SDL's SDL_hidapi_ps5.c and Linux's
drivers/hid/hid-playstation.c. Supports generic hidraw on Jetson, without
requiring hid_playstation, rumble, adaptive-trigger or LED output writes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

VENDOR, PRODUCT = 0x054C, 0x0CE6
STALE_SECONDS = 0.25


def axis(raw: int, deadzone: float = 0.12) -> float:
    """Zero-centred stick with symmetric endpoints and a rescaled deadzone."""
    if not math.isfinite(deadzone) or not 0 <= deadzone < 1:
        raise ValueError("invalid deadzone")
    value = (raw - 127.5) / 127.5
    if abs(value) <= deadzone:
        return 0.0
    return math.copysign((abs(value) - deadzone) / (1 - deadzone), value)


@dataclass(frozen=True)
class DualSenseSample:
    raw_axes: tuple[int, int, int, int]
    sticks: tuple[float, float, float, float]
    triggers: tuple[float, float]
    buttons: frozenset[str]
    received_at: float
    sequence: int
    report_format: str

    @property
    def neutral(self) -> bool:
        return not any(self.sticks) and max(self.triggers) < 0.05 and not self.buttons


def decode_report(report: bytes, now: float) -> DualSenseSample:
    if not math.isfinite(now):
        raise ValueError("invalid receive timestamp")
    if len(report) in (10, 78) and report[0] == 0x01:
        # Bluetooth compatibility reports have DS4-style button/trigger order.
        data = report[1:10]
        raw = tuple(data[:4])
        buttons = data[4:7]
        triggers = data[7:9]
        sequence = buttons[2] >> 2
        report_format = "bluetooth-simple"
    elif (len(report) == 78 and report[0] == 0x31) or (len(report) == 64 and report[0] == 0x01):
        bluetooth = report[0] == 0x31
        if bluetooth:
            # The HID protocol's existing CRC includes the input transaction byte.
            crc = zlib.crc32(b"\xa1" + report[:-4])
            if int.from_bytes(report[-4:], "little") != crc:
                raise ValueError("DualSense Bluetooth report CRC mismatch")
        offset = 2 if bluetooth else 1
        data = report[offset:]
        raw = tuple(data[:4])
        triggers, buttons = data[4:6], data[7:10]
        sequence = int.from_bytes(data[27:31], "little")
        report_format = "bluetooth-full" if bluetooth else "usb-full"
    else:
        raise ValueError("unsupported or incomplete DualSense input report")
    names = set()
    for i, name in enumerate(("square", "cross", "circle", "triangle")):
        if buttons[0] & (0x10 << i):
            names.add(name)
    for i, name in enumerate(("l1", "r1", "l2", "r2", "create", "options", "l3", "r3")):
        if buttons[1] & (1 << i):
            names.add(name)
    for i, name in enumerate(("ps", "touchpad")):
        if buttons[2] & (1 << i):
            names.add(name)
    if report_format != "bluetooth-simple" and buttons[2] & 4:
        names.add("mute")
    hat = buttons[0] & 15
    if hat > 8:
        raise ValueError("invalid DualSense direction pad")
    for name, values in (("up", (7, 0, 1)), ("right", (1, 2, 3)), ("down", (3, 4, 5)), ("left", (5, 6, 7))):
        if hat in values:
            names.add(name)
    return DualSenseSample(
        raw,
        tuple(axis(v) for v in raw),
        tuple(v / 255 for v in triggers),
        frozenset(names),
        now,
        sequence,
        report_format,
    )


def hid_module():
    try:
        import hid
    except ImportError as exc:
        raise RuntimeError("缺少 hidapi；请在当前 Python 环境安装 hidapi") from exc
    return hid


def linux_devices(root: Path = Path("/sys/class/hidraw")) -> list[dict]:
    found = []
    for node in sorted(root.glob("hidraw*")):
        try:
            fields = dict(
                line.split("=", 1) for line in (node / "device/uevent").read_text().splitlines() if "=" in line
            )
            bus, vendor, product = (int(v, 16) for v in fields.get("HID_ID", "").split(":"))
        except (OSError, ValueError):
            continue
        if (vendor, product) == (VENDOR, PRODUCT) and bus in (3, 5):
            found.append(
                {
                    "path": "/dev/" + node.name,
                    "vendor_id": vendor,
                    "product_id": product,
                    "product_string": fields.get("HID_NAME"),
                    "serial_number": fields.get("HID_UNIQ"),
                    "transport": "bluetooth" if bus == 5 else "usb",
                }
            )
    return found


def devices() -> list[dict]:
    if sys.platform.startswith("linux"):
        # Some hidapi wheels use libusb and silently omit Bluetooth devices.
        return linux_devices()
    return hid_module().enumerate(VENDOR, PRODUCT)


class LinuxHidraw:
    def open_path(self, path):
        self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)

    def set_nonblocking(self, enabled):
        os.set_blocking(self.fd, not enabled)

    def read(self, size):
        try:
            data = os.read(self.fd, size)
        except BlockingIOError:
            return b""
        if not data:
            raise OSError("DualSense HID device closed")
        return data

    def close(self):
        if getattr(self, "fd", None) is not None:
            os.close(self.fd)
            self.fd = None


class DualSenseDevice:
    def __init__(self):
        found = {item["path"]: item for item in devices()}
        if len(found) != 1:
            raise RuntimeError(f"需要恰好一只原装 DualSense，当前检测到 {len(found)} 只")
        self.descriptor = next(iter(found.values()))
        self.device = LinuxHidraw() if sys.platform.startswith("linux") else hid_module().device()
        self.last = None
        self.reports = 0
        try:
            self.device.open_path(self.descriptor["path"])
            self.device.set_nonblocking(True)
        except BaseException:
            self.device.close()
            raise

    def poll(self) -> DualSenseSample:
        # Drain the queue before consuming an input, never replay queued motion.
        # The finite limit also fails closed on an unexpectedly flooded device.
        for _ in range(256):
            report = bytes(self.device.read(128))
            if not report:
                break
            sample = decode_report(report, time.monotonic())
            self.reports += 1
            if self.last is None or (sample.report_format, sample.sequence) != (
                self.last.report_format,
                self.last.sequence,
            ):
                self.last = sample
        else:
            raise RuntimeError("DualSense 输入积压；请退出后重新连接")
        if self.last is None or time.monotonic() - self.last.received_at > STALE_SECONDS:
            raise RuntimeError("DualSense 输入过期或已断开")
        return self.last

    def close(self):
        self.device.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("list", "monitor"), default="monitor")
    parser.add_argument("--seconds", type=float, default=20)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")
    if args.mode == "list":
        print(json.dumps(devices(), default=str, ensure_ascii=False))
        return
    device = DualSenseDevice()
    print("仅测试手柄输入：不连接机器人，不会驱动车轮。", flush=True)
    start = time.monotonic()
    last_display = 0
    seen = set()
    low, high = [255] * 4, [0] * 4
    try:
        # Wait briefly for the first physical report after opening the HID node.
        while time.monotonic() - start < args.seconds:
            try:
                sample = device.poll()
            except RuntimeError:
                if device.last is not None or time.monotonic() - start >= 1:
                    raise
                time.sleep(0.01)
                continue
            seen.update(sample.buttons)
            low = [min(a, b) for a, b in zip(low, sample.raw_axes)]
            high = [max(a, b) for a, b in zip(high, sample.raw_axes)]
            now = time.monotonic()
            if now - last_display >= 0.5:
                payload = asdict(sample)
                payload["buttons"] = sorted(sample.buttons)
                payload["reports"] = device.reports
                print(json.dumps(payload, ensure_ascii=False), flush=True)
                last_display = now
            time.sleep(0.01)
    finally:
        device.close()
        print(
            json.dumps(
                {
                    "summary": {
                        "reports": device.reports,
                        "buttons_seen": sorted(seen),
                        "axis_min": low,
                        "axis_max": high,
                    }
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
