"""Bounded HID input for original Switch Joy-Con L/R; no robot access.

Report layout and output framing follow XLeRobot/software/joyconrobotics/joycon.py.
Unlike that reader, every read has a timeout and disconnected reports expire.
Stick centres are measured while the operator leaves both sticks at rest.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

VENDOR = 0x057E
PRODUCTS = {"left": 0x2006, "right": 0x2007}
BUTTONS = {
    "left": {
        "down": (5, 0),
        "up": (5, 1),
        "right": (5, 2),
        "left": (5, 3),
        "sr": (5, 4),
        "sl": (5, 5),
        "l": (5, 6),
        "zl": (5, 7),
        "minus": (4, 0),
        "stick": (4, 3),
        "capture": (4, 5),
    },
    "right": {
        "y": (3, 0),
        "x": (3, 1),
        "b": (3, 2),
        "a": (3, 3),
        "sr": (3, 4),
        "sl": (3, 5),
        "r": (3, 6),
        "zr": (3, 7),
        "plus": (4, 1),
        "stick": (4, 2),
        "home": (4, 4),
    },
}


@dataclass(frozen=True)
class JoyconSample:
    side: str
    raw_stick: tuple[int, int]
    buttons: frozenset[str]
    battery_level: int
    received_at: float
    timer: int
    stick: tuple[float, float] = (0.0, 0.0)


def decode_report(report: bytes, side: str, now: float) -> JoyconSample:
    if side not in PRODUCTS:
        raise ValueError("side must be left or right")
    if len(report) < 49 or report[0] != 0x30:
        raise ValueError("expected a complete Joy-Con 0x30 report")
    offset = 6 if side == "left" else 9
    a, b, c = report[offset : offset + 3]
    return JoyconSample(
        side,
        (a | ((b & 15) << 8), (b >> 4) | (c << 4)),
        frozenset(name for name, (byte, bit) in BUTTONS[side].items() if report[byte] & (1 << bit)),
        (report[2] >> 5) & 7,
        now,
        report[1],
    )


def normalize_axis(raw: int, centre: float, *, span: float = 1000, deadzone: float = 0.15) -> float:
    value = max(-1.0, min(1.0, (raw - centre) / span))
    if abs(value) <= deadzone:
        return 0.0
    return (1 if value > 0 else -1) * (abs(value) - deadzone) / (1 - deadzone)


def resting_centre(samples: list[JoyconSample]) -> tuple[float, float]:
    if len(samples) < 20:
        raise RuntimeError("手柄报告不足，检查连接后重试")
    if any(sample.buttons for sample in samples):
        raise RuntimeError("测量摇杆中心时请松开所有按键")
    axes = list(zip(*(sample.raw_stick for sample in samples)))
    if any(max(axis) - min(axis) > 100 for axis in axes):
        raise RuntimeError("摇杆仍在移动，请归中并重新运行")
    centres = tuple(sum(axis) / len(axis) for axis in axes)
    if any(not 1400 <= centre <= 2700 for centre in centres):
        raise RuntimeError("摇杆读数偏离正常中心，请松开摇杆或检查手柄型号")
    return centres


def hid_module():
    try:
        import hid
    except ImportError as exc:
        raise RuntimeError("缺少 hidapi；用当前 Python 运行 -m pip install hidapi") from exc
    return hid


def devices() -> list[dict]:
    return [item for item in hid_module().enumerate(VENDOR, 0) if item["product_id"] in PRODUCTS.values()]


class JoyconDevice:
    def __init__(self, side: str, descriptor: dict):
        self.side, self.last, self.centre = side, None, None
        self.device = hid_module().device()
        try:
            self.device.open_path(descriptor["path"])
            # Select the standard full input report; do not enable rumble or IMU.
            self.device.write(bytes([0x01, 0]) + bytes.fromhex("0001404000014040") + bytes([0x03, 0x30]))
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                report = bytes(self.device.read(64, 50))
                if len(report) >= 49 and report[0] == 0x30:
                    self.last = decode_report(report, side, time.monotonic())
                    return
            raise RuntimeError(f"{side}: 已打开 HID，但没有收到 0x30 输入；检查配对/驱动")
        except BaseException:
            self.device.close()
            raise

    def poll(self, now: float | None = None) -> JoyconSample:
        # Drain a bounded queue so a stalled consumer cannot replay old motion.
        for _ in range(128):
            report = bytes(self.device.read(64, 1))
            if not report:
                break
            if len(report) >= 49 and report[0] == 0x30:
                sample = decode_report(report, self.side, time.monotonic())
                if self.last is None or sample.timer != self.last.timer:
                    self.last = sample
        else:
            raise RuntimeError(f"{self.side}: 手柄输入积压，请重新连接")
        now = time.monotonic() if now is None else now
        if self.last is None or now - self.last.received_at > 0.25:
            raise RuntimeError(f"{self.side}: 手柄输入已过期/连接断开")
        sample = self.last
        if self.centre is not None:
            sample = replace(
                sample,
                stick=tuple(normalize_axis(raw, centre) for raw, centre in zip(sample.raw_stick, self.centre)),
            )
        return sample

    def close(self):
        self.device.close()


class JoyconPair:
    def __init__(self):
        self.controllers = {}
        found = devices()
        try:
            for side, product in PRODUCTS.items():
                matches = {item["path"]: item for item in found if item["product_id"] == product}
                if len(matches) != 1:
                    raise RuntimeError(f"{side}: 需要恰好一只手柄，当前检测到 {len(matches)} 只")
                self.controllers[side] = JoyconDevice(side, next(iter(matches.values())))
        except BaseException:
            self.close()
            raise

    def calibrate(self):
        print("请松开两只摇杆和全部按键，静置约 1.5 秒测量中心。", flush=True)
        samples = {side: [] for side in PRODUCTS}
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            for side, sample in self.read().items():
                if not samples[side] or samples[side][-1].timer != sample.timer:
                    samples[side].append(sample)
            time.sleep(0.01)
        for side, controller in self.controllers.items():
            controller.centre = resting_centre(samples[side])
        return {side: controller.centre for side, controller in self.controllers.items()}

    def read(self):
        return {side: controller.poll() for side, controller in self.controllers.items()}

    def close(self):
        for controller in self.controllers.values():
            controller.close()
        self.controllers.clear()
