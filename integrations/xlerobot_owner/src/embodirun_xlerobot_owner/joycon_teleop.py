"""Dual Joy-Con input test and local client of the existing Orin robot service.

Run with --mode monitor first. Control remains disabled until the operator
holds minus + plus for one second with the sticks and motion buttons released.
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path
from urllib.parse import urlparse

from .control import CAMERAS, JOINT_NAMES, MappingConfig, QuestMapper, clamp, finite
from .joycon_input import PRODUCTS, JoyconPair, JoyconSample, devices
from .robot import RemoteRobot


def neutral(samples: dict[str, JoyconSample]) -> bool:
    return all(not (sample.buttons - {"minus", "plus"}) and not any(sample.stick) for sample in samples.values())


class EnableGesture:
    def __init__(self):
        self.since = None
        self.released = False

    def update(self, samples, now):
        chord = "minus" in samples["left"].buttons and "plus" in samples["right"].buttons
        if not chord:
            self.released, self.since = True, None
            return False
        if not self.released or not neutral(samples):
            self.since = None
            return False
        if self.since is None:
            self.since = now
        if now - self.since >= 1:
            self.released, self.since = False, None
            return True
        return False


class JoyconMapper:
    """Joystick reach/pan, shoulder-button height, face-button wrist control.

    Reuses the installed SO101 IK and joint speed/limit handling. Does not reset
    the arm, alter calibration, or send any base/head command.
    """

    def __init__(self, config: MappingConfig | None = None):
        self.solver = QuestMapper(config or MappingConfig(max_joint_speed_deg_s=10))
        self.targets = None
        self.previous = {side: frozenset() for side in PRODUCTS}
        self.gripper_direction = dict.fromkeys(PRODUCTS, 1)

    def reset(self):
        self.targets = None
        self.previous = {side: frozenset() for side in PRODUCTS}
        self.gripper_direction = dict.fromkeys(PRODUCTS, 1)

    def map(self, samples, state, dt, limits):
        if not 0 < finite(dt) <= 0.25:
            raise RuntimeError("control loop delayed; stop and restart")
        dt = min(dt, 0.05)
        if set(samples) != set(PRODUCTS):
            raise RuntimeError("both Joy-Cons required")
        if self.targets is None:
            self.targets = {name: finite(state[name], name) for name in JOINT_NAMES}
        result = dict(self.targets)
        cfg = self.solver.config
        for side in PRODUCTS:
            sample = samples[side]
            buttons = sample.buttons
            if side not in cfg.enabled_arms:
                raise RuntimeError("this client requires both enabled arms")
            x, y = sample.stick
            up = "l" if side == "left" else "r"
            # The planar solver uses screen coordinates: negative z is reach.
            delta = (
                x * 8 * dt / cfg.pan_deg_per_m,
                (int(up in buttons) - int("stick" in buttons)) * 0.02 * dt / cfg.position_scale,
                -y * 0.02 * dt / cfg.position_scale,
            )
            pitch_up, pitch_down, roll_left, roll_right = (
                ("up", "down", "left", "right") if side == "left" else ("x", "b", "y", "a")
            )
            rotation = (
                (int(pitch_up in buttons) - int(pitch_down in buttons)) * 8 * dt,
                (int(roll_right in buttons) - int(roll_left in buttons)) * 8 * dt,
            )
            names = tuple(name for name in JOINT_NAMES if name.startswith(side + "_"))
            arm, _, _ = self.solver._arm_step(names, result, delta, rotation, dt, limits, side)
            result.update(arm)
            trigger = "zl" if side == "left" else "zr"
            if trigger in buttons:
                if trigger not in self.previous[side]:
                    self.gripper_direction[side] *= -1
                low, high = limits[names[-1]]
                result[names[-1]] = clamp(
                    result[names[-1]] + self.gripper_direction[side] * cfg.max_gripper_speed_pct_s * dt,
                    low,
                    high,
                )
            self.previous[side] = buttons
        self.targets = result
        return result

    def accept(self, feedback):
        if feedback.get("accepted") is False or feedback.get("command_accepted") is False:
            raise RuntimeError("动作被拒绝: " + "; ".join(feedback.get("errors", [])))
        applied = feedback.get("applied_action", {})
        if any(name not in applied for name in JOINT_NAMES):
            raise RuntimeError("双臂动作回执不完整")
        self.targets = {name: finite(applied[name], name) for name in JOINT_NAMES}


def validate_observation(observation, images, metadata, previous_timestamp=None):
    limits = metadata.get("joint_limits", {})
    if metadata.get("joint_unit") != "degrees" or metadata.get("gripper_unit") != "range_0_100":
        raise RuntimeError("需要 degrees 关节与 range_0_100 夹爪配置")
    source = observation.get("source_timestamp_ns")
    if not isinstance(source, int) or source <= 0 or (previous_timestamp is not None and source <= previous_timestamp):
        raise RuntimeError("机器人观测没有更新")
    timestamps = observation.get("camera_timestamps_ns", {})
    state_stamp = observation.get("state_timestamp_ns")
    if not isinstance(state_stamp, int) or abs(source - state_stamp) > 500_000_000:
        raise RuntimeError("机器人关节反馈已过期")
    if observation.get("errors"):
        raise RuntimeError("机器人观测报错: " + str(observation["errors"]))
    if metadata.get("camera_roles_confirmed") is not True or any(
        not images.get(name)
        or not isinstance(timestamps.get(name), int)
        or abs(source - timestamps[name]) > 500_000_000
        for name in CAMERAS
    ):
        raise RuntimeError("三路相机缺失、角色未确认或图像已过期")
    for name in JOINT_NAMES:
        bounds = limits.get(name)
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise RuntimeError(f"缺少当前标定限位: {name}")
        low, high = (finite(v, name) for v in bounds)
        value = finite(observation["state"][name], name)
        if not low <= value <= high:
            raise RuntimeError(f"关节实测位置超出标定限位: {name}")
    return source


def run(args):
    if args.mode == "list":
        found = devices()
        print(
            json.dumps(
                [
                    {
                        "side": next(side for side, product in PRODUCTS.items() if product == item["product_id"]),
                        "name": item.get("product_string"),
                        "path": str(item["path"]),
                    }
                    for item in found
                ],
                ensure_ascii=False,
            )
        )
        return
    if args.mode == "control" and (
        urlparse(args.robot_url).hostname not in ("127.0.0.1", "localhost", "::1") or args.token_file is None
    ):
        raise ValueError("控制模式需要本机 robot URL 和 --token-file；跨机器使用 SSH 隧道")
    pair, robot = None, None
    try:
        pair = JoyconPair()
        print("摇杆中心: " + json.dumps(pair.calibrate()), flush=True)
        mapper = JoyconMapper(
            MappingConfig(**json.loads(args.mapping_config.read_text())) if args.mapping_config else None
        )
        gesture = EnableGesture()
        if args.mode == "control":
            robot = RemoteRobot(args.robot_url, args.token_file.read_text().strip(), timeout=2)
            robot.connect()
            if robot.metadata.get("allow_motion") is not True:
                raise RuntimeError("现有机器人服务未开放运动")
            robot.timeout = 0.3
            print(
                "控制尚未启用。摇杆归中，松开其他键，同时长按 − 和 + 1 秒启用。"
                " Home/截图键或 Ctrl+C 停止。底盘/头部不受本程序控制。",
                flush=True,
            )
        else:
            print("输入测试：不会连接机器人；可按键、转动摇杆。Ctrl+C 退出。", flush=True)
        start = last_tick = time.monotonic()
        last_display = 0
        previous_timestamp = None
        while args.seconds == 0 or time.monotonic() - start < args.seconds:
            now = time.monotonic()
            samples = pair.read()
            if any(now - sample.received_at > 0.25 for sample in samples.values()):
                raise RuntimeError("手柄输入过期")
            stop_pressed = "capture" in samples["left"].buttons or "home" in samples["right"].buttons
            if robot and stop_pressed:
                break
            if robot:
                was_armed = robot.armed
                observation, images = robot.read()
                if was_armed and not robot.armed:
                    raise RuntimeError("机器人侧已停止，请检查原因后重新运行")
                stamp = validate_observation(observation, images, robot.metadata, previous_timestamp)
                previous_timestamp = stamp
                # Reading the cameras/network may have blocked. Read HID again
                # before enabling or commanding; expired input can never move.
                samples = pair.read()
                now = time.monotonic()
                if "capture" in samples["left"].buttons or "home" in samples["right"].buttons:
                    break
                if not robot.armed and gesture.update(samples, now):
                    robot.timeout = 2
                    try:
                        robot.arm()
                    finally:
                        robot.timeout = 0.3
                    mapper.reset()
                    last_tick = time.monotonic()
                    print("已启用；左右摇杆分别控制左右臂。", flush=True)
                    # Re-read inputs and feedback after the enable round trip.
                    continue
                if robot.armed:
                    action = mapper.map(
                        samples,
                        observation["state"],
                        now - last_tick,
                        robot.metadata["joint_limits"],
                    )
                    mapper.accept(robot.command(action))
            if now - last_display >= 0.5:
                print(
                    json.dumps(
                        {
                            "armed": bool(robot and robot.armed),
                            **{
                                side: {
                                    "stick": [round(v, 3) for v in sample.stick],
                                    "raw_stick": sample.raw_stick,
                                    "buttons": sorted(sample.buttons),
                                    "battery_level_0_to_4": sample.battery_level,
                                }
                                for side, sample in samples.items()
                            },
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                last_display = now
            last_tick = now
            time.sleep(max(0, 0.05 - (time.monotonic() - now)))
    finally:
        try:
            if robot and robot.armed:
                robot.timeout = 2
                result = robot.stop()
                print("停止反馈: " + json.dumps(result, ensure_ascii=False), flush=True)
                if result.get("stop_confirmed") is not True:
                    raise RuntimeError("停止尚未得到确认，请检查现有机器人服务")
        finally:
            if pair:
                pair.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("list", "monitor", "control"), default="monitor")
    parser.add_argument("--seconds", type=float, default=0, help="0 runs until Ctrl+C")
    parser.add_argument("--robot-url", default="http://127.0.0.1:8766")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--mapping-config", type=Path)
    args = parser.parse_args()
    if args.seconds < 0:
        parser.error("--seconds must be nonnegative")

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        run(args)
    except KeyboardInterrupt:
        print("已退出。", flush=True)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Joy-Con: {exc}\n")


if __name__ == "__main__":
    main()
