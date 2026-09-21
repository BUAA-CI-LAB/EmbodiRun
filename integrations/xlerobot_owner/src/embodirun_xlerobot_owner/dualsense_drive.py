"""AGX-local DualSense base control through the existing scoped robot API.

No serial access, arm targets, automatic enable or robot-service startup.
The operator must keep the robot in sight. Use dualsense_input first.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import signal
import time
from pathlib import Path
from urllib.parse import urlparse

from .dualsense_input import STALE_SECONDS, DualSenseDevice
from .dualsense_recording import DemoRecording, input_record
from .robot import RemoteRobot

ZERO = {"x.vel": 0.0, "theta.vel": 0.0}
TETHERED_MAX_LINEAR_M_S = 0.025
TETHERED_MAX_HOLD_S = 2.0


def positive(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("速度上限必须是有限正数")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("速度上限必须是有限正数")
    return float(value)


def resolve_motion_profile(*, takeover: bool, profile: str | None) -> str:
    """Use a conservative default only for explicit human takeover."""

    selected = profile or ("tethered" if takeover else "free")
    if selected not in ("tethered", "free"):
        raise ValueError("motion profile must be tethered or free")
    return selected


class DriveMapping:
    def __init__(self, metadata, *, motion_profile="free"):
        if motion_profile not in ("tethered", "free"):
            raise ValueError("motion profile must be tethered or free")
        if metadata.get("allow_motion") is not True or metadata.get("enable_base") is not True:
            raise RuntimeError("机器人服务尚未开放底盘")
        if "base" not in metadata.get("control_scopes", []):
            raise RuntimeError("机器人服务不支持独立 base 控制")
        limits = metadata.get("base_velocity_limits", {})
        self.motion_profile = motion_profile
        self.max_linear = positive(limits.get("linear_m_s"))
        self.max_angular = positive(limits.get("angular_deg_s"))
        self.level = 1
        self.ready = False
        self.previous = frozenset()

    @property
    def speed(self):
        return (
            min(
                self.level * 0.05,
                self.max_linear,
                TETHERED_MAX_LINEAR_M_S if self.motion_profile == "tethered" else float("inf"),
            ),
            0.0 if self.motion_profile == "tethered" else min(self.level * 10, self.max_angular),
        )

    def require_new_enable_edge(self):
        """Require R1 release and a fresh press after a bounded hold."""

        self.ready = False

    def update(self, sample):
        """Returns enable edge and desired base velocity; never arm positions."""
        buttons = sample.buttons
        rising = buttons - self.previous
        self.previous = buttons
        resting = not any(sample.sticks) and max(sample.triggers) < 0.05
        if sample.neutral:
            self.ready = True
        # Change speed only while stopped and on a fresh d-pad press.
        if resting and not (buttons - {"up", "down"}):
            self.level = max(1, min(3, self.level + int("up" in rising) - int("down" in rising)))
        enable = self.ready and resting and buttons == {"r1"} and "r1" in rising
        if "r1" in rising:
            self.ready = False
        if "r1" not in buttons or "cross" in buttons or "circle" in buttons:
            return enable, dict(ZERO)
        linear, angular = self.speed
        # DualSense up/left are negative; base forward/left yaw are positive.
        return enable, {"x.vel": -sample.sticks[1] * linear, "theta.vel": -sample.sticks[2] * angular}


class HeadTiltMapping:
    def __init__(self, metadata, *, enabled=True):
        if not enabled:
            self.joint = None
            self.target = None
            return
        info = metadata.get("head_tilt") or {}
        self.joint = info.get("joint")
        self.target = None
        if self.joint is None:
            return
        if self.joint != "head_motor_2.pos" or info.get("scope") != "base":
            raise ValueError("不支持的相机俯仰控制配置")
        self.low, self.high = metadata["joint_limits"][self.joint]
        if not all(math.isfinite(v) for v in (self.low, self.high)) or self.low >= self.high:
            raise ValueError("相机俯仰限位无效")
        self.speed = min(positive(info["max_speed_deg_s"]), 15)
        self.sign = info["up_sign"]
        if type(self.sign) is not int or self.sign not in (-1, 1):
            raise ValueError("相机俯仰方向未确认")

    def reset(self, enable_result):
        if self.joint:
            value = enable_result.get("held_positions", {}).get(self.joint)
            if not isinstance(value, (int, float)) or not self.low <= value <= self.high:
                raise RuntimeError("缺少启用时的相机当前保持位置")
            self.target = float(value)

    def action(self, sample, dt):
        if not self.joint:
            return {}
        if self.target is None or not 0 < dt <= STALE_SECONDS:
            raise RuntimeError("相机俯仰未初始化或输入延迟")
        if "r1" in sample.buttons:
            self.target = max(
                self.low, min(self.high, self.target - sample.sticks[3] * self.speed * min(dt, 0.1) * self.sign)
            )
        return {self.joint: self.target}

    def accept(self, applied):
        if self.joint:
            value = applied[self.joint]
            if not self.low <= value <= self.high:
                raise RuntimeError("相机回执超出限位")
            self.target = value


def check_feedback(feedback, expected_keys=None):
    if feedback.get("accepted") is not True or feedback.get("command_accepted") is False:
        raise RuntimeError("底盘拒绝动作: " + str(feedback.get("errors", [])))
    applied = feedback.get("applied_action", {})
    if set(applied) != (set(ZERO) if expected_keys is None else set(expected_keys)) or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in applied.values()
    ):
        raise RuntimeError("底盘动作回执不完整")


def fresh_input(device):
    """Discard pre-enable backlog; require a newly arrived physical report."""
    for _ in range(256):
        if not device.device.read(128):
            break
    else:
        raise RuntimeError("启用后手柄输入积压")
    device.last = None
    deadline = time.monotonic() + STALE_SECONDS
    while time.monotonic() < deadline:
        try:
            return device.poll()
        except RuntimeError:
            if device.last is not None:
                raise
            time.sleep(0.005)
    raise RuntimeError("没有收到新鲜 DualSense 输入")


class BaseSession:
    """Only sends scoped base commands; never stops someone else's control on rejection."""

    def __init__(self, robot, recording=None):
        self.robot = robot
        self.recording = recording
        self.command_index = 0
        self.owns = False
        self.enable_uncertain = False

    def enable(self):
        if self.recording:
            self.recording.emit("enable_requested")
        try:
            result = self.robot.arm()
        except (TimeoutError, ConnectionError):
            # The server might have enabled before its reply was lost.
            self.enable_uncertain = True
            raise
        self.owns = True
        if self.recording:
            self.recording.emit("enable_result", feedback=result)
        return result

    def takeover(self):
        """Preempt only the scoped base owner before a fresh arm request."""

        if self.owns or self.enable_uncertain:
            raise RuntimeError("底盘已由本程序启用，不能重复接管")
        if self.recording:
            self.recording.emit("takeover_requested", scope="base")
        # RemoteRobot(scope="base").stop() maps to /robot/stop with the
        # base scope. It never calls stop_all and therefore leaves arms alone.
        result = self.robot.stop()
        if not isinstance(result, dict) or result.get("stop_confirmed") is not True or result.get("errors"):
            raise RuntimeError("接管前底盘停止未确认，请先检查机器人反馈: " + str(result))
        self.owns = self.enable_uncertain = False
        if self.recording:
            self.recording.emit("takeover_result", feedback=result)
        return result

    def send(self, action, input_sample=None):
        if not self.owns:
            raise RuntimeError("尚未启用底盘")
        self.command_index += 1
        if self.recording:
            self.recording.emit(
                "command_requested", command_index=self.command_index, action=action, input_sample=input_sample
            )
        feedback = self.robot.command(action)
        if self.recording:
            self.recording.emit("command_feedback", command_index=self.command_index, feedback=feedback)
        check_feedback(feedback, action)
        return feedback["applied_action"]

    def stop(self):
        if self.owns or self.enable_uncertain:
            result = self.robot.release()
            # Control release happens before any possibly failed recording I/O.
            if self.recording:
                with contextlib.suppress(RuntimeError):
                    self.recording.emit("release_result", feedback=result)
            if result.get("control_owned") is not False and result.get("stop_confirmed") is not True:
                raise RuntimeError("底盘停止未确认，请现场检查: " + str(result))
            self.owns = self.enable_uncertain = False


def run(args):
    url = urlparse(args.robot_url)
    if url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("请在 AGX 本机使用回环 robot URL")
    token = args.token_file.read_text().strip()
    robot = RemoteRobot(args.robot_url, token, timeout=0.25, scope="base")
    robot.connect()
    motion_profile = resolve_motion_profile(takeover=args.takeover, profile=args.motion_profile)
    mapping = DriveMapping(robot.metadata, motion_profile=motion_profile)
    head = HeadTiltMapping(robot.metadata, enabled=motion_profile == "free")
    device = DualSenseDevice()
    recording = None
    session = BaseSession(robot)
    last_display = 0
    try:
        if args.output is not None:
            recording = DemoRecording(
                args.output,
                args.task,
                args.robot_url,
                token,
                {**robot.metadata, "takeover": args.takeover, "motion_profile": motion_profile},
                max_seconds=args.max_seconds,
            )
            session.recording = recording
            recording.wait_ready()
            recording.emit("ready", metadata=robot.metadata)
            print("示范录制已开始: " + str(recording.path), flush=True)
        sample = fresh_input(device)
        # Prime the neutral/deadman edge from the fresh startup report. A
        # centered report followed by R1 can therefore arm on the next poll;
        # a pre-held R1 still requires release and repress.
        mapping.update(sample)
        last_tick = time.monotonic()
        profile_hint = (
            "当前为 tethered：仅直行、速度不超过 0.025 m/s；每次按住 R1 最多 2 秒，停车后必须松开再按住 R1。"
            if motion_profile == "tethered"
            else "当前为 free：使用服务公布的底盘速度和相机俯仰上限。"
        )
        print(
            "底盘尚未启用。请保持小车在视线内；摇杆归中并松开所有键，然后按住 R1。\n"
            "左杆前后、右杆左右转向、右杆上下相机俯仰；松 R1 停车/保持镜头；\n"
            "↑/↓切换速度；×/○ 停止并释放底盘。\n" + profile_hint + "\n"
            "只控制底盘，不会启动机器人服务或修改双臂。",
            flush=True,
        )
        if not head.joint:
            print("当前服务未启用相机俯仰，右杆上下无效。", flush=True)
        takeover_pending = args.takeover
        hold_started = None
        while True:
            if recording:
                recording.check()
            sample = device.poll()
            now = time.monotonic()
            if now - last_tick > STALE_SECONDS:
                raise RuntimeError("底盘控制循环延迟，停止并退出")
            dt = max(0.001, now - last_tick)
            last_tick = now
            if {"cross", "circle"} & sample.buttons:
                break
            enable, action = mapping.update(sample)
            if not session.owns and enable:
                if takeover_pending:
                    takeover_result = session.takeover()
                    print(
                        "已停止原底盘控制并确认接管窗口: " + json.dumps(takeover_result, ensure_ascii=False), flush=True
                    )
                    takeover_pending = False
                    # The stop request can block for the full control timeout.
                    # Never arm from the pre-stop R1 packet.
                    sample = fresh_input(device)
                    if sample.buttons != {"r1"} or any(sample.sticks) or max(sample.triggers) >= 0.05:
                        mapping.require_new_enable_edge()
                        last_tick = time.monotonic()
                        print("停止期间已松开 R1 或输入不再中立；本次未接管，请松开后重新按住 R1。", flush=True)
                        continue
                try:
                    enable_result = session.enable()
                except RuntimeError as exc:
                    refused = getattr(robot, "last_arm_result", None)
                    if session.owns or not refused or refused.get("armed") is not False:
                        raise
                    if recording:
                        recording.emit("enable_refused", feedback=refused)
                    print(
                        "本次启用被拒绝（未运动）: " + str(exc) + "；松开所有键后，可重新按 R1 手动尝试。", flush=True
                    )
                    mapping = DriveMapping(robot.metadata, motion_profile=motion_profile)
                    fresh_input(device)
                    last_tick = time.monotonic()
                    continue
                head.reset(enable_result)
                # Enabling hardware can block; never use a pre-enable stick value.
                sample = fresh_input(device)
                if sample.buttons != {"r1"} or any(sample.sticks) or max(sample.triggers) >= 0.05:
                    raise RuntimeError("启用过程中请保持摇杆归中且按住 R1；本次已取消")
                last_tick = time.monotonic()
                action = dict(ZERO)
            if session.owns:
                if time.monotonic() - sample.received_at > STALE_SECONDS:
                    raise RuntimeError("DualSense 输入已过期")
                if motion_profile == "tethered" and "r1" in sample.buttons:
                    hold_started = now if hold_started is None else hold_started
                    if now - hold_started >= TETHERED_MAX_HOLD_S:
                        session.stop()
                        mapping.require_new_enable_edge()
                        hold_started = None
                        sample = fresh_input(device)
                        last_tick = time.monotonic()
                        print("tethered 单次保持已到 2 秒；已停车，请松开并重新按住 R1。", flush=True)
                        continue
                elif "r1" not in sample.buttons:
                    hold_started = None
                action.update(head.action(sample, dt))
                head.accept(session.send(action, input_record(sample)))
            elif recording:
                recording.emit("input", input_sample=input_record(sample), enabled=False)
            if now - last_display >= 1:
                print(
                    json.dumps(
                        {
                            "enabled": session.owns,
                            "r1": "r1" in sample.buttons,
                            "speed_level": mapping.level,
                            "speed_limits": mapping.speed,
                            "requested_action": action if session.owns else ZERO,
                        }
                    ),
                    flush=True,
                )
                last_display = now
            time.sleep(max(0, 0.05 - (time.monotonic() - last_tick)))
    except BaseException as exc:
        if recording:
            with contextlib.suppress(RuntimeError):
                recording.emit("control_exit", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            session.stop()
        finally:
            device.close()
            if recording:
                result = recording.close()
                print("示范素材已保存: " + json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-url", default="http://127.0.0.1:8766")
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="保存三路JPEG、原始观测和独立手柄指令日志")
    parser.add_argument("--task", default="DualSense 手动导航示范，供 AI 参考")
    parser.add_argument("--max-seconds", type=lambda value: positive(float(value)), default=600)
    parser.add_argument(
        "--takeover",
        action="store_true",
        help="显式停止当前 base 控制者后再接管；只影响底盘，不停止双臂",
    )
    parser.add_argument(
        "--motion-profile",
        choices=("tethered", "free"),
        default=None,
        help="运动约束；接管默认 tethered，普通手动驾驶默认保留 free 行为",
    )
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        run(args)
    except KeyboardInterrupt:
        print("驾驶程序退出。", flush=True)


if __name__ == "__main__":
    main()
