"""Mac browser gateway and AGX robot endpoint. Run via ``python -m embodirun_xlerobot_owner``."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import ipaddress
import json
import logging
import os
import secrets
import signal
import time
from collections import deque
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web

from .control import (
    CAMERAS,
    JOINT_NAMES,
    InputClock,
    InputDelayed,
    InputFrame,
    MappingConfig,
    QuestMapper,
    finite,
)
from .keyboard import KeyboardFrame
from .recording import EpisodeRecorder
from .robot import RemoteRobot

LOG = logging.getLogger(__name__)
PLATFORM = web.AppKey("platform", object)


def xr_diagnostics(value: Any) -> dict:
    """Bounded, informational browser state; never used to authorize motion."""
    if not isinstance(value, dict):
        return {}
    result = {
        name: value[name]
        for name in ("active", "document_hidden", "viewer_tracked", "safety_tripped")
        if type(value.get(name)) is bool
    }
    if value.get("visibility") in ("visible", "visible-blurred", "hidden", "none"):
        result["visibility"] = value["visibility"]
    sources = value.get("sources", [])
    if isinstance(sources, list):
        result["sources"] = [
            {
                "hand": source.get("hand") if source.get("hand") in ("left", "right", "none") else "none",
                "gamepad": source.get("gamepad") is True,
                "grip_space": source.get("grip_space") is True,
                "profiles": [p[:80] for p in source.get("profiles", [])[:4] if isinstance(p, str)]
                if isinstance(source.get("profiles"), list)
                else [],
            }
            for source in sources[:6]
            if isinstance(source, dict)
        ]
    return result


class LeaderProcess:
    """One explicitly configured local dual-leader process; never a shell command."""

    def __init__(self, command: tuple[str, ...] | None):
        self.command = command
        self.process: asyncio.subprocess.Process | None = None
        self.reader_task: asyncio.Task | None = None
        self.output: deque[str] = deque(maxlen=30)
        self.started_at = 0.0
        self.last_exit_code: int | None = None
        self.error: str | None = None
        self.stop_requested = False

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.returncode is None

    def status(self) -> dict:
        if self.command is None:
            state = "unavailable"
        elif self.running:
            joined = "\n".join(self.output)
            if self.stop_requested:
                state = "stopping"
            elif "同步完成；" in joined:
                state = "following"
            elif "正在平滑完成初始姿态同步" in joined:
                state = "aligning"
            else:
                state = "starting"
        elif self.error or (self.last_exit_code not in (None, 0) and not self.stop_requested):
            state = "failed"
        else:
            state = "stopped"
        return {
            "available": self.command is not None,
            "state": state,
            "pid": self.process.pid if self.running else None,
            "exit_code": self.last_exit_code,
            "error": self.error,
            "output": list(self.output),
        }

    async def start(self) -> dict:
        if self.command is None:
            raise RuntimeError("dual-leader command is not configured")
        if self.running:
            raise RuntimeError("dual-leader follow is already running")
        if self.reader_task is not None:
            await self.reader_task
        self.output.clear()
        self.error = None
        self.last_exit_code = None
        self.stop_requested = False
        self.started_at = time.monotonic()
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        try:
            process = await asyncio.create_subprocess_exec(
                *self.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
                env=environment,
            )
        except OSError as exc:
            self.error = str(exc)
            raise RuntimeError(f"cannot start dual-leader follow: {exc}") from exc
        self.process = process
        self.reader_task = asyncio.create_task(self._read_output(process))
        return self.status()

    async def _read_output(self, process: asyncio.subprocess.Process) -> None:
        assert process.stdout is not None
        while line := await process.stdout.readline():
            self.output.append(line.decode("utf-8", errors="replace").rstrip())
        exit_code = await process.wait()
        if self.process is process:
            self.last_exit_code = exit_code

    async def stop(self) -> dict:
        process = self.process
        if process is None or process.returncode is not None:
            if self.reader_task is not None:
                await self.reader_task
            return self.status()
        self.stop_requested = True
        # Signal only the Python leader process first. Its finally block needs
        # the child SSH tunnel alive long enough to issue and verify arm Stop.
        with contextlib.suppress(ProcessLookupError):
            process.send_signal(signal.SIGINT)
        try:
            await asyncio.wait_for(process.wait(), timeout=20)
        except asyncio.TimeoutError:
            self.error = "dual-leader process did not stop after SIGINT"
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), timeout=3)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
        if self.reader_task is not None:
            await self.reader_task
        return self.status()


class Platform:
    def __init__(
        self,
        robot: Any,
        token: str,
        *,
        output: Path,
        fps: float = 20,
        mapping: MappingConfig | None = None,
        robot_api: bool = False,
        browser_no_token_cidr: str | None = None,
        leader_command: tuple[str, ...] | None = None,
    ):
        if not token or len(token) < 16:
            raise ValueError("access token must contain at least 16 characters")
        if not 1 <= fps <= 60:
            raise ValueError("fps must be 1..60")
        self.robot, self.token, self.fps, self.robot_api = robot, token, fps, robot_api
        self.browser_network = None
        if browser_no_token_cidr is not None:
            if robot_api or robot.mode == "hardware":
                raise ValueError("token-free browser access is not allowed on the robot endpoint")
            network = ipaddress.ip_network(browser_no_token_cidr, strict=True)
            private = ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "127.0.0.0/8", "fc00::/7", "::1/128")
            if not any(
                network.version == allowed.version and network.subnet_of(allowed)
                for allowed in map(ipaddress.ip_network, private)
            ):
                raise ValueError("token-free browser network must be private or loopback")
            self.browser_network = network
        self.mapper = QuestMapper(mapping)
        self.recorder = EpisodeRecorder(output, fps=fps)
        self.io_lock = asyncio.Lock()
        self.control_lock = asyncio.Lock()
        self.observation: dict = {}
        self.images: dict[str, bytes] = {}
        self.connected = False
        self.error: str | None = None
        self.connection_error: str | None = None
        self.received_at = 0.0
        self.armed = False
        self.owner: web.WebSocketResponse | None = None
        self.robot_owner: str | None = None
        self.robot_owners: dict[str, str] = {}
        self.clients: set[web.WebSocketResponse] = set()
        self.frames: dict[web.WebSocketResponse, tuple[InputFrame | KeyboardFrame, dict, float]] = {}
        self.keyboard_paused = False
        self.delayed_keyboard_packets = 0
        self.keyboard_speeds: dict[web.WebSocketResponse, dict[str, float]] = {}
        self.last_feedback: dict | None = None
        self.last_action: dict | None = None
        self.last_command_at = 0.0
        self.sample_count = 0
        self.started_at = time.monotonic()
        self.task: asyncio.Task | None = None
        self.closing = False
        self.last_recording_path: str | None = None
        self.last_recorded_source: int | None = None
        self.skipped_duplicate_samples = 0
        self.session_tokens: set[str] = set()
        self.stop_unconfirmed = bool(getattr(robot, "stop_unconfirmed", False))
        self.pairing_code: str | None = None
        self.pairing_expires_at = 0.0
        self.pairing_attempts = 0
        self.video_service = None
        self.leader = LeaderProcess(leader_command)
        self.leader_lock = asyncio.Lock()
        self.leader_owner: web.WebSocketResponse | None = None

    def issue_pairing_code(self) -> str:
        """One browser, ten minutes, at most five failed attempts; never a robot credential."""
        self.pairing_code = f"{secrets.randbelow(1_000_000):06d}"
        self.pairing_expires_at = time.monotonic() + 600
        self.pairing_attempts = 0
        return self.pairing_code

    def browser_address_allowed(self, remote: str | None) -> bool:
        if self.browser_network is None or not remote:
            return False
        try:
            return ipaddress.ip_address(remote) in self.browser_network
        except ValueError:
            return False

    def authenticate_browser(self, candidate: Any) -> bool:
        if not isinstance(candidate, str):
            return False
        if secrets.compare_digest(candidate.encode(), self.token.encode()):
            return True
        if self.pairing_code is None or time.monotonic() >= self.pairing_expires_at or self.pairing_attempts >= 5:
            return False
        if secrets.compare_digest(candidate.encode(), self.pairing_code.encode()):
            self.pairing_code = None
            return True
        self.pairing_attempts += 1
        return False

    async def call(self, method: str, *args: Any) -> Any:
        # Don't cancel a serial operation and pretend its worker has stopped.
        async with self.io_lock:
            work = asyncio.create_task(asyncio.to_thread(getattr(self.robot, method), *args))
            try:
                return await asyncio.shield(work)
            except asyncio.CancelledError:
                await work
                raise

    def status(self, client: web.WebSocketResponse | None = None) -> dict:
        now = time.monotonic()
        recent = [value for value in self.frames.values() if now - value[2] < 0.5]
        latest = max(recent, key=lambda value: value[2], default=None)
        last_seen = max(self.frames.values(), key=lambda value: value[2], default=None)
        if self.owner is not None:
            last_seen = self.frames.get(self.owner)
            latest = last_seen if last_seen and now - last_seen[2] < 0.5 else None
        if self.leader_owner is not None and not self.leader.running:
            self.leader_owner = None
        return {
            "mode": self.robot.mode,
            "browser_access": "trusted_lan" if self.browser_network else "token",
            "control_mode": self.mapper.control_mode,
            "mapping_status": self.mapper.diagnostics,
            "drive_available": self.drive_unavailable_reason() is None,
            "drive_unavailable_reason": self.drive_unavailable_reason(),
            "keyboard_drive_unavailable_reason": self.keyboard_drive_unavailable_reason(),
            "drive_limits": self.keyboard_speed_limits(),
            "drive_speed": self.keyboard_speed(client if client is not None else self.owner),
            "connected": self.connected,
            "armed": self.armed or bool((self.robot_owner or self.robot_owners) and self.robot.armed),
            "control_owner": "self"
            if self.owner is not None and self.owner is client
            else "other"
            if (
                self.owner is not None
                or self.robot_owner
                or self.robot_owners
                or (
                    self.robot.mode == "remote"
                    and not self.robot.armed
                    and self.observation.get("control_state", {}).get(getattr(self.robot, "scope", "all"))
                )
            )
            else None,
            "recording": self.recorder.active,
            "recording_path": self.last_recording_path,
            "cameras": list(self.images),
            "state": self.observation.get("state", {}),
            "metadata": self.robot.metadata,
            "control_scope": getattr(self.robot, "scope", "all"),
            "control_state": self.observation.get("control_state", {}),
            "parallel_control": set(self.robot.metadata.get("control_scopes", [])) >= {"arms", "base"},
            "leader_follow": {
                **self.leader.status(),
                "owner": "self"
                if self.leader_owner is not None and self.leader_owner is client
                else "other"
                if self.leader_owner is not None
                else None,
                "unavailable_reason": self.leader_start_unavailable_reason(),
            },
            "observation_errors": self.observation.get("errors", []),
            "error": self.error,
            "feedback": self.last_feedback,
            "stop_unconfirmed": self.stop_unconfirmed,
            "keyboard_paused": self.keyboard_paused,
            "delayed_keyboard_packets": self.delayed_keyboard_packets,
            "input_status": {
                "connected_browsers": len(self.clients),
                "clients": len(recent),
                "age_ms": round((now - latest[2]) * 1000) if latest else None,
                "tracked": {side: c.tracked for side, c in latest[0].controllers.items()}
                if latest and isinstance(latest[0], InputFrame)
                else {},
                "kind": latest[1].get("type") if latest else None,
                "keys": sorted(latest[0].keys) if latest and isinstance(latest[0], KeyboardFrame) else [],
                "seq": latest[0].seq if latest else None,
                "xr": xr_diagnostics(last_seen[1].get("xr")) if last_seen else {},
                "xr_age_ms": round((now - last_seen[2]) * 1000) if last_seen else None,
                "controllers": {
                    side: {
                        "tracked": controller.tracked,
                        "neutral": controller.neutral,
                        "grip": controller.grip,
                        "trigger": controller.trigger,
                        "position": list(controller.position),
                        "orientation": list(controller.orientation),
                        "thumbstick": list(controller.thumbstick),
                        "buttons": [bool(value) for value in buttons[:8]]
                        if isinstance(buttons := latest[1]["controllers"][side].get("buttons", []), list)
                        else [],
                    }
                    for side, controller in latest[0].controllers.items()
                }
                if latest and isinstance(latest[0], InputFrame)
                else {},
            },
            "stats": {
                "skipped_duplicate_samples": self.skipped_duplicate_samples,
                "samples": self.sample_count,
                "sample_hz": round(self.sample_count / max(1, time.monotonic() - self.started_at), 2),
                "observation_age_ms": round((time.monotonic() - self.received_at) * 1000) if self.received_at else None,
                "input_age_ms": round((time.monotonic() - self.frames[self.owner][2]) * 1000)
                if self.owner in self.frames
                else None,
            },
        }

    def leader_start_unavailable_reason(self) -> str | None:
        if self.leader.command is None:
            return "本机未配置双主臂启动命令"
        if self.leader.running:
            return None
        if self.robot.mode != "remote" or getattr(self.robot, "scope", "all") != "base":
            return "双主臂网页入口只能与独立底盘网关一起运行"
        if not self.connected or time.monotonic() - self.received_at > 0.5:
            return "AGX 观测不可用或已过期"
        if self.robot.metadata.get("source") != "physical":
            return "AGX 尚未确认物理机器人来源"
        if self.robot.metadata.get("allow_motion") is not True:
            return "AGX 动作权限已关闭"
        if set(self.robot.metadata.get("control_scopes", [])) < {"arms", "base"}:
            return "AGX 不支持双臂和底盘独立控制"
        if self.observation.get("errors"):
            return "AGX 硬件反馈异常"
        if self.observation.get("control_state", {}).get("arms"):
            return "双臂控制通道已被占用"
        if self.stop_unconfirmed:
            return "上一次停止尚未确认"
        return None

    async def start_leader(self, owner: web.WebSocketResponse) -> dict:
        async with self.leader_lock:
            if reason := self.leader_start_unavailable_reason():
                raise RuntimeError(reason)
            result = await self.leader.start()
            self.leader_owner = owner
            return result

    async def stop_leader(self) -> dict:
        async with self.leader_lock:
            result = await self.leader.stop()
            self.leader_owner = None
            return result

    def drive_unavailable_reason(self) -> str | None:
        if not self.mapper.config.enable_base:
            return "Mac 配置未开放底盘控制（enable_base=false）"
        if not self.connected:
            return "等待机器人连接"
        metadata = self.robot.metadata
        if metadata.get("source") == "synthetic":
            return None
        if metadata.get("source") != "physical":
            return "机器人来源尚未确认"
        if metadata.get("enable_base") is not True:
            return "AGX 未开放底盘；需先核对轮向、轮径、轮距和停止行为"
        if metadata.get("allow_motion") is not True:
            return "AGX 动作权限已关闭"
        if (
            self.robot.mode == "remote"
            and not self.robot.armed
            and self.observation.get("control_state", {}).get("base")
        ):
            return "底盘由其他操作端控制；可继续观看，等待对方释放后再接管"
        return None

    def set_control_mode(self, mode: str) -> None:
        if mode not in ("arms", "drive"):
            raise ValueError("control mode must be arms or drive")
        if mode == self.mapper.control_mode:
            return  # A second viewing tab requesting the current mode changes nothing.
        if self.armed or self.robot.armed or self.owner is not None or self.robot_owner or self.robot_owners:
            raise RuntimeError("Stop and release control before changing mode")
        if self.stop_unconfirmed:
            raise RuntimeError("previous stop unconfirmed; mode change refused")
        if self.recorder.active:
            raise RuntimeError("finish the current recording before changing mode")
        if mode == "drive" and (reason := self.drive_unavailable_reason()):
            raise RuntimeError(reason)
        self.mapper.set_control_mode(mode)
        self.frames.clear()
        self.last_action = None

    def keyboard_speed_limits(self) -> dict[str, float]:
        limits = {
            "linear_m_s": self.mapper.config.max_linear_m_s,
            "angular_deg_s": self.mapper.config.max_angular_deg_s,
        }
        if self.robot.metadata.get("source") == "physical":
            # Older endpoints did not advertise their configured caps. Never
            # assume that raising the Mac setting raises the AGX limit too.
            remote = self.robot.metadata.get("base_velocity_limits", {})
            for key, fallback in (("linear_m_s", 0.05), ("angular_deg_s", 10.0)):
                value = remote.get(key, fallback) if isinstance(remote, dict) else fallback
                try:
                    value = finite(value, key)
                    if value <= 0:
                        raise ValueError("nonpositive limit")
                except (ValueError, TypeError):
                    value = fallback
                limits[key] = min(limits[key], value)
        return limits

    def keyboard_speed(self, client: web.WebSocketResponse | None) -> dict[str, float]:
        selected = self.keyboard_speeds.get(client, {"linear_m_s": 0.05, "angular_deg_s": 10.0})
        return {key: min(selected[key], limit) for key, limit in self.keyboard_speed_limits().items()}

    def set_keyboard_speed(self, client: web.WebSocketResponse, data: dict) -> dict:
        if self.mapper.control_mode != "drive":
            raise RuntimeError("请先切换到键盘驾驶模式")
        if reason := self.keyboard_drive_unavailable_reason():
            raise RuntimeError(reason)
        if self.stop_unconfirmed:
            raise RuntimeError("停止尚未确认，暂不能调速")
        if (self.owner is not None and self.owner is not client) or self.robot_owner or self.robot_owners:
            raise RuntimeError("另一个操作端正在驾驶，不能修改速度")
        if self.armed:
            record = self.frames.get(client)
            if (
                not record
                or not isinstance(record[0], KeyboardFrame)
                or time.monotonic() - record[2] > 0.35
                or not record[0].neutral
            ):
                raise RuntimeError("请先松开 WASD，再应用速度")
        selected = {key: finite(data.get(key), key) for key in ("linear_m_s", "angular_deg_s")}
        for key, limit in self.keyboard_speed_limits().items():
            if not 0 < selected[key] <= limit:
                raise ValueError(f"{key} 需大于 0 且不超过 {limit}")
        # Per-page selection: another tab/reload never inherits fast settings.
        self.keyboard_speeds[client] = selected
        return self.keyboard_speed(client)

    def keyboard_drive_unavailable_reason(self) -> str | None:
        if reason := self.drive_unavailable_reason():
            return reason
        if (
            self.robot.metadata.get("source") == "physical"
            and self.robot.metadata.get("enabled_arms") != []
            and (
                getattr(self.robot, "scope", "all") != "base"
                or "base" not in self.robot.metadata.get("control_scopes", [])
            )
        ):
            return "键盘开车需使用独立底盘通道；请更新 AGX 服务和本机网关"
        return None

    async def stop(self, reason: str, *, interrupt_recording: bool = True, all_scopes: bool = False) -> dict:
        # A disconnect may cancel the HTTP/WebSocket handler while stop is
        # waiting for an in-flight read's I/O lock. Finish the entire stop,
        # not just a serial call that may not have started yet.
        work = asyncio.create_task(self._stop(reason, interrupt_recording=interrupt_recording, all_scopes=all_scopes))
        try:
            return await asyncio.shield(work)
        except asyncio.CancelledError:
            await work
            raise

    async def _stop(self, reason: str, *, interrupt_recording: bool, all_scopes: bool = False) -> dict:
        self.armed = False
        self.owner = None
        self.keyboard_paused = False
        self.robot_owner = None
        self.robot_owners.clear()
        self.mapper.reset()
        self.last_action = None
        try:
            method = "stop_all" if all_scopes and hasattr(self.robot, "stop_all") else "stop"
            if not all_scopes and isinstance(self.robot, RemoteRobot) and self.robot.scope != "all":
                method = "release"
            self.last_feedback = await self.call(method)
        except Exception as exc:  # noqa: BLE001 -- any SDK failure leaves stop unconfirmed
            self.last_feedback = {"stop_confirmed": False, "error": str(exc)}
            self.error = f"stop unconfirmed: {exc}"
        self.stop_unconfirmed = (
            self.last_feedback.get("control_owned") is not False
            and self.last_feedback.get("stop_confirmed") is not True
        )
        self.stop_unconfirmed |= bool(getattr(self.robot, "stop_unconfirmed", False))
        if interrupt_recording and self.recorder.active:
            try:
                await asyncio.to_thread(self.recorder.finish, success=None, reason=reason)
            except Exception as exc:  # noqa: BLE001 -- retain recording failure in live status
                self.error = f"recording interrupted: {exc}"
        return self.last_feedback

    async def stop_scope(self, scope: str) -> dict:
        """Caller holds control_lock; finish even if the HTTP request disconnects."""

        async def finish():
            self.robot_owners.pop(scope, None)
            result = await self.call("stop", scope)
            self.last_feedback = result
            self.stop_unconfirmed = self.stop_unconfirmed or result.get("stop_confirmed") is not True
            return result

        work = asyncio.create_task(finish())
        try:
            return await asyncio.shield(work)
        except asyncio.CancelledError:
            await work
            raise

    async def refresh_robot_owners(self) -> dict:
        active = await self.call("control_state") if hasattr(self.robot, "control_state") else {}
        self.robot_owners = {s: o for s, o in self.robot_owners.items() if active.get(s)}
        if not self.robot.armed:
            self.robot_owner = None
        return active

    def validate_observation(self) -> None:
        if self.stop_unconfirmed:
            raise RuntimeError("previous stop unconfirmed; retry Stop and verify before re-arming")
        if not self.connected or time.monotonic() - self.received_at > 0.5:
            raise RuntimeError("robot observation unavailable or stale")
        timestamps = self.observation.get("camera_timestamps_ns", {})
        source = self.observation.get("source_timestamp_ns")
        if self.robot.metadata.get("source") == "physical":
            if not source or not all(name in self.images and name in timestamps for name in CAMERAS):
                raise RuntimeError("three current camera frames required before motion")
            if any(
                not isinstance(timestamps.get(name), int) or abs(source - timestamps[name]) > 500_000_000
                for name in CAMERAS
            ):
                raise RuntimeError("robot camera frames are stale")

    async def tick(self) -> None:
        while not self.closing:
            began = time.monotonic()
            try:
                observation, images = await self.call("read")
                self.observation, self.images = observation, images
                if self.connection_error is not None:
                    if not self.stop_unconfirmed:
                        self.error = None
                    self.connection_error = None
                self.received_at = time.monotonic()
                self.observation["received_timestamp_ns"] = time.time_ns()
                self.connected = True
                self.sample_count += 1
                async with self.control_lock:
                    if self.armed:
                        if not self.robot.armed:
                            raise RuntimeError("robot-side control disabled; explicitly re-arm")
                        record = self.frames.get(self.owner)
                        keyboard = bool(record and isinstance(record[0], KeyboardFrame))
                        age = time.monotonic() - record[2] if record else float("inf")
                        # Movement expires at 350 ms as before. A keyboard
                        # session can briefly retain ownership while sending
                        # ZERO only, never refreshing an old movement command.
                        if not record or age > (2.0 if keyboard else 0.35):
                            raise RuntimeError("controller input timeout")
                        if keyboard and age > 0.35:
                            self.keyboard_paused = True
                        self.validate_observation()
                        if self.mapper.control_mode == "drive" and (reason := self.drive_unavailable_reason()):
                            raise RuntimeError(reason)
                        frame, raw_input, _ = record
                        now = time.monotonic()
                        if isinstance(frame, KeyboardFrame):
                            if self.mapper.control_mode != "drive":
                                raise RuntimeError("keyboard input is drive-only")
                            if reason := self.keyboard_drive_unavailable_reason():
                                raise RuntimeError(reason)
                            action = frame.action(self.mapper.config, **self.keyboard_speed(self.owner))
                            if self.keyboard_paused:
                                action = {"x.vel": 0.0, "theta.vel": 0.0}
                        else:
                            action = self.mapper.map(
                                frame,
                                observation["state"],
                                now - self.last_command_at,
                                self.robot.metadata.get("joint_limits"),
                            )
                        action = self.selected_action(action)
                        feedback = await self.call("command", action)
                        self.check_feedback(feedback)
                        self.mapper.accept_applied_action(feedback["applied_action"])
                        self.last_feedback = feedback
                        self.last_action = feedback.get("applied_action")
                        self.last_command_at = now
                        if self.recorder.active:
                            await self.record_frame(
                                observation=dict(observation),
                                action=self.last_action,
                                images=dict(images),
                                input_sample=raw_input,
                                feedback=feedback,
                            )
                    elif self.recorder.active:
                        await self.record_frame(
                            observation=dict(observation),
                            action=None,
                            images=dict(images),
                            feedback=None,
                        )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 -- hardware/codec/disk errors all require stop
                self.error = str(exc)
                if not self.observation or time.monotonic() - self.received_at > 0.5:
                    self.connected = False
                    self.images = {}
                    self.connection_error = self.error
                async with self.control_lock:
                    if self.armed or self.recorder.active or self.robot_owner or self.robot_owners:
                        await self.stop(f"fault: {exc}")
            # Status is small and lossy. A slow observer must not stall motion.
            for client in list(self.clients):
                if not client.closed:
                    try:
                        await asyncio.wait_for(client.send_json({"type": "status", **self.status(client)}), 0.03)
                    except (asyncio.TimeoutError, ConnectionError, RuntimeError):
                        # Abort the slow socket instead of waiting for a close
                        # handshake in the robot loop.
                        if client._req and client._req.transport:
                            client._req.transport.close()
            await asyncio.sleep(max(0 if self.connected else 0.5, 1 / self.fps - (time.monotonic() - began)))

    async def record_frame(self, *, observation: dict, **values: Any) -> None:
        source = observation.get("source_timestamp_ns")
        if not isinstance(source, int):
            raise TypeError("recording requires robot source timestamp")
        if self.last_recorded_source is not None:
            if source < self.last_recorded_source:
                raise RuntimeError("robot clock regressed during recording")
            if source == self.last_recorded_source:
                # HTTP can return the same cached AGX observation twice.
                # Don't invent another training sample or a new sample time.
                self.skipped_duplicate_samples += 1
                return
        await asyncio.to_thread(self.recorder.append, observation=observation, **values)
        self.last_recorded_source = source

    async def start(self) -> None:
        try:
            await self.call("connect")
            self.stop_unconfirmed |= bool(getattr(self.robot, "stop_unconfirmed", False))
            inherited = getattr(self.robot, "inherited_stop_feedback", None)
            if inherited is not None:
                self.last_feedback = inherited
        except Exception as exc:  # A remote reboot must not remove the operator UI.
            if self.robot.mode != "remote":
                raise
            self.connection_error = self.error = f"waiting for AGX: {exc}"
        self.task = asyncio.create_task(self.tick())

    def selected_action(self, action: dict) -> dict:
        if self.robot.metadata.get("source") != "physical":
            return action
        if self.mapper.control_mode == "drive":
            # Keep existing servo hold targets; driving sends no arm or
            # gripper goals at all. Base permission is checked before mapping.
            return {name: action[name] for name in ("x.vel", "theta.vel")}
        enabled = self.robot.metadata.get("enabled_arms", ["left", "right"])
        return {
            name: value
            for name, value in action.items()
            if any(name.startswith(side + "_arm_") for side in enabled)
            or (
                name in ("x.vel", "theta.vel")
                and self.robot.metadata.get("enable_base", False)
                and getattr(self.robot, "scope", "all") != "arms"
            )
        }

    @staticmethod
    def check_feedback(feedback: dict) -> None:
        if feedback.get("accepted") is False or feedback.get("command_accepted") is False:
            raise RuntimeError("command refused: " + "; ".join(feedback.get("errors", [])))
        if not isinstance(feedback.get("applied_action"), dict) or not feedback["applied_action"]:
            raise RuntimeError("robot did not report the actual sent action")

    async def close(self) -> None:
        self.closing = True
        await self.stop_leader()
        if self.video_service is not None:
            await self.video_service.close()
        if self.task:
            # Let any in-flight SDK call complete before releasing devices.
            await self.task
        async with self.control_lock:
            if self.armed or self.robot.armed or self.recorder.active:
                await self.stop("server shutdown")
        for client in list(self.clients):
            await client.close()
        await self.call("close")


@web.middleware
async def auth(request: web.Request, handler):
    platform: Platform = request.app[PLATFORM]
    origin = request.headers.get("Origin")
    expected_origin = f"{request.scheme}://{request.host}"
    # Reject cross-site cookies and WebSocket hijacking, including mutating
    # session setup. Headless robot calls use bearer auth with no Origin.
    if origin and origin != expected_origin:
        return web.json_response({"error": "cross-origin request denied"}, status=403)
    protected = request.path.startswith(("/api/", "/robot/")) or request.path == "/ws"
    if protected and request.path != "/api/session":
        bearer = request.headers.get("Authorization", "")
        valid_bearer = secrets.compare_digest(bearer, "Bearer " + platform.token)
        valid_cookie = request.cookies.get("teleop_session") in platform.session_tokens
        # Numeric LAN host only: a foreign site's DNS rebinding must not turn
        # a matching Origin/Host pair into unauthenticated local control.
        valid_network = platform.browser_address_allowed(request.remote) and platform.browser_address_allowed(
            request.url.host
        )
        if request.path.startswith("/robot/"):
            valid_cookie = False
            valid_network = False
        if not valid_bearer and not valid_cookie and not valid_network:
            return web.json_response({"error": "authentication required"}, status=401)
    try:
        response = await handler(request)
    except (ValueError, KeyError, TypeError, RuntimeError) as exc:
        response = web.json_response({"error": str(exc)}, status=400)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "xr-spatial-tracking=(self)"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; "
        "connect-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'self'"
    )
    return response


async def session(request: web.Request) -> web.Response:
    platform = request.app[PLATFORM]
    data = await request.json()
    if not isinstance(data, dict):
        raise TypeError("JSON object required")
    if not platform.authenticate_browser(data.get("token")):
        return web.json_response({"error": "invalid or expired token/pairing code"}, status=401)
    if len(platform.session_tokens) >= 32:
        raise RuntimeError("too many browser sessions; restart gateway")
    cookie = secrets.token_urlsafe(32)
    platform.session_tokens.add(cookie)
    response = web.json_response({"authenticated": True})
    response.set_cookie(
        "teleop_session",
        cookie,
        httponly=True,
        secure=request.secure,
        samesite="Strict",
        max_age=86400,
    )
    return response


async def status(request: web.Request) -> web.Response:
    return web.json_response(request.app[PLATFORM].status())


async def video_offer(request: web.Request) -> web.Response:
    platform = request.app[PLATFORM]
    if platform.video_service is None:
        try:
            from .video import VideoService
        except ImportError as exc:
            raise RuntimeError("Mac video service needs the quest_teleop extra (aiortc)") from exc
        platform.video_service = VideoService(platform)
    return web.json_response(await platform.video_service.offer(await request.json()))


async def camera(request: web.Request) -> web.Response:
    platform = request.app[PLATFORM]
    name = request.match_info["name"]
    if not platform.connected or time.monotonic() - platform.received_at > 0.5 or name not in platform.images:
        return web.json_response({"error": "camera unavailable or stale"}, status=503)
    stamps = platform.observation.get("camera_timestamps_ns", {})
    source = platform.observation.get("source_timestamp_ns", 0)
    if not isinstance(stamps.get(name), int) or abs(source - stamps[name]) > 500_000_000:
        return web.json_response({"error": "camera frame stale"}, status=503)
    return web.Response(
        body=platform.images[name],
        content_type="image/jpeg",
        headers={"X-Camera-Timestamp-Ns": str(stamps[name])},
    )


async def websocket(request: web.Request) -> web.WebSocketResponse:
    platform = request.app[PLATFORM]
    if len(platform.clients) >= 4:
        raise RuntimeError("at most four browsers can connect")
    ws = web.WebSocketResponse(heartbeat=10, max_msg_size=32768)
    await ws.prepare(request)
    platform.clients.add(ws)
    clock = InputClock()
    try:
        await ws.send_json({"type": "status", **platform.status(ws)})
        async for message in ws:
            if message.type != WSMsgType.TEXT:
                continue
            try:
                data = json.loads(message.data)
                if not isinstance(data, dict):
                    raise TypeError("websocket JSON object required")
                kind = data.get("type")
                if kind in ("input", "keyboard_input"):
                    frame = KeyboardFrame.parse(data) if kind == "keyboard_input" else InputFrame.parse(data)
                    previous = platform.frames.get(ws)
                    if platform.owner is ws and previous and type(frame) is not type(previous[0]):
                        raise ValueError("Stop before changing input device")
                    try:
                        sampled_at = clock.accept(frame, time.monotonic())
                    except InputDelayed:
                        if not isinstance(frame, KeyboardFrame):
                            raise
                        # Drop queued input without replacing the last fresh
                        # sample or renewing its lease. Only the owner pauses.
                        if platform.owner is ws:
                            platform.keyboard_paused = True
                            platform.delayed_keyboard_packets += 1
                        continue
                    platform.frames[ws] = (
                        frame,
                        data,
                        (sampled_at if isinstance(frame, KeyboardFrame) else time.monotonic()),
                    )
                    if platform.owner is ws:
                        if isinstance(frame, KeyboardFrame):
                            if not frame.ready:
                                raise ValueError("keyboard focus or video lost")
                            if frame.neutral:
                                platform.keyboard_paused = False
                        elif not all(c.tracked for c in frame.controllers.values()):
                            raise ValueError("controller tracking lost")
                elif kind == "set_control_mode":
                    async with platform.control_lock:
                        platform.set_control_mode(data.get("mode"))
                    await ws.send_json({"type": "status", **platform.status(ws)})
                elif kind == "arm":
                    async with platform.control_lock:
                        if platform.owner is not None or platform.robot_owner or platform.robot_owners:
                            raise RuntimeError("control already owned; stop before arming another browser")
                        frame_info = platform.frames.get(ws)
                        activation = data.get("activation", "neutral")
                        if activation not in ("neutral", "grip", "keyboard"):
                            raise ValueError("unknown control activation")
                        if activation == "keyboard":
                            if platform.mapper.control_mode != "drive":
                                raise RuntimeError("keyboard activation is drive-only")
                            if reason := platform.keyboard_drive_unavailable_reason():
                                raise RuntimeError(reason)
                            if not frame_info or not isinstance(frame_info[0], KeyboardFrame):
                                raise RuntimeError("fresh keyboard input required")
                        elif frame_info and isinstance(frame_info[0], KeyboardFrame):
                            raise RuntimeError("keyboard input requires keyboard activation")
                        if (
                            not frame_info
                            or time.monotonic() - frame_info[2] > 0.35
                            or not (
                                frame_info[0].grip_activation_ready if activation == "grip" else frame_info[0].neutral
                            )
                        ):
                            raise RuntimeError(
                                "fresh neutral input required; release WASD / grips, triggers and sticks"
                            )
                        platform.validate_observation()
                        if platform.mapper.control_mode == "drive" and (reason := platform.drive_unavailable_reason()):
                            raise RuntimeError(reason)
                        result = await platform.call("arm")
                        if result.get("armed") is not True:
                            raise RuntimeError("cannot arm: " + "; ".join(result.get("errors", [])))
                        platform.owner, platform.armed = ws, True
                        platform.keyboard_paused = False
                        platform.mapper.reset()
                        platform.last_command_at = time.monotonic()
                        platform.error = None
                elif kind == "set_drive_speed":
                    try:
                        async with platform.control_lock:
                            speed = platform.set_keyboard_speed(ws, data)
                        await ws.send_json({"type": "drive_speed_result", "ok": True, "speed": speed})
                    except (ValueError, TypeError, RuntimeError) as exc:
                        # A refused setting is not a broken controller packet.
                        await ws.send_json({"type": "drive_speed_result", "ok": False, "error": str(exc)})
                elif kind == "leader_start":
                    try:
                        await platform.start_leader(ws)
                    except RuntimeError as exc:
                        await ws.send_json({"type": "leader_error", "error": str(exc)})
                        continue
                    await ws.send_json({"type": "status", **platform.status(ws)})
                elif kind == "leader_stop":
                    try:
                        await platform.stop_leader()
                    except RuntimeError as exc:
                        await ws.send_json({"type": "leader_error", "error": str(exc)})
                        continue
                    await ws.send_json({"type": "status", **platform.status(ws)})
                elif kind in ("stop", "stop_all"):
                    # An authenticated observer can stop, but never acquire
                    # control simply by sending input.
                    if kind == "stop_all":
                        await platform.stop_leader()
                    async with platform.control_lock:
                        await platform.stop("operator stop", interrupt_recording=True, all_scopes=kind == "stop_all")
                else:
                    raise ValueError("unknown websocket message")
            except (ValueError, TypeError, KeyError, RuntimeError) as exc:
                if platform.owner is None or platform.owner is ws:
                    platform.error = str(exc)
                if platform.owner is ws:
                    async with platform.control_lock:
                        await platform.stop(f"input error: {exc}")
                await ws.send_json({"type": "error", "error": str(exc)})
    finally:
        platform.clients.discard(ws)
        platform.frames.pop(ws, None)
        platform.keyboard_speeds.pop(ws, None)
        if platform.leader_owner is ws:
            await platform.stop_leader()
        if platform.owner is ws:
            async with platform.control_lock:
                await platform.stop("controller browser disconnected")
    return ws


async def record_start(request: web.Request) -> web.Response:
    platform = request.app[PLATFORM]
    data = await request.json()
    async with platform.control_lock:
        platform.validate_observation() if platform.armed else None
        metadata = {
            **platform.robot.metadata,
            "fps": platform.fps,
            "max_skew_ms": 100,
            "mode": platform.robot.mode,
            "operator_mode": "quest_webxr",
            "control_mode": platform.mapper.control_mode,
            "collection_mode": "teleoperation" if platform.armed else "observation_only",
            "max_gap_ms": 150,
            "trainable": platform.armed and platform.mapper.control_mode == "arms",
        }
        path = await asyncio.to_thread(platform.recorder.start, data.get("task", ""), metadata)
        platform.last_recorded_source = None
        platform.skipped_duplicate_samples = 0
        platform.last_recording_path = str(path)
    return web.json_response({"recording": True, "path": str(path)})


async def record_stop(request: web.Request) -> web.Response:
    platform = request.app[PLATFORM]
    data = await request.json()
    success = data.get("success")
    if success is not None and type(success) is not bool:
        raise ValueError("success must be true, false, or null")
    async with platform.control_lock:
        result = await asyncio.to_thread(platform.recorder.finish, success=success, reason="operator")
    return web.json_response(result)


async def robot_endpoint(request: web.Request) -> web.Response:
    platform = request.app[PLATFORM]
    endpoint = request.match_info["endpoint"]
    if not platform.robot_api:
        raise web.HTTPNotFound()
    scope = request.headers.get("X-Teleop-Scope", "all")
    owner = request.headers.get("X-Teleop-Owner")
    if scope not in ("all", "arms", "base"):
        raise ValueError("invalid control scope")
    if scope != "all" and scope not in platform.robot.metadata.get("control_scopes", []):
        raise RuntimeError("robot does not support independent control scopes")
    if request.method == "GET":
        if endpoint == "diagnostics":
            if platform.robot.mode != "hardware" or not callable(
                getattr(platform.robot, "read_motor_diagnostics", None)
            ):
                raise web.HTTPNotFound()
            if list(request.query) != ["motor"]:
                raise ValueError("diagnostics require exactly one motor query parameter")
            async with platform.control_lock:
                if platform.owner is not None or platform.robot_owner or platform.robot_owners:
                    raise web.HTTPConflict(text="diagnostics require unowned, inactive control scopes")
                result = await platform.call("read_motor_diagnostics", request.query["motor"])
            return web.json_response(result)
        if endpoint == "status":
            # Never take the synchronous SDK lock on the event-loop thread.
            async with platform.control_lock:
                safety = await platform.call("safety_state") if hasattr(platform.robot, "safety_state") else None
                return web.json_response(
                    {**platform.status(), "metadata": platform.robot.metadata, "hardware_safety": safety}
                )
        if endpoint == "observe":
            if not platform.connected or time.monotonic() - platform.received_at > 0.5:
                raise RuntimeError("robot observation stale")
            async with platform.control_lock:
                active = await platform.refresh_robot_owners()
                owned = (
                    platform.robot_owner == owner and bool(owner) and platform.robot.armed
                    if scope == "all"
                    else platform.robot_owners.get(scope) == owner and bool(owner)
                )
            observation = dict(platform.observation)
            queued_age = max(0, int((time.monotonic() - platform.received_at) * 1e9))
            observation["camera_ages_ns"] = {
                name: age + queued_age
                for name, age in observation.get("camera_ages_ns", {}).items()
                if isinstance(age, int) and not isinstance(age, bool) and age >= 0
            }
            state_age = observation.get("state_age_ns")
            if isinstance(state_age, int) and not isinstance(state_age, bool) and state_age >= 0:
                observation["state_age_ns"] = state_age + max(0, int((time.monotonic() - platform.received_at) * 1e9))
            return web.json_response(
                {
                    "observation": {
                        **observation,
                        "armed": platform.robot.armed,
                        "control_state": active,
                        "control_owned": owned,
                    },
                    "images": {k: base64.b64encode(v).decode() for k, v in platform.images.items()},
                }
            )
    elif request.method == "POST":
        if not owner or len(owner) > 128:
            raise ValueError("robot client owner identifier required")
        async with platform.control_lock:
            await platform.refresh_robot_owners()
            if endpoint == "arm":
                if (
                    platform.owner is not None
                    or platform.robot_owner
                    or (platform.robot_owners if scope == "all" else scope in platform.robot_owners)
                ):
                    raise RuntimeError("robot already controlled")
                platform.validate_observation()
                try:
                    result = await platform.call("arm", *(() if scope == "all" else (scope,)))
                except asyncio.CancelledError:
                    if scope == "all":
                        await platform.stop("arm request cancelled")
                    else:
                        await platform.stop_scope(scope)
                    raise
                if result.get("armed") is True:
                    if scope == "all":
                        platform.robot_owner = owner
                    else:
                        platform.robot_owners[scope] = owner
                    platform.error = None
                return web.json_response(result)
            if endpoint == "release":
                if scope == "all":
                    raise ValueError("owner-scoped release requires arms or base scope")
                if platform.robot_owners.get(scope) != owner:
                    return web.json_response({"released": False, "control_owned": False})
                return web.json_response({**await platform.stop_scope(scope), "released": True, "control_owned": True})
            if endpoint in ("stop", "stop_all"):
                if scope == "all" or endpoint == "stop_all":
                    return web.json_response(await platform.stop("Mac requested whole-robot stop", all_scopes=True))
                if platform.robot_owner or platform.owner is not None:
                    raise RuntimeError("legacy whole-robot controller active; use stop_all to stop it")
                return web.json_response(await platform.stop_scope(scope))
            if endpoint == "command":
                lease = platform.robot_owner if scope == "all" else platform.robot_owners.get(scope)
                if lease != owner or not platform.robot.armed:
                    raise RuntimeError("robot client does not own control")
                data = await request.json()
                try:
                    action = data["action"]
                    if not isinstance(action, dict) or not action:
                        raise ValueError("action must be a non-empty mapping")
                    if scope == "base":
                        base_keys = {"x.vel", "theta.vel"}
                        head = platform.robot.metadata.get("head_tilt") or {}
                        allowed_keys = base_keys | ({head["joint"]} if head.get("joint") else set())
                        if not base_keys <= set(action) <= allowed_keys:
                            raise ValueError("base scope accepts velocities and the configured camera tilt only")
                    if scope == "arms" and not set(action) <= set(JOINT_NAMES):
                        raise ValueError("arms scope accepts only arm joints")
                    result = await platform.call("command", data["action"])
                    platform.check_feedback(result)
                except Exception:
                    if scope == "all":
                        await platform.stop("robot command error")
                    else:
                        await platform.stop_scope(scope)
                    raise
                platform.last_feedback = result
                return web.json_response(result)
    raise web.HTTPNotFound()


def create_app(platform: Platform) -> web.Application:
    app = web.Application(middlewares=[auth], client_max_size=65536)
    app[PLATFORM] = platform
    assets = Path(__file__).with_name("web")

    async def index(request):
        return web.FileResponse(assets / "index.html")

    async def drive(request):
        return web.FileResponse(assets / "drive.html")

    app.router.add_get("/", index)
    app.router.add_get("/drive", drive)
    app.router.add_post("/api/session", session)
    app.router.add_get("/api/status", status)
    app.router.add_post("/api/webrtc/offer", video_offer)
    app.router.add_get("/api/cameras/{name}.jpg", camera)
    app.router.add_get("/ws", websocket)
    app.router.add_post("/api/record/start", record_start)
    app.router.add_post("/api/record/stop", record_stop)
    app.router.add_get("/robot/{endpoint}", robot_endpoint)
    app.router.add_post("/robot/{endpoint}", robot_endpoint)
    if assets.exists():
        app.router.add_static("/static/", assets, show_index=False)

    async def lifecycle(app):
        await platform.start()
        try:
            yield
        finally:
            await platform.close()

    app.cleanup_ctx.append(lifecycle)
    return app
