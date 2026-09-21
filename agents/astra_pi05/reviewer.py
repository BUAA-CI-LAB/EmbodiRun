"""Isolated GPT-6 Astra review for one software proposal packet.

The reviewer is an upper-layer decision helper.  It only receives a detached
packet and staged image paths; it never owns a Deploy client, a camera, a
robot, or an SSH transport.  Tests inject ``command_runner`` so the real Astra
CLI is never needed for local verification.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import shutil
import signal
import struct
import tempfile
import time
import zlib
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision import (
    ACTION_DIM,
    HORIZON,
    MAX_CORRECTIONS,
    MAX_PREFIX,
    validate_decision,
)

MODEL = "gpt-6-astra"
REASONING_EFFORTS = {"minimal", "low", "medium", "high", "xhigh", "max", "ultra"}

_WAYPOINT_KEYS = {
    "reach_m",
    "height_m",
    "pan_deg",
    "wrist_flex_deg",
    "wrist_roll_deg",
    "gripper",
}

DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "proposal_id",
        "observation_id",
        "decision",
        "execute_steps",
        "corrections",
        "reason",
    ],
    "properties": {
        "proposal_id": {"type": "string"},
        "observation_id": {"type": "string"},
        "decision": {
            "type": "string",
            "enum": ["execute_prefix", "correct", "hold"],
        },
        "execute_steps": {
            "type": "integer",
            "minimum": 0,
            "maximum": MAX_PREFIX,
        },
        "corrections": {
            "type": "array",
            "maxItems": MAX_CORRECTIONS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["left", "right", "frame", "units", "duration_s"],
                "properties": {
                    "left": {"anyOf": [{"$ref": "#/$defs/waypoint"}, {"type": "null"}]},
                    "right": {"anyOf": [{"$ref": "#/$defs/waypoint"}, {"type": "null"}]},
                    "frame": {
                        "type": "string",
                        "const": "so101_shoulder_plane",
                    },
                    "units": {"type": "string", "const": "m_deg"},
                    "duration_s": {"type": "number", "exclusiveMinimum": 0},
                },
            },
        },
        "reason": {"type": "string", "minLength": 1},
    },
    "$defs": {
        "waypoint": {
            "type": "object",
            "additionalProperties": False,
            "required": sorted(_WAYPOINT_KEYS),
            "properties": {key: {"type": "number"} for key in _WAYPOINT_KEYS},
        }
    },
}

CommandRunner = Callable[[Sequence[str], Path], Awaitable[tuple[int, str, str]]]


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _validate_packet(
    packet: Mapping[str, Any],
) -> tuple[str, str, list[Path], dict[str, Any]]:
    """Validate a detached review packet before starting a model process."""

    observation = _mapping(packet.get("observation"), "observation")
    proposal = _mapping(packet.get("proposal"), "proposal")
    observation_id = observation.get("observation_id")
    proposal_id = proposal.get("proposal_id")
    if not isinstance(observation_id, str) or not observation_id.strip():
        raise ValueError("observation.observation_id must be a non-empty string")
    if not isinstance(proposal_id, str) or not proposal_id.strip():
        raise ValueError("proposal.proposal_id must be a non-empty string")
    if proposal.get("observation_id") != observation_id:
        raise ValueError("proposal.observation_id must match observation.observation_id")

    max_prefix_steps = packet.get("max_prefix_steps", MAX_PREFIX)
    if (
        isinstance(max_prefix_steps, bool)
        or not isinstance(max_prefix_steps, int)
        or not 1 <= max_prefix_steps <= MAX_PREFIX
    ):
        raise ValueError("max_prefix_steps must be an integer from 1 to 15")

    metadata = _mapping(proposal.get("metadata"), "proposal.metadata")
    expected = {
        "action_semantics": "biso101_so101_v1",
        "action_layout": "left6_right6",
        "action_encoding": "absolute",
        # Astra's packet prompt describes physical degree values.  A producer
        # using range_m100_100 must convert explicitly before creating it.
        "joint_position_unit": "degrees",
        "action_dim": ACTION_DIM,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"proposal.metadata.{key} must be {value!r}")
    if "horizon" in metadata and metadata["horizon"] != HORIZON:
        raise ValueError(f"proposal.metadata.horizon must be {HORIZON!r}")

    actions = proposal.get("actions")
    if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)) or len(actions) != HORIZON:
        raise ValueError("proposal.actions must be a 50-step sequence")
    for index, row in enumerate(actions):
        if (
            not isinstance(row, Sequence)
            or isinstance(row, (str, bytes))
            or len(row) != ACTION_DIM
            or not all(_finite_number(value) for value in row)
        ):
            raise ValueError(f"proposal.actions[{index}] must contain 12 finite values")

    state = observation.get("state")
    if (
        not isinstance(state, Sequence)
        or isinstance(state, (str, bytes))
        or len(state) != ACTION_DIM
        or not all(_finite_number(value) for value in state)
    ):
        raise ValueError("observation.state must contain 12 finite values")

    images = _mapping(observation.get("images"), "observation.images")
    required = {"front", "left_wrist", "right_wrist"}
    if set(images) != required:
        raise ValueError("observation.images must contain exactly front, left_wrist, right_wrist")
    paths: list[Path] = []
    for name, value in images.items():
        if not isinstance(name, str) or not isinstance(value, str) or not os.path.isabs(value):
            raise ValueError("observation.images must map names to absolute paths")
        path = Path(value)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"image is missing or empty: {value}")
        paths.append(path)

    preview = _mapping(proposal.get("trajectory_preview"), "proposal.trajectory_preview")
    if preview.get("frame") != "so101_shoulder_plane":
        raise ValueError("trajectory_preview.frame must be so101_shoulder_plane")
    for side in ("left", "right"):
        points = preview.get(side)
        if (
            not isinstance(points, Sequence)
            or isinstance(points, (str, bytes))
            or len(points) != HORIZON
            or not all(
                isinstance(point, Sequence)
                and not isinstance(point, (str, bytes))
                and len(point) == 2
                and all(_finite_number(value) for value in point)
                for point in points
            )
        ):
            raise ValueError(f"trajectory_preview.{side} must contain 50 finite points")
    return proposal_id, observation_id, paths, dict(packet)


def _validate_decision(raw: Any, proposal_id: str, observation_id: str) -> dict[str, Any]:
    """Compatibility helper for callers that validate fake reviewer output."""

    return validate_decision(raw, proposal_id=proposal_id, observation_id=observation_id)


def _trajectory_png(preview: Mapping[str, Any], target: Path) -> None:
    """Render the nominal 2D review aid using Pillow when available."""

    try:
        from PIL import Image, ImageDraw
    except ImportError:  # pragma: no cover - exercised on dependency-light hosts
        _trajectory_png_stdlib(preview, target)
        return
    image = Image.new("RGB", (900, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((45, 25, 875, 455), outline="black")
    points = [point for side in ("left", "right") for point in preview[side]]
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    xlo, xhi = min(xs), max(xs)
    ylo, yhi = min(ys), max(ys)
    xscale = 780.0 / (xhi - xlo) if xhi != xlo else 1.0
    yscale = 380.0 / (yhi - ylo) if yhi != ylo else 1.0
    colors = ((30, 95, 190), (210, 75, 45))
    for arm, side in enumerate(("left", "right")):
        scaled = [
            (
                int(70 + (float(point[0]) - xlo) * xscale),
                int(420 - (float(point[1]) - ylo) * yscale),
            )
            for point in preview[side]
        ]
        draw.line(scaled, fill=colors[arm], width=3)
        stride = max(1, len(scaled) // 8)
        for step, point in enumerate(scaled[::stride]):
            draw.text((point[0] + 3, point[1] + 3), str(step), fill=colors[arm])
    draw.text((55, 35), "BiSO101 nominal planar trajectory", fill="black")
    draw.text(
        (55, 465),
        f"blue left · red right · {preview.get('source', 'source unspecified')} · no collision prediction",
        fill="black",
    )
    image.save(target, format="PNG")


def _trajectory_png_stdlib(preview: Mapping[str, Any], target: Path) -> None:
    """Write a small RGB PNG without making Pillow a core dependency."""

    width, height = 900, 500
    pixels = bytearray([255, 255, 255] * width * height)

    def put(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            offset = (y * width + x) * 3
            pixels[offset : offset + 3] = bytes(color)

    def line(a: tuple[int, int], b: tuple[int, int], color: tuple[int, int, int]) -> None:
        x0, y0 = a
        x1, y1 = b
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        while True:
            put(x0, y0, color)
            if (x0, y0) == (x1, y1):
                return
            twice = 2 * error
            if twice >= dy:
                error += dy
                x0 += sx
            if twice <= dx:
                error += dx
                y0 += sy

    points = [point for side in ("left", "right") for point in preview[side]]
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    xlo, xhi = min(xs), max(xs)
    ylo, yhi = min(ys), max(ys)
    xscale = 780.0 / (xhi - xlo) if xhi != xlo else 1.0
    yscale = 380.0 / (yhi - ylo) if yhi != ylo else 1.0
    for side, color in (("left", (30, 95, 190)), ("right", (210, 75, 45))):
        scaled = [
            (
                int(70 + (float(point[0]) - xlo) * xscale),
                int(420 - (float(point[1]) - ylo) * yscale),
            )
            for point in preview[side]
        ]
        for first, second in zip(scaled, scaled[1:]):
            line(first, second, color)
    raw = b"".join(b"\x00" + bytes(pixels[row * width * 3 : (row + 1) * width * 3]) for row in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    target.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=6))
        + chunk(b"IEND", b"")
    )


async def _default_runner(argv: Sequence[str], cwd: Path) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = await process.communicate()
    except asyncio.CancelledError:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), timeout=2.0)
        except (asyncio.TimeoutError, ProcessLookupError):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
        raise
    return (
        process.returncode or 0,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


def _failure_message(stdout: str, stderr: str) -> str:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            for key in ("error", "message", "detail", "reason"):
                message = value.get(key)
                if isinstance(message, str) and message.strip():
                    return message.strip()[-1000:]
    return stderr.strip()[-1000:] or "Astra exec failed without a diagnostic"


class AstraCodexReviewer:
    """Run one isolated, structured ``gpt-6-astra`` review.

    ``packet`` must contain detached image paths.  Obtaining those paths is an
    integration concern; a Deploy caller may use its public ``media`` route,
    while a replay caller may use fixture files.  This class never captures a
    camera or sends an action.
    """

    decision_schema = DECISION_SCHEMA

    def _validate_packet(self, packet: Mapping[str, Any]) -> tuple[str, str, list[Path], dict[str, Any]]:
        return _validate_packet(packet)

    def _render_preview(
        self,
        proposal: Mapping[str, Any],
        target: Path,
        _observation: Mapping[str, Any],
        _max_prefix_steps: int,
    ) -> None:
        _trajectory_png(proposal["trajectory_preview"], target)

    def _schema(self) -> dict[str, Any]:
        return self.decision_schema

    def _validate_result(
        self,
        raw: Any,
        proposal_id: str,
        observation_id: str,
        max_prefix_steps: int,
    ) -> dict[str, Any]:
        result = _validate_decision(raw, proposal_id, observation_id)
        if result["decision"] == "execute_prefix" and result["execute_steps"] > max_prefix_steps:
            raise ValueError("Astra execute_prefix exceeds max_prefix_steps")
        return result

    def __init__(
        self,
        *,
        codex_bin: str = "codex",
        timeout_s: float = 90.0,
        reasoning_effort: str | None = None,
        command_runner: CommandRunner | None = None,
    ) -> None:
        if not _finite_number(timeout_s) or float(timeout_s) <= 0:
            raise ValueError("timeout_s must be positive and finite")
        if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {sorted(REASONING_EFFORTS)}")
        self.codex_bin = codex_bin
        self.timeout_s = float(timeout_s)
        self.reasoning_effort = reasoning_effort
        self.command_runner = command_runner or _default_runner
        self.last_timing: dict[str, float] = {}

    async def review(self, packet: Mapping[str, Any]) -> Mapping[str, Any]:
        started = time.monotonic()
        max_prefix = packet.get("max_prefix_steps", MAX_PREFIX)
        if isinstance(max_prefix, bool) or not isinstance(max_prefix, int) or not 1 <= max_prefix <= MAX_PREFIX:
            raise ValueError("max_prefix_steps must be an integer from 1 to 15")
        proposal_id, observation_id, image_paths, copied_packet = self._validate_packet(packet)
        with tempfile.TemporaryDirectory(prefix="astra-pi05-review-") as work:
            root = Path(work)
            staged: list[Path] = []
            image_names = list(copied_packet["observation"]["images"])
            for index, source in enumerate(image_paths):
                destination = root / f"camera-{index}{source.suffix.lower() or '.png'}"
                shutil.copy2(source, destination)
                staged.append(destination)
            trajectory = root / "trajectory.png"
            self._render_preview(
                copied_packet["proposal"],
                trajectory,
                copied_packet["observation"],
                max_prefix,
            )
            schema = root / "decision.schema.json"
            schema.write_text(json.dumps(self._schema()), encoding="utf-8")
            output = root / "decision.json"
            staged_mapping = dict(zip(image_names, (path.name for path in staged)))
            prompt_packet = dict(copied_packet)
            prompt_observation = dict(prompt_packet["observation"])
            prompt_observation["images"] = staged_mapping
            prompt_packet["observation"] = prompt_observation
            prompt_packet["max_prefix_steps"] = max_prefix
            argv = [
                self.codex_bin,
                "exec",
                "--ignore-user-config",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "--disable",
                "shell_tool",
                "--disable",
                "apps",
                "--disable",
                "plugins",
                "--disable",
                "hooks",
                "--disable",
                "memories",
                "--disable",
                "multi_agent",
                "--disable",
                "browser_use",
                "--disable",
                "computer_use",
                "-c",
                'web_search="disabled"',
            ]
            if self.reasoning_effort:
                argv.extend(("-c", f'model_reasoning_effort="{self.reasoning_effort}"'))
            argv.extend(
                (
                    "--model",
                    MODEL,
                    "--output-schema",
                    str(schema),
                    "--output-last-message",
                    str(output),
                    "--json",
                    "-C",
                    str(root),
                )
            )
            for image in (*staged, trajectory):
                argv.extend(("--image", str(image)))
            argv.extend(("--", self._prompt(prompt_packet, staged_mapping | {"trajectory": trajectory.name})))
            try:
                code, stdout, stderr = await asyncio.wait_for(self.command_runner(argv, root), timeout=self.timeout_s)
            except asyncio.TimeoutError as error:
                raise TimeoutError("Astra reviewer timed out") from error
            if code != 0:
                raise RuntimeError(f"Astra reviewer failed ({code}): {_failure_message(stdout, stderr)}")
            if not output.is_file():
                raise ValueError("Astra returned no output-last-message JSON")
            try:
                raw = json.loads(output.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ValueError("Astra returned malformed JSON") from error
            result = self._validate_result(raw, proposal_id, observation_id, max_prefix)
        self.last_timing = {"total_s": time.monotonic() - started}
        return result

    @staticmethod
    def _prompt(packet: Mapping[str, Any], image_names: Mapping[str, str]) -> str:
        return (
            "You are a constrained visual reviewer for a BiSO101 mobile manipulator. "
            "Review the attached camera images and nominal trajectory preview. "
            "The proposal is exactly one 50-step, 12D absolute-degree sequence with "
            "left6_right6 layout. Return only schema-conforming JSON: execute a "
            "prefix of 1..15 rows, give 1..5 constrained planar correction "
            "waypoints, or hold with zero actions when evidence is insufficient. "
            "Do not emit commands, tool calls, motor values, or task-success claims. "
            "The Deploy client separately validates freshness, reachability, limits, "
            "and authority. Corrections use frame so101_shoulder_plane and units m_deg. "
            "Attachment mapping: "
            + json.dumps(dict(image_names))
            + "\nPACKET:\n"
            + json.dumps(packet, ensure_ascii=False, allow_nan=False)
        )


__all__ = [
    "AstraCodexReviewer",
    "ACTION_DIM",
    "DECISION_SCHEMA",
    "HORIZON",
    "MAX_CORRECTIONS",
    "MAX_PREFIX",
    "MODEL",
    "REASONING_EFFORTS",
    "_validate_decision",
]
