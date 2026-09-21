"""Explicit LightNav-0 local-segment teleoperation entry point.

This mode is for the configured two-wheel XLeRobot when metric localization is
not available.  It predicts from RGB, selects the first cumulative waypoint
with a forward or yaw component, sends one bounded command for a short interval,
stops through the existing RemoteRobot route, and predicts again.  It never
creates a world pose or odometry.

The robot is supplied either by an explicit ``module:function`` factory or by
the built-in HTTP route (``--robot-url`` plus ``--robot-token-env``).  Both
routes return an already connected and explicitly authorized robot before any
motion command is accepted; the built-in route connects after inference readiness
and arms only with ``--authorize-motion``.  A real camera source must include
``camera_status[camera].fresh`` and ``timestamp_ns``.  Those source camera
timestamps are kept separate from the local monotonic wheel-feedback receive
time and are never inferred from it.

Example factory shape (kept outside this repository so credentials and the
user's existing authorization flow remain local)::

    def build_robot():
        from embodirun.bindings.xlerobot.lightnav0 import build_remote_robot_from_env
        return build_remote_robot_from_env(
            "http://orin:8080", token_env="XLEROBOT_TOKEN", authorize_motion=True
        )

The repository now also has a concrete route, for example::

    XLEROBOT_TOKEN='...' python -m embodirun.bindings.xlerobot.lightnav0.cli \
      --robot-url http://orin:8080 --robot-token-env XLEROBOT_TOKEN \
      --authorize-motion --inference-url http://inference:8050 \
      --instruction 'go forward' --camera front

Without ``--authorize-motion`` the built-in route connects for inspection but
the loop refuses to command the base.  Construction of this module is
side-effect free until the selected factory is invoked after inference readiness.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import importlib
import inspect
import io
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from embodirun.bindings.xlerobot.lightnav0 import (
    LocalSegmentConfig,
    LocalSegmentExecutor,
    XLeRobotTeleopLocalSegmentBackend,
    XLeRobotTeleopRobot,
    build_remote_robot_from_env,
)

from .client import LightNav0Client, create_client


class TeleopFactoryError(RuntimeError):
    """The injected robot factory is missing or returned an unusable object."""


class MotionAuthorizationRequired(TeleopFactoryError):
    """The injected robot has not explicitly enabled base motion."""


class CameraFreshnessUnavailable(RuntimeError):
    """The source did not provide explicit fresh camera evidence."""


class PredictionStale(CameraFreshnessUnavailable):
    """Inference exceeded the local camera-to-command freshness bound."""


@dataclass(frozen=True, slots=True)
class TeleopRunConfig:
    """Bounded RGB closed-loop settings."""

    instruction: str
    camera: str
    session_id: str = "lightnav0-xlerobot-local-segment"
    max_steps: int = 80
    segment: LocalSegmentConfig = field(default_factory=LocalSegmentConfig)
    prediction_timeout_s: float = 0.5
    max_prediction_retries: int = 2

    def __post_init__(self) -> None:
        if not self.instruction.strip():
            raise ValueError("instruction must not be empty")
        if not self.camera.strip():
            raise ValueError("camera must not be empty")
        if not self.session_id.strip():
            raise ValueError("session_id must not be empty")
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or self.max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if (
            isinstance(self.prediction_timeout_s, bool)
            or not isinstance(self.prediction_timeout_s, (int, float))
            or not math.isfinite(float(self.prediction_timeout_s))
            or float(self.prediction_timeout_s) <= 0.0
        ):
            raise ValueError("prediction_timeout_s must be finite and positive")
        if (
            isinstance(self.max_prediction_retries, bool)
            or not isinstance(self.max_prediction_retries, int)
            or self.max_prediction_retries < 0
        ):
            raise ValueError("max_prediction_retries must be a non-negative integer")


def load_robot_factory(spec: str) -> Callable[[], Any]:
    """Load a user-owned zero-argument robot factory without importing an SDK."""

    if not isinstance(spec, str) or spec.count(":") != 1:
        raise TeleopFactoryError("robot factory must be written as module:function")
    module_name, function_name = spec.split(":", 1)
    if not module_name or not function_name:
        raise TeleopFactoryError("robot factory must be written as module:function")
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name, None)
    if not callable(factory):
        raise TeleopFactoryError(f"robot factory is not callable: {spec}")
    return factory


def resolve_robot_factory(args: argparse.Namespace) -> Callable[[], Any]:
    """Resolve custom or built-in factory without connecting during parsing."""

    custom = getattr(args, "robot_factory", None)
    url = getattr(args, "robot_url", None)
    if bool(custom) == bool(url):
        raise TeleopFactoryError("provide exactly one of --robot-factory or --robot-url")
    if custom:
        if getattr(args, "authorize_motion", False):
            raise TeleopFactoryError("--authorize-motion applies only to the built-in --robot-url route")
        return load_robot_factory(custom)
    token_env = getattr(args, "robot_token_env", "XLEROBOT_TOKEN")
    timeout_s = getattr(args, "robot_timeout_s", 2.0)
    scope = getattr(args, "robot_scope", "base")
    authorize_motion = bool(getattr(args, "authorize_motion", False))

    def build() -> Any:
        return build_remote_robot_from_env(
            url,
            token_env=token_env,
            timeout=timeout_s,
            scope=scope,
            authorize_motion=authorize_motion,
        )

    return build


def _json_safe(value: Any) -> Any:
    """Convert model/native scalar containers into strict JSON values.

    LightNav-0 providers commonly expose actions as a numpy array.  Keeping
    this conversion at the recording boundary avoids coupling the deploy app
    to numpy or to the provider's decode implementation.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("JSON output contains a non-finite float")
        return value
    if is_dataclass(value):
        return _json_safe(asdict(value))
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_safe(tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, memoryview)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _split_robot_read(raw: Any) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if isinstance(raw, tuple):
        if len(raw) != 2 or not isinstance(raw[0], Mapping) or not isinstance(raw[1], Mapping):
            raise CameraFreshnessUnavailable("robot.read() must return (observation, images)")
        return raw[0], raw[1]
    if not isinstance(raw, Mapping):
        raise CameraFreshnessUnavailable("robot.read() must return a mapping or (mapping, images)")
    images = raw.get("images")
    if not isinstance(images, Mapping):
        raise CameraFreshnessUnavailable("robot observation has no images mapping")
    return raw, images


def read_fresh_camera(
    robot: XLeRobotTeleopRobot,
    camera: str,
) -> tuple[Any, int, float]:
    """Read one RGB frame with source freshness evidence.

    The returned timestamp is the camera source timestamp.  Wheel feedback is
    read separately by ``XLeRobotTeleopLocalSegmentBackend`` and uses a local
    monotonic receive timestamp.
    """

    raw_observation, images = _split_robot_read(robot.read())
    status_map = raw_observation.get("camera_status")
    if not isinstance(status_map, Mapping):
        raise CameraFreshnessUnavailable("camera_status is absent; the source cannot prove a fresh camera frame")
    status = status_map.get(camera)
    if not isinstance(status, Mapping) or status.get("fresh") is not True:
        raise CameraFreshnessUnavailable(f"camera {camera!r} is stale or unavailable")
    timestamp_ns = status.get("timestamp_ns")
    if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int) or timestamp_ns <= 0:
        raise CameraFreshnessUnavailable(f"camera {camera!r} has no valid source timestamp")
    encoded = images.get(camera)
    if isinstance(encoded, str):
        try:
            encoded = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise CameraFreshnessUnavailable(f"camera {camera!r} payload is not valid base64") from exc
    if not isinstance(encoded, (bytes, bytearray, memoryview)):
        raise CameraFreshnessUnavailable(f"camera {camera!r} payload is not bytes")
    try:
        import numpy as np
        from PIL import Image

        with Image.open(io.BytesIO(bytes(encoded))) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except (ImportError, OSError, ValueError) as exc:
        raise CameraFreshnessUnavailable(f"camera {camera!r} payload is not a readable RGB image") from exc
    return rgb, timestamp_ns, float(timestamp_ns) / 1_000_000_000.0


def _read_fresh_camera_with_receive_time(
    robot: XLeRobotTeleopRobot,
    camera: str,
    *,
    clock: Callable[[], float],
) -> tuple[Any, int, float, float]:
    """Read camera source time and local receive time as separate values."""

    rgb, timestamp_ns, source_timestamp_s = read_fresh_camera(robot, camera)
    received_at_s = float(clock())
    if not math.isfinite(received_at_s):
        raise CameraFreshnessUnavailable("local camera receive clock is not finite")
    return rgb, timestamp_ns, source_timestamp_s, received_at_s


def _feedback_summary(feedback: Any) -> dict[str, Any]:
    return {
        "received_at_s": feedback.received_at_s,
        "state_timestamp_ns": feedback.state_timestamp_ns,
        "velocity": {
            "vx": feedback.velocity.vx,
            "vy": feedback.velocity.vy,
            "omega": feedback.velocity.omega,
        },
        "collision": feedback.collision,
    }


async def run_local_segment_loop(
    provider: LightNav0Client,
    robot: XLeRobotTeleopRobot,
    config: TeleopRunConfig,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Any] = asyncio.sleep,
) -> dict[str, Any]:
    """Run RGB -> one local command -> stop -> RGB closed-loop execution."""

    if not isinstance(robot, XLeRobotTeleopRobot):
        raise TeleopFactoryError("robot must implement read / command / stop / close")
    if getattr(robot, "armed", None) is not True:
        raise MotionAuthorizationRequired(
            "robot is not explicitly authorized; factory must connect and arm base motion"
        )
    if not callable(clock) or not callable(sleep):
        raise TypeError("clock and sleep must be callable")
    backend = XLeRobotTeleopLocalSegmentBackend(
        robot,
        max_linear_velocity_m_s=config.segment.max_linear_velocity_m_s,
        max_angular_velocity_rad_s=config.segment.max_angular_velocity_rad_s,
        clock=clock,
    )
    executor = LocalSegmentExecutor(backend, config.segment, clock=clock)
    records: list[dict[str, Any]] = []
    model_stop = False
    last_camera_timestamp_ns: int | None = None
    failure: BaseException | None = None
    initial_feedback = None
    parked_feedback = None
    result: dict[str, Any] | None = None
    try:
        await provider.reset_session(config.session_id)
        # Keep the explicit lease while proving initial measured stationarity;
        # do not infer a parked state from an arm flag or a prior command.
        initial_feedback = await asyncio.to_thread(executor.hold_zero)
        parked_feedback = initial_feedback
        for decision in range(config.max_steps):
            accepted_prediction = False
            for retry in range(config.max_prediction_retries + 1):
                rgb, camera_timestamp_ns, camera_timestamp_s, camera_received_at_s = (
                    _read_fresh_camera_with_receive_time(robot, config.camera, clock=clock)
                )
                if last_camera_timestamp_ns is not None and camera_timestamp_ns <= last_camera_timestamp_ns:
                    raise CameraFreshnessUnavailable("camera source timestamp repeated or moved backwards")
                last_camera_timestamp_ns = camera_timestamp_ns
                prediction = await provider.predict(
                    rgb,
                    instruction=config.instruction,
                    session_id=config.session_id,
                    timestamp_s=camera_timestamp_s,
                )
                # A long inference may outlive the control lease.  Check a
                # fresh owned state before deciding whether the prediction is
                # usable, including on stale-prediction retries.
                inference_feedback = await asyncio.to_thread(executor.check_fresh_feedback)
                prediction_age_s = float(clock()) - camera_received_at_s
                if (
                    not math.isfinite(prediction_age_s)
                    or prediction_age_s < -config.prediction_timeout_s
                    or prediction_age_s > config.prediction_timeout_s
                ):
                    # Every accepted segment is stopped before the next camera
                    # read.  Keep that parked state while retrying instead of
                    # issuing another physical stop, which would revoke the
                    # caller's base control lease on the inspected RemoteRobot.
                    records.append(
                        {
                            "decision": decision,
                            "retry": retry,
                            "camera_timestamp_ns": camera_timestamp_ns,
                            "camera_timestamp_basis": "source_camera_timestamp_ns",
                            "camera_received_at_s": camera_received_at_s,
                            "prediction_age_s": prediction_age_s,
                            "parked_feedback": _feedback_summary(parked_feedback),
                            "post_inference_feedback": _feedback_summary(inference_feedback),
                            "reason": "prediction-stale",
                            "metric_pose_available": False,
                        }
                    )
                    if retry < config.max_prediction_retries:
                        continue
                    raise PredictionStale(
                        f"prediction age {prediction_age_s:.3f}s exceeds {config.prediction_timeout_s:.3f}s"
                    )

                chunk = prediction.output
                metadata = dict(chunk.metadata or {})
                stop = metadata.get("stop")
                if type(stop) is not bool:
                    raise RuntimeError("LightNav-0 prediction is missing an explicit boolean stop")
                # Normalize recording only; the original provider output is
                # still passed to the trajectory validator/executor.
                json_waypoints = _json_safe(chunk.actions)
                json_raw_text = _json_safe(metadata.get("raw_text", ""))
                json_pointing = _json_safe(metadata.get("pointing"))
                # Parsing/normalizing must not consume the remaining camera to
                # command budget.  If it does, discard this prediction before
                # handing it to the executor and retry from a new RGB frame.
                prediction_age_s = float(clock()) - camera_received_at_s
                if (
                    not math.isfinite(prediction_age_s)
                    or prediction_age_s < -config.prediction_timeout_s
                    or prediction_age_s > config.prediction_timeout_s
                ):
                    records.append(
                        {
                            "decision": decision,
                            "retry": retry,
                            "camera_timestamp_ns": camera_timestamp_ns,
                            "camera_timestamp_basis": "source_camera_timestamp_ns",
                            "camera_received_at_s": camera_received_at_s,
                            "prediction_age_s": prediction_age_s,
                            "parked_feedback": _feedback_summary(parked_feedback),
                            "post_inference_feedback": _feedback_summary(inference_feedback),
                            "reason": "prediction-stale-before-command",
                            "metric_pose_available": False,
                        }
                    )
                    if retry < config.max_prediction_retries:
                        continue
                    raise PredictionStale(
                        f"prediction age {prediction_age_s:.3f}s exceeds {config.prediction_timeout_s:.3f}s"
                    )
                # Reuse the fresh ownership/velocity sample just checked.  A
                # second immediate read can be the concrete client's cached
                # 20ms sample and must never be mislabeled as fresh.
                step = executor.execute(
                    {"waypoints": chunk.actions, "stop": stop},
                    feedback=inference_feedback,
                )
                model_stop = stop
                records.append(
                    {
                        "decision": decision,
                        "retry": retry,
                        "camera_timestamp_ns": camera_timestamp_ns,
                        "camera_timestamp_basis": "source_camera_timestamp_ns",
                        "camera_received_at_s": camera_received_at_s,
                        "prediction_age_s": prediction_age_s,
                        "parked_feedback": _feedback_summary(parked_feedback),
                        "post_inference_feedback": _feedback_summary(inference_feedback),
                        "wheel_feedback_received_at_s": step.feedback.received_at_s,
                        "wheel_state_timestamp_ns": step.feedback.state_timestamp_ns,
                        "waypoint_index": step.waypoint_index,
                        "waypoints": json_waypoints,
                        "command": asdict(step.command),
                        "model": {
                            "stop": stop,
                            "raw_text": json_raw_text,
                            "pointing": json_pointing,
                        },
                        "reason": step.reason,
                        "metric_pose_available": False,
                    }
                )
                accepted_prediction = True
                if step.reason == "explicit-stop":
                    break
                await sleep(config.segment.duration_s)
                # Keep the lease between decisions: command zero and require a
                # fresh measured stationary sample.  Disarming is reserved for
                # an explicit model stop, final close, or an exception.
                parked_feedback = await asyncio.to_thread(executor.hold_zero)
                records[-1]["settled_feedback"] = _feedback_summary(parked_feedback)
                break
            if not accepted_prediction or model_stop:
                break
        result = {
            "mode": "xlerobot-local-segment",
            "camera": config.camera,
            "camera_timestamp_basis": "source_camera_timestamp_ns",
            "wheel_feedback_timestamp_basis": "local_monotonic_receive",
            "metric_pose_available": False,
            "model_stop": model_stop,
            "initial_stationary_feedback": _feedback_summary(initial_feedback),
            "decisions": len(records),
            "steps": records,
        }
        return result
    except BaseException as error:
        failure = error
        raise
    finally:
        close_failure: BaseException | None = None
        try:
            executor.close()
        except BaseException as close_error:  # noqa: BLE001 - attach concrete stop evidence
            if failure is None:
                failure = close_error
                close_failure = close_error
            else:
                failure.lightnav_close_error = f"{type(close_error).__name__}: {close_error}"
        finally:
            if failure is not None:
                failure.lightnav_stop_report = executor.last_stop_report
                failure.lightnav_partial_result = {
                    "mode": "xlerobot-local-segment",
                    "camera": config.camera,
                    "camera_timestamp_basis": "source_camera_timestamp_ns",
                    "wheel_feedback_timestamp_basis": "local_monotonic_receive",
                    "metric_pose_available": False,
                    "model_stop": model_stop,
                    "decisions": len(records),
                    "steps": _json_safe(records),
                }
            elif result is not None:
                result["final_stop_report"] = executor.last_stop_report
            if close_failure is not None:
                raise close_failure


async def _run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    output = _prepare_output_path(args.output)
    factory: Callable[[], Any] | None = None
    provider: LightNav0Client | None = None
    robot = None
    completed = False
    result: dict[str, Any] | None = None
    failure: BaseException | None = None
    try:
        factory = resolve_robot_factory(args)
        config = TeleopRunConfig(
            instruction=args.instruction,
            camera=args.camera,
            session_id=args.session_id,
            max_steps=args.max_steps,
            segment=LocalSegmentConfig(
                duration_s=args.duration_s,
                max_linear_velocity_m_s=args.max_linear_velocity,
                max_angular_velocity_rad_s=args.max_angular_velocity,
            ),
            prediction_timeout_s=args.prediction_timeout_s,
            max_prediction_retries=args.max_prediction_retries,
        )
        provider = create_client(args)
        # Check the inference service before acquiring a robot control lease.
        await provider.start()
        assert factory is not None
        robot = factory()
        if inspect.isawaitable(robot):
            robot = await robot
        result = await run_local_segment_loop(provider, robot, config)
        completed = True
        return result
    except BaseException as error:
        failure = error
        result = {
            "status": "error",
            "mode": "xlerobot-local-segment",
            "error": {"type": type(error).__name__, "message": str(error)},
            "partial_result": getattr(error, "lightnav_partial_result", None),
            "stop_report": getattr(error, "lightnav_stop_report", None),
        }
        raise
    finally:
        provider_close_error: BaseException | None = None
        if provider is not None:
            try:
                await provider.aclose()
            except BaseException as close_error:  # noqa: BLE001 - preserve provider cleanup evidence
                provider_close_error = close_error
                if failure is None:
                    failure = close_error
                    result = {
                        "status": "error",
                        "mode": "xlerobot-local-segment",
                        "error": {
                            "type": type(close_error).__name__,
                            "message": str(close_error),
                        },
                        "partial_result": None,
                        "stop_report": None,
                    }
                elif result is not None:
                    result["provider_cleanup_error"] = {
                        "type": type(close_error).__name__,
                        "message": str(close_error),
                    }
        _cleanup_failed_robot(robot, completed, failure, result)
        if output is not None and result is not None:
            _write_result(output, result)
        if provider_close_error is not None and failure is provider_close_error:
            raise provider_close_error


def _cleanup_failed_robot(
    robot: Any,
    completed: bool,
    failure: BaseException | None,
    result: dict[str, Any] | None,
) -> None:
    if not completed:
        close = getattr(robot, "close", None) if robot is not None else None
        if callable(close):
            try:
                close()
            except BaseException as close_error:
                if failure is None:
                    raise
                if result is not None:
                    result["cleanup_error"] = {
                        "type": type(close_error).__name__,
                        "message": str(close_error),
                    }


def _prepare_output_path(path_value: str | None) -> Path | None:
    """Reserve a result path before model loading or any robot factory call."""

    if not path_value:
        return None
    output = Path(path_value).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.parent.is_dir() or not os.access(output.parent, os.W_OK):
        raise OSError(f"output parent is not writable: {output.parent}")
    # Reserve the exact destination so a concurrent run cannot overwrite it.
    output.touch(mode=0o600, exist_ok=False)
    return output


def _write_result(output: Path, result: Mapping[str, Any]) -> None:
    output.write_text(json.dumps(_json_safe(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    robot_source = parser.add_mutually_exclusive_group(required=True)
    robot_source.add_argument("--robot-factory", help="module:function returning an authorized robot")
    robot_source.add_argument("--robot-url", help="AGX robot URL for the built-in HTTP RemoteRobot client")
    parser.add_argument(
        "--robot-token-env",
        default="XLEROBOT_TOKEN",
        help="environment variable containing the robot token",
    )
    parser.add_argument("--robot-scope", choices=("all", "arms", "base"), default="base")
    parser.add_argument("--robot-timeout-s", type=float, default=2.0)
    parser.add_argument(
        "--authorize-motion",
        action="store_true",
        help="explicitly arm base motion for the built-in --robot-url route",
    )
    parser.add_argument("--instruction", required=True)
    parser.add_argument(
        "--camera",
        required=True,
        help="camera key with camera_status freshness evidence",
    )
    parser.add_argument("--output")
    parser.add_argument("--session-id", default="lightnav0-xlerobot-local-segment")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--duration-s", type=float, default=0.25)
    parser.add_argument("--max-linear-velocity", type=float, default=0.3)
    parser.add_argument("--max-angular-velocity", type=float, default=0.6)
    parser.add_argument("--prediction-timeout-s", type=float, default=0.5)
    parser.add_argument("--max-prediction-retries", type=int, default=2)
    parser.add_argument("--inference-url", required=True)
    parser.add_argument("--inference-token-env")
    parser.add_argument("--inference-timeout-s", type=float, default=30.0)
    args = parser.parse_args(argv)
    result = asyncio.run(_run_from_args(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
