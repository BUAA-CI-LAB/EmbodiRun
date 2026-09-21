"""Application service coordinating robot control and runtime resources.

``ControlService`` owns task serialization, control/robot/camera/client locks,
resource leases, arbitration ownership, shared observations, and cleanup for one
configured robot owner. It does not open an HTTP listener; the Control server
assembles this service with the transport API. Lease and stop failures remain
visible to callers so cleanup cannot fabricate a confirmed stopped state.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

from embodirun.bindings import BindingDefinition, binding_definition
from embodirun.devices import (
    DeviceLease,
    DeviceManager,
    DeviceResource,
    ResourceIdentity,
    canonical_resource_identity,
)
from embodirun.devices.execution.arbitration import (
    ArbiterCommandSink,
    AuthorityState,
    EmergencyStopPolicy,
    RobotAdapterCommandPort,
    RobotControlArbiter,
)
from embodirun.devices.execution.teleop import resolve_teleop_action
from embodirun.devices.observations import ObservationRecorder, ObservationSnapshot
from embodirun.devices.observations.hub import SharedSensorError, SharedSensorHub
from embodirun.devices.observations.views import (
    ObservationViewError,
)
from embodirun.devices.observations.views import (
    snapshot_payload as _snapshot_payload,
)
from embodirun.devices.observations.views import (
    snapshot_robot_observation as _snapshot_robot_observation,
)
from embodirun.devices.recording import ActionEvent
from embodirun.model_services import build_inference_client
from embodirun.robots import (
    RobotAction,
    RobotAdapter,
    RobotDefinition,
    RobotObservation,
    robot_definition,
)
from embodirun.robots.sensors import SensorInput
from embodirun.robots.sensors.cameras import (
    CameraFrame,
    CameraSource,
    CameraSources,
    create_camera_source,
)

from .auth import AuthPolicy
from .contracts import (
    CONTROL_CONFIG_SCHEMA,
    ControlContractError,
    ControlRuntimeProfile,
    ControlServiceConfig,
    TaskRequest,
    TaskResult,
)
from .model_loop import ControlRuntime

_MAX_CONFIG_BYTES = 1024 * 1024

_MAX_REQUEST_BYTES = 64 * 1024

_TRUSTED_MANUAL_OWNER = ("trusted", "loopback")


class ControlServiceError(RuntimeError):
    """A configured control service cannot execute a task safely."""


class ControlTaskRejected(ControlServiceError):
    """A valid task conflicts with this runtime or its current state."""


def _manual_owner_key(
    caller_id: str | None,
    session_id: str | None,
) -> tuple[str, str]:
    """Normalize legacy and authenticated manual ownership scopes."""

    if caller_id is None and session_id is None:
        return _TRUSTED_MANUAL_OWNER
    if (
        not isinstance(caller_id, str)
        or not caller_id.strip()
        or not isinstance(session_id, str)
        or not session_id.strip()
    ):
        raise ControlTaskRejected("manual ownership requires both caller_id and session_id")
    return caller_id, session_id


class _ManagedCameraLease:
    """A group of per-camera leases with one consumer-facing capture source."""

    def __init__(self, source: CameraSource, leases: tuple[DeviceLease, ...]):
        self.value = source
        self._leases = leases
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        for lease in reversed(self._leases):
            try:
                lease.close()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise errors[0]


class _CameraNameView:
    """Rename a shared physical frame for one runtime's input binding.

    The owner source is opened once and its encoded bytes are reused. Runtime
    profiles may use different logical names for that source, so the view
    changes only the frame label without reopening or re-encoding the camera.
    """

    def __init__(self, source: CameraSource, name: str):
        self._source = source
        self._name = name

    def capture(self) -> tuple[CameraFrame, ...]:
        frames = tuple(self._source.capture())
        if len(frames) > 1 and self._name not in {frame.name for frame in frames}:
            raise ControlServiceError("multi-frame camera sources require an explicit frame-name mapping")
        if len(frames) != 1 or frames[0].name == self._name:
            return frames
        return (replace(frames[0], name=self._name),)

    def close(self) -> None:
        # The DeviceLease owns the physical source and performs the one close.
        return None


class ControlService:
    """Serve one physical robot owner and its configured runtime profiles."""

    def __init__(
        self,
        config: ControlServiceConfig,
        *,
        camera_factory: Callable[[Sequence[SensorInput]], CameraSource] = (create_camera_source),
        client_factory: Callable[[ControlServiceConfig, float], Any] | None = None,
        runtime_factory: Callable[..., Any] = ControlRuntime,
        device_manager: DeviceManager | None = None,
        recorder: ObservationRecorder | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.camera_factory = camera_factory
        self.client_factory = client_factory or create_inference_client
        self.runtime_factory = runtime_factory
        self.device_manager = device_manager
        if self.device_manager is None:
            self.device_manager = DeviceManager(
                config.node_id,
                owner_id=f"control:{config.runtime_id}",
            )
        self.monotonic = monotonic
        self.sleep = sleep
        self._started_at_unix = time.time()
        self._task_lock = threading.Lock()
        self._client_lock = threading.Lock()
        self._control_lock = threading.Lock()
        # Serialize the ownership check with the corresponding arbiter call.
        # Without this small lock, a delayed old caller could pass its check,
        # yield while a new caller acquired manual control, then mutate the
        # new owner's deadman or action state.
        self._manual_operation_lock = threading.Lock()
        self._robot_lock = threading.Lock()
        self._camera_lock = threading.Lock()
        self._wireless_client: Any | None = None
        self._wireless_clients: dict[tuple[str, str], Any] = {}
        self._current_arbiter: RobotControlArbiter | None = None
        self._current_task_cancel: threading.Event | None = None
        # Manual ownership is enforced at the service boundary as well as in
        # the arbiter.  The arbiter intentionally has one physical manual
        # authority, while this tuple prevents one authenticated HTTP client
        # from changing or releasing another client's lease.
        self._manual_owner: tuple[str, str] | None = None
        self._robot: Any | None = None
        self._robot_lease: DeviceLease | None = None
        self._robot_prepared = False
        # Camera leases belong to the service lifetime.  A task or observer
        # receives a view over these retained owners, so model/session cleanup
        # cannot close a source still needed by another runtime.
        self._camera_owners: dict[str, DeviceLease] = {}
        self._camera_sources: dict[str, CameraSource] = {}
        self._camera_profiles: dict[str, tuple[str, str]] = {}
        self._camera_observers: dict[str, CameraSource] = {}
        self._estop_latched = False
        self._closed = False
        self._shared_observations = SharedSensorHub(
            self._read_shared_state,
        )
        if recorder is not None and recorder.store is not self._shared_observations.store:
            raise ValueError("recorder must use the service observation store")
        self._recorder = recorder

    def health(self) -> dict[str, Any]:
        """Report ready only when the configured inference service is reachable."""

        if not self.config.inference_enabled:
            return {
                "status": "ok",
                "runtime_id": self.config.runtime_id,
                "inference": "disabled",
                "devices": self.describe()["devices"],
            }
        profile = self.config.profile_for_runtime(self.config.runtime_id)
        profile_config = self.config.for_profile(profile)
        client = self._inference_client(profile_config, 2.0)
        try:
            health = client.health()
        finally:
            self._release_inference_client(client, profile_config)
        if health.get("status") != "ok":
            raise ControlServiceError(f"inference service at {self.config.inference_endpoint} is not healthy")
        return {"status": "ok", "runtime_id": self.config.runtime_id}

    def execute(
        self,
        request: TaskRequest,
        *,
        cancel_event: threading.Event | None = None,
    ) -> TaskResult:
        """Run one exclusive model task; the service retains robot ownership."""

        if cancel_event is not None and not isinstance(cancel_event, threading.Event):
            raise TypeError("cancel_event must be a threading.Event or None")

        if request.runtime_id not in {
            self.config.runtime_id,
            *self.config.runtime_profiles,
        }:
            raise ControlTaskRejected(f"runtime {request.runtime_id!r} is not served by this control service")
        if not self.config.inference_enabled:
            raise ControlTaskRejected("control service has no inference runtime")
        try:
            profile = self.config.profile_for_runtime(request.runtime_id)
        except ControlContractError as error:
            raise ControlTaskRejected(str(error)) from error
        with self._control_lock:
            if self._closed:
                raise ControlTaskRejected("control service is closed")
            if self._current_arbiter is None and self._estop_latched:
                raise ControlTaskRejected("emergency stop is latched")
        if not self._task_lock.acquire(blocking=False):
            raise ControlTaskRejected("control service is already executing a task")
        try:
            return self._execute_locked(request, profile, cancel_event=cancel_event)
        finally:
            self._task_lock.release()

    def _execute_locked(
        self,
        request: TaskRequest,
        profile: ControlRuntimeProfile | None = None,
        *,
        cancel_event: threading.Event | None = None,
    ) -> TaskResult:
        profile = profile or self.config.profile_for_runtime(request.runtime_id)
        profile_config = self.config.for_profile(profile)
        binding, robot_definition_value = _definitions(profile_config)
        if request.chunk_steps > binding.maximum_chunk_steps:
            raise ControlTaskRejected(
                f"chunk_steps {request.chunk_steps} exceeds binding "
                f"{binding.kind!r} maximum {binding.maximum_chunk_steps}"
            )
        if profile_config.runtime_options:
            names = ", ".join(sorted(profile_config.runtime_options))
            raise ControlServiceError(f"unsupported runtime options: {names}")
        client = self._inference_client(profile_config, request.inference_timeout_s)
        cameras: CameraSource | None = None
        controller: Any | None = None
        task_cancel = None
        current_snapshot: ObservationSnapshot | None = None
        completed = 0
        try:
            health = client.health()
            if health.get("status") != "ok":
                raise ControlServiceError(f"inference service at {self.config.inference_endpoint} is not healthy")
            cameras, _ = self._acquire_cameras(profile_config)
            robot, arbiter = self._ensure_robot_arbiter(
                robot_definition_value,
                profile_config,
            )
            task_cancel = arbiter.begin_model_task(cancel_event)
            with self._control_lock:
                self._current_task_cancel = task_cancel

            def observation_source() -> RobotObservation:
                if current_snapshot is None:
                    raise ControlServiceError("shared observation was not published before model input")
                try:
                    return _snapshot_robot_observation(current_snapshot)
                except ObservationViewError as error:
                    raise ControlServiceError(str(error)) from error

            controller = self.runtime_factory(
                robot,
                client,
                instruction=request.prompt,
                mapper=binding.mapper_factory(),
                chunk_steps=request.chunk_steps,
                control_hz=request.control_hz,
                command_sink=ArbiterCommandSink(arbiter, task_cancel),
                observation_source=observation_source,
                cancel_event=task_cancel,
                monotonic=self.monotonic,
                sleep=self.sleep,
            )
            for _ in range(request.max_steps):
                requested = _runtime_camera_requests(profile_config)
                current_snapshot = self._shared_observations.snapshot(
                    require_state=True,
                    required_source_ids=[source_id for source_id, _ in requested],
                    force_publish=True,
                )
                frames = self._shared_observations.frames_for(
                    current_snapshot,
                    requested,
                )
                controller.step(frames)
                completed += 1
        finally:
            with self._control_lock:
                if self._current_task_cancel is task_cancel:
                    self._current_task_cancel = None
            try:
                if controller is not None:
                    controller.close()
            finally:
                try:
                    if self.device_manager is None and cameras is not None:
                        cameras.close()
                finally:
                    self._release_inference_client(client, profile_config)
        return TaskResult(request.request_id, request.runtime_id, completed)

    def _ensure_robot_arbiter(
        self,
        robot_definition_value: RobotDefinition,
        profile_config: ControlServiceConfig | None = None,
    ) -> tuple[Any, RobotControlArbiter]:
        profile_config = profile_config or self.config
        with self._robot_lock:
            robot = self._robot
            arbiter = self._current_arbiter
            if robot is not None and arbiter is not None:
                if self._closed:
                    raise ControlTaskRejected("control service is closed")
                if not self._robot_prepared or getattr(robot, "prepared", True) is False:
                    self._prepare_existing_robot(robot, profile_config)
                return robot, arbiter
            with self._control_lock:
                if self._closed or self._estop_latched:
                    raise ControlTaskRejected("service is closed or emergency stop is latched")
            try:
                robot_config = robot_definition_value.config_factory(
                    profile_config.robot_id,
                    profile_config.robot_options,
                )
            except (TypeError, ValueError) as error:
                raise ControlServiceError(
                    f"robot {self.config.robot_id!r} configuration is invalid: {error}"
                ) from error
            try:
                robot = robot_definition_value.adapter_type(robot_config)
            except (TypeError, ValueError) as error:
                raise ControlServiceError(
                    f"robot {profile_config.robot_id!r} adapter construction failed: {error}"
                ) from error
            lease: DeviceLease | None = None
            robot_resource = _configured_resource(profile_config, kind="robot")
            external_owner = bool(robot_resource and robot_resource.get("external_owner"))
            if profile_config.robot_kind == "lerobot.xlerobot" and not external_owner:
                raise ControlServiceError("lerobot.xlerobot requires an external_owner robot resource")
            if self.device_manager is not None:
                if external_owner and profile_config.robot_kind not in {
                    "unitree.go2",
                    "lerobot.xlerobot",
                }:
                    raise ControlServiceError(
                        "external device owner requires a supported proxy adapter; "
                        f"{profile_config.robot_kind!r} cannot be attached"
                    )

                def open_robot() -> Any:
                    try:
                        if not _supports_passive_connect(robot) or not _supports_prepare_operation(robot):
                            # A legacy adapter with no passive boundary may be
                            # opened only for an explicit control request.  It
                            # is never used by the observation path.
                            _connect_robot(robot, prepare=True)
                            robot._embodirun_prepared_on_connect = True
                        else:
                            _connect_robot(robot, prepare=False)
                            robot._embodirun_prepared_on_connect = False
                    except BaseException:
                        with contextlib.suppress(BaseException):
                            robot.close()
                        raise
                    return robot

                identity = _resource_identity_from_config(profile_config, robot_resource)
                try:
                    lease = self.device_manager.acquire(
                        DeviceResource(
                            identity,
                            open_robot,
                            closer=lambda value: value.close(),
                            preparer=None if external_owner else _prepare_adapter,
                            external_owner=external_owner,
                            component_identities=_robot_component_identities(profile_config),
                        ),
                        role="control",
                        prepare=not external_owner,
                    )
                    if lease.value is not robot:
                        # Another lease in this manager may already own the
                        # physical adapter.  The newly constructed candidate
                        # was never opened by DeviceManager and must not be
                        # retained as a second bus handle.
                        with contextlib.suppress(BaseException):
                            robot.close()
                        robot = lease.value
                except BaseException:
                    # DeviceManager owns rollback after it invokes open_robot;
                    # closing here as well could hide a failed cleanup and
                    # release a handle that has been quarantined there.
                    raise
                robot = lease.value
                self._robot_lease = lease
            try:
                if self.device_manager is None:
                    _connect_robot(robot, prepare=not external_owner)
                arbiter = self._make_robot_arbiter(robot, profile_config)
            except BaseException:
                if lease is not None:
                    try:
                        lease.close()
                    finally:
                        self._robot_lease = None
                else:
                    with contextlib.suppress(BaseException):
                        robot.close()
                raise
            with self._control_lock:
                self._robot = robot
                self._current_arbiter = arbiter
                self._robot_prepared = not external_owner or bool(
                    getattr(robot, "_embodirun_prepared_on_connect", False)
                )
                stopped = self._estop_latched or self._closed
            if stopped:
                arbiter.emergency_stop()
                raise ControlTaskRejected("connection interrupted by emergency stop or shutdown")
            if external_owner and not self._robot_prepared:
                self._prepare_existing_robot(robot, profile_config)
            return robot, arbiter

    def describe(self) -> dict[str, Any]:
        """Describe technical availability without contacting inference.

        This is intentionally useful for a no-model service and for startup
        diagnostics.  It reports configured resources and unresolved lifecycle
        state; it never calls ``health`` on a model client or opens a motor.
        """

        resources: list[dict[str, Any]] = []
        if self.device_manager is not None:
            resources = [status.to_dict() for status in self.device_manager.status()]
            state_error = self.device_manager.state_store.error
            if state_error:
                resources.append({"state": "uncertain", "error": state_error})
        else:
            resources = [
                {
                    "identity": item.get("identity"),
                    "kind": item.get("kind"),
                    "state": "configured",
                    "owner_id": self.config.runtime_id,
                }
                for item in self.config.device_resources
            ]
        runtime_ids = sorted({self.config.runtime_id, *self.config.runtime_profiles})
        return {
            "status": "ok",
            "process": {
                "pid": os.getpid(),
                "started_at_unix": self._started_at_unix,
                "package_version": _package_version(),
            },
            "config": {
                "schema": CONTROL_CONFIG_SCHEMA,
                "runtime_id": self.config.runtime_id,
            },
            "runtime_id": self.config.runtime_id,
            "runtime_ids": runtime_ids,
            "robot_id": self.config.robot_id,
            "robot_kind": self.config.robot_kind,
            # External-owner recipes may bind separate Control services to
            # one robot.  Expose the configured scope so a caller can verify
            # that it is talking to the intended base or arm boundary before
            # sending an action.
            "control_scope": self.config.robot_options.get("scope"),
            "inference_enabled": self.config.inference_enabled,
            "binding": _binding_description(self.config),
            "devices": resources,
            "capabilities": {
                "describe": True,
                "observe": True,
                "execute": self.config.inference_enabled,
                "propose": self.config.inference_enabled,
                "prepare": True,
            },
        }

    @property
    def observation_store(self):
        """Return the bounded store shared by model, observe, and recording."""

        return self._shared_observations.store

    def attach_recorder(self, recorder: ObservationRecorder) -> None:
        """Attach a recorder that subscribes to this service's store."""

        if not isinstance(recorder, ObservationRecorder):
            raise TypeError("recorder must be an ObservationRecorder")
        if recorder.store is not self._shared_observations.store:
            raise ValueError("recorder must use the service observation store")
        self._recorder = recorder

    def start_recording(self) -> Any:
        """Start the configured recorder without taking a device lock."""

        recorder = self._require_recorder()
        return recorder.start().status()

    def stop_recording(self, timeout_s: float = 1.0) -> Any:
        """Request bounded recorder shutdown and return its status."""

        recorder = self._require_recorder()
        recorder.stop(timeout_s=timeout_s)
        return recorder.status()

    def recording_status(self) -> Any:
        """Return recorder diagnostics, including queue and storage failures."""

        return self._require_recorder().status()

    def get_record(self, observation_id: str) -> dict[str, Any] | None:
        """Read one raw recorder record by its original snapshot ID."""

        return self._require_recorder().get_record(observation_id)

    def get_snapshot(
        self,
        observation_id: str,
        *,
        runtime_id: str | None = None,
        include_robot: bool = False,
    ) -> dict[str, Any]:
        """Return one retained shared snapshot in a runtime's frame-name view."""

        profile_config = self._profile_config(runtime_id)
        snapshot = self._shared_observations.get_snapshot(observation_id)
        requested, frames = self._snapshot_frame_view(profile_config, snapshot)
        try:
            return _snapshot_payload(
                snapshot,
                frames,
                profile_config.runtime_id,
                [source_id for source_id, _ in requested],
                require_state=include_robot,
                include_robot=include_robot,
            )
        except ObservationViewError as error:
            raise ControlServiceError(str(error)) from error

    def proposal_context(
        self,
        observation_id: str,
        *,
        runtime_id: str | None = None,
    ) -> tuple[
        ControlServiceConfig,
        BindingDefinition,
        RobotObservation,
        tuple[CameraFrame, ...],
    ]:
        """Resolve one retained snapshot for a non-executing model proposal.

        This seam deliberately reads the shared store only.  It does not call
        ``observe``, acquire a camera lease, connect a robot, or prepare an
        actuator; the application layer has already checked freshness and
        identity before calling it.
        """

        profile_config = self._profile_config(runtime_id)
        if not profile_config.inference_enabled:
            raise ControlTaskRejected("control service has no inference runtime")
        snapshot = self._shared_observations.get_snapshot(observation_id)
        requested, frames = self._snapshot_frame_view(profile_config, snapshot)
        if snapshot.state is None:
            raise ControlServiceError(f"snapshot {observation_id!r} has no robot state")
        if any(
            source_id not in snapshot.source_timestamps_ns or source_id in snapshot.errors for source_id, _ in requested
        ):
            raise ControlServiceError(f"snapshot {observation_id!r} is missing a configured camera source")
        if requested and len(frames) != len(requested):
            raise ControlServiceError(f"snapshot {observation_id!r} is missing a configured camera frame")
        binding, _ = _definitions(profile_config)
        try:
            robot_observation = _snapshot_robot_observation(snapshot)
        except ObservationViewError as error:
            raise ControlServiceError(str(error)) from error
        return profile_config, binding, robot_observation, tuple(frames)

    def get_media(
        self,
        observation_id: str,
        frame_name: str,
        *,
        runtime_id: str | None = None,
    ) -> CameraFrame:
        """Return one encoded frame from a retained snapshot."""

        snapshot = self._shared_observations.get_snapshot(observation_id)
        if runtime_id is None:
            return self._shared_observations.get_media(observation_id, frame_name)
        profile_config = self._profile_config(runtime_id)
        requested = _runtime_camera_requests(profile_config)
        for frame in self._shared_observations.frames_for(snapshot, requested):
            if frame.name == frame_name:
                return frame
        raise SharedSensorError(f"snapshot {observation_id!r} has no frame named {frame_name!r}")

    def subscribe(self, max_queue: int = 8, *, replay_latest: bool = False):
        """Subscribe to immutable snapshots without owning camera lifecycle."""

        return self._shared_observations.subscribe(
            max_queue=max_queue,
            replay_latest=replay_latest,
        )

    def _require_recorder(self) -> ObservationRecorder:
        recorder = self._recorder
        if recorder is None:
            raise ControlServiceError("observation recorder is not configured")
        return recorder

    def _profile_config(self, runtime_id: str | None) -> ControlServiceConfig:
        selected_runtime = runtime_id or self.config.runtime_id
        if not self.config.inference_enabled:
            if selected_runtime != self.config.runtime_id:
                raise ControlTaskRejected(f"runtime {selected_runtime!r} is not served by this device service")
            return self.config
        try:
            profile = self.config.profile_for_runtime(selected_runtime)
        except ControlContractError as error:
            raise ControlTaskRejected(str(error)) from error
        return self.config.for_profile(profile)

    def _snapshot_frame_view(
        self,
        profile_config: ControlServiceConfig,
        snapshot: ObservationSnapshot,
    ) -> tuple[tuple[tuple[str, str], ...], tuple[CameraFrame, ...]]:
        """Resolve a profile view while preserving unavailable-source status."""

        requested = _runtime_camera_requests(profile_config)
        try:
            frames = self._shared_observations.frames_for(snapshot, requested)
        except SharedSensorError:
            source_frames = snapshot.metadata.get("source_frames", {})
            if not isinstance(source_frames, Mapping) or any(
                source_id not in source_frames for source_id, _ in requested
            ):
                # A disconnected source has no frame mapping in this
                # snapshot; let the payload carry its per-view diagnostics.
                return requested, ()
            # A source that did publish frames can still fail view resolution
            # because its profile has an ambiguous multi-frame mapping. Keep
            # that configuration error visible.
            raise
        return requested, frames

    def _read_shared_state(self) -> RobotObservation | None:
        """Read robot state through the existing owner scheduler, if present."""

        # Setup and the first passive read publish the arbiter before their
        # scheduler transaction returns.  Share the owner lock so the producer
        # cannot race that initial read and exhaust the bounded read queue.
        with self._robot_lock:
            with self._control_lock:
                arbiter = self._current_arbiter
            if arbiter is None:
                return None
            return arbiter.observe()

    def observe(
        self,
        *,
        runtime_id: str | None = None,
        include_robot: bool = False,
    ) -> dict[str, Any]:
        """Capture passive camera data and optional verified robot state.

        Camera observation is a lazy path.  ``include_robot`` is explicit and
        uses a read-only adapter when the driver exposes that capability; it
        never turns a camera request into torque/prepare.
        """

        profile_config = self._profile_config(runtime_id)
        if include_robot:
            with self._control_lock:
                owner_exists = self._current_arbiter is not None
            if not owner_exists:
                # This path performs only the adapter's verified passive
                # connect/read boundary.  It must not call ``prepare``.
                self._observe_robot_read_only(profile_config, read=False)
        if profile_config.inputs:
            self._acquire_cameras(profile_config, role="observer", keep=True)
        try:
            snapshot = self._shared_observations.snapshot(
                require_state=include_robot,
                required_source_ids=[source_id for source_id, _ in _runtime_camera_requests(profile_config)],
            )
            requested, frames = self._snapshot_frame_view(profile_config, snapshot)
            try:
                return _snapshot_payload(
                    snapshot,
                    frames,
                    profile_config.runtime_id,
                    [source_id for source_id, _ in requested],
                    require_state=include_robot,
                    include_robot=include_robot,
                )
            except ObservationViewError as error:
                raise ControlServiceError(str(error)) from error
        except SharedSensorError as error:
            raise ControlServiceError(str(error)) from error

    def release_observer(self, runtime_id: str | None = None) -> None:
        """Release one camera observer lease without touching control ownership."""
        # Observation consumers own only a subscription/view.  Removing that
        # view must never stop the shared producer or close a physical source.
        self._profile_config(runtime_id)

    def _acquire_cameras(
        self,
        profile_config: ControlServiceConfig,
        *,
        role: str = "task",
        keep: bool = False,
    ) -> tuple[CameraSource, _ManagedCameraLease | None]:
        if self.device_manager is None:
            return self.camera_factory(profile_config.inputs), None
        sources: list[CameraSource] = []
        seen: set[str] = set()
        pending: list[tuple[SensorInput, ResourceIdentity, tuple[str, str]]] = []
        profiles: dict[str, tuple[str, str]] = {}
        with self._camera_lock:
            for item in profile_config.inputs:
                identity = _sensor_identity(profile_config, item)
                if identity.key in seen:
                    continue
                seen.add(identity.key)
                profile = _camera_capture_profile(item)
                existing = self._camera_profiles.get(identity.key)
                if existing is not None and existing != profile:
                    raise ControlServiceError(f"camera {identity.key!r} has incompatible capture profiles")
                local_existing = profiles.get(identity.key)
                if local_existing is not None and local_existing != profile:
                    raise ControlServiceError(f"camera {identity.key!r} has incompatible capture profiles")
                profiles[identity.key] = profile
                pending.append((item, identity, profile))
            to_register: list[str] = []
            for item, identity, profile in pending:
                resource = _configured_resource(profile_config, identity=identity)
                if resource is not None and resource.get("external_owner"):
                    raise ControlServiceError(
                        f"sensor {identity.key!r} is externally owned; a proxy camera source is required"
                    )
                source = self._camera_sources.get(identity.key)
                if source is None:
                    lease = self.device_manager.acquire(
                        DeviceResource(
                            identity,
                            lambda input_item=item: self.camera_factory((input_item,)),
                            closer=lambda value: value.close(),
                        ),
                        role=role,
                        prepare=False,
                    )
                    self._camera_owners[identity.key] = lease
                    source = lease.value
                    self._camera_sources[identity.key] = source
                if not self._shared_observations.has_source(identity.key):
                    to_register.append(identity.key)
                self._camera_profiles[identity.key] = profile
                sources.append(_CameraNameView(source, item.name))
            # Register only after every requested source has opened.  If a
            # later camera fails, already-open service owners remain valid
            # and can be reused by the next profile; no producer worker is
            # left pointing at a lease that rollback just closed.
            for key in dict.fromkeys(to_register):
                self._shared_observations.add_source(
                    key,
                    self._camera_sources[key],
                )
        source: CameraSource = sources[0] if len(sources) == 1 else CameraSources(tuple(sources))
        return source, None

    def _make_robot_arbiter(
        self,
        robot: Any,
        profile_config: ControlServiceConfig,
    ) -> RobotControlArbiter:
        """Create the sole command/observation queue for one robot owner."""

        return RobotControlArbiter(
            RobotAdapterCommandPort(
                robot,
                action_resolver=resolve_teleop_action,
                emergency_stop_policy=(
                    EmergencyStopPolicy.PREEMPTIVE
                    if profile_config.robot_kind == "franka.fr3"
                    else EmergencyStopPolicy.SERIALIZED
                ),
            ),
            clock=self.monotonic,
            event_callback=self._record_action_event,
        )

    def _record_action_event(self, payload: Mapping[str, Any]) -> None:
        """Queue arbiter facts for the optional recorder without blocking I/O."""

        recorder = self._recorder
        if recorder is None:
            return
        try:
            recorder.record_action(
                ActionEvent(
                    action_id=str(payload.get("action_id", "")),
                    source=str(payload.get("source", "")),
                    stage=str(payload.get("stage", "")),
                    executed=bool(payload.get("executed", False)),
                    observation_id=(
                        payload.get("observation_id") if isinstance(payload.get("observation_id"), str) else None
                    ),
                    outcome=(payload.get("outcome") if isinstance(payload.get("outcome"), str) else None),
                    timestamp_ns=(
                        payload.get("timestamp_ns") if isinstance(payload.get("timestamp_ns"), int) else None
                    ),
                    payload=(payload.get("payload") if isinstance(payload.get("payload"), Mapping) else {}),
                )
            )
        except BaseException:
            # The arbiter records callback failures in its own diagnostics;
            # recorder failures must never change motor-loop status.
            return

    def _prepare_existing_robot(
        self,
        robot: Any,
        profile_config: ControlServiceConfig,
    ) -> None:
        """Upgrade a passive owner through the manager's real prepare hook."""

        if self.device_manager is None:
            _prepare_adapter(robot)
            self._robot_prepared = True
            return
        identity = _resource_identity_from_config(
            profile_config,
            _configured_resource(profile_config, kind="robot"),
        )
        upgraded = self.device_manager.acquire(
            DeviceResource(
                identity,
                lambda: robot,
                closer=lambda value: value.close(),
                preparer=self._prepare_via_arbiter,
                external_owner=bool(
                    (_configured_resource(profile_config, kind="robot") or {}).get("external_owner", False)
                ),
                component_identities=_robot_component_identities(profile_config),
            ),
            role="control",
            prepare=True,
        )
        previous = self._robot_lease
        if previous is not None and previous is not upgraded:
            previous.close()
        self._robot_lease = upgraded
        self._robot_prepared = True

    def _prepare_via_arbiter(self, robot: Any) -> None:
        """Run a passive-to-control prepare transaction on the bus queue."""

        with self._control_lock:
            arbiter = self._current_arbiter
        if arbiter is None:
            _prepare_adapter(robot)
            return
        arbiter.prepare()

    def _observe_robot_read_only(
        self,
        profile_config: ControlServiceConfig,
        *,
        read: bool = True,
    ) -> Any:
        # Hold the same owner lock as execute-side setup across check, open,
        # arbiter construction, and publication.  Two concurrent first reads
        # must never create two schedulers around one adapter.
        with self._robot_lock:
            with self._control_lock:
                current = self._robot
                arbiter = self._current_arbiter
            if current is not None:
                # A prepared owner already provides a read operation; opening
                # a second adapter would race its bus and could toggle torque.
                if arbiter is None:
                    raise ControlServiceError("robot owner has no observation arbiter")
                return _observation_payload(arbiter.observe()) if read else None
            try:
                _binding, definition = _definitions(profile_config)
            except ControlServiceError:
                try:
                    definition = robot_definition(profile_config.robot_kind)
                except (KeyError, TypeError):
                    raise ControlServiceError(f"robot type {profile_config.robot_kind!r} is not available") from None
            try:
                robot_config = definition.config_factory(
                    profile_config.robot_id,
                    profile_config.robot_options,
                )
                adapter = definition.adapter_type(robot_config)
            except TypeError as error:
                raise ControlServiceError(
                    f"{profile_config.robot_kind!r} has no verified passive connection; explicit prepare is required"
                ) from error
            if not _supports_passive_connect(adapter) or not _supports_prepare_operation(adapter):
                with contextlib.suppress(BaseException):
                    adapter.close()
                raise ControlServiceError(
                    f"{profile_config.robot_kind!r} has no verified passive connection; explicit prepare is required"
                )
            resource = _configured_resource(profile_config, kind="robot")
            external_owner = bool(resource and resource.get("external_owner"))
            if profile_config.robot_kind == "lerobot.xlerobot" and not external_owner:
                adapter.close()
                raise ControlServiceError("lerobot.xlerobot requires an external_owner robot resource")
            if self.device_manager is None:
                try:
                    _connect_robot(adapter, prepare=False)
                    arbiter = self._make_robot_arbiter(adapter, profile_config)
                    with self._control_lock:
                        self._robot = adapter
                        self._current_arbiter = arbiter
                        self._robot_prepared = False
                    return _observation_payload(arbiter.observe()) if read else None
                except BaseException:
                    with contextlib.suppress(BaseException):
                        adapter.close()
                    raise
            identity = _resource_identity_from_config(
                profile_config,
                _configured_resource(profile_config, kind="robot"),
            )
            lease = self.device_manager.acquire(
                DeviceResource(
                    identity,
                    lambda: _connect_and_return(adapter),
                    closer=lambda value: value.close(),
                    preparer=_prepare_adapter,
                    external_owner=external_owner,
                    component_identities=_robot_component_identities(profile_config),
                ),
                role="observer",
                prepare=False,
            )
            if lease.value is not adapter:
                with contextlib.suppress(BaseException):
                    adapter.close()
            robot = lease.value
            try:
                arbiter = self._make_robot_arbiter(robot, profile_config)
            except BaseException:
                lease.close()
                raise
            with self._control_lock:
                self._robot = robot
                self._robot_lease = lease
                self._current_arbiter = arbiter
                self._robot_prepared = False
            return _observation_payload(arbiter.observe()) if read else None

    def get_control_arbiter(self) -> RobotControlArbiter:
        """Prepare and return the sole command arbiter for direct/manual work.

        Device-only services have no model binding, so this path resolves the
        robot definition directly.  It still enters the same prepared adapter
        and arbiter ownership used by model tasks; no second driver is opened.
        """

        profile_config = self._profile_config(None)
        definition = self._control_robot_definition(profile_config)
        _robot, arbiter = self._ensure_robot_arbiter(definition, profile_config)
        return arbiter

    @staticmethod
    def _control_robot_definition(
        profile_config: ControlServiceConfig,
    ) -> RobotDefinition:
        try:
            _binding, definition = _definitions(profile_config)
        except ControlServiceError:
            try:
                definition = robot_definition(profile_config.robot_kind)
            except (KeyError, TypeError):
                raise ControlServiceError(f"robot type {profile_config.robot_kind!r} is not available") from None
        return definition

    def emergency_stop(self) -> dict[str, Any]:
        with self._control_lock:
            arbiter = self._current_arbiter
            if arbiter is None:
                self._estop_latched = True
            task_cancel = self._current_task_cancel
            if task_cancel is not None:
                task_cancel.set()
        if arbiter is not None:
            arbiter.emergency_stop()
        return self.control_snapshot()

    def reset_emergency_stop(self) -> dict[str, Any]:
        with self._control_lock:
            arbiter = self._current_arbiter
        if arbiter is not None:
            arbiter.reset_emergency_stop()
        else:
            with self._control_lock:
                self._estop_latched = False
        return self.control_snapshot()

    def acquire_manual(
        self,
        *,
        caller_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Acquire manual control for one caller/session scope.

        The arbiter remains the single physical authority.  This additional
        service-level lease is deliberately a small tuple rather than a
        second state machine: it makes a late release/deadman/action from an
        old HTTP client fail before it can reach the shared arbiter.
        """

        owner = _manual_owner_key(caller_id, session_id)
        with self._manual_operation_lock:
            with self._control_lock:
                if self._closed:
                    raise ControlTaskRejected("control service is closed")
                current_owner = self._manual_owner
                if current_owner is not None and current_owner != owner:
                    raise ControlTaskRejected("manual control is owned by another caller")
                reserved = current_owner is None
                if reserved:
                    # Reserve before opening/preparing the adapter.  Otherwise
                    # a concurrent second acquire could pass the check while
                    # the first caller is still connecting its robot.
                    self._manual_owner = owner
            try:
                profile_config = self._profile_config(None)
                definition = self._control_robot_definition(profile_config)
                _, arbiter = self._ensure_robot_arbiter(definition, profile_config)
                with self._control_lock:
                    task_cancel = self._current_task_cancel
                if task_cancel is not None:
                    task_cancel.set()
                arbiter.acquire_manual()
            except BaseException:
                if reserved:
                    with self._control_lock:
                        if self._manual_owner == owner:
                            self._manual_owner = None
                raise
            return self.control_snapshot()

    def release_manual(
        self,
        *,
        caller_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        owner = _manual_owner_key(caller_id, session_id)
        with self._manual_operation_lock:
            with self._control_lock:
                self._require_manual_owner_locked(owner)
                arbiter = self._current_arbiter
            if arbiter is None:
                raise ControlTaskRejected("manual control must be acquired first")
            try:
                arbiter.release_manual()
            except BaseException:
                # A failed/unknown stop leaves the owner in place so another
                # caller cannot take over while the physical state is
                # uncertain.
                raise
            with self._control_lock:
                if self._manual_owner == owner:
                    self._manual_owner = None
            return self.control_snapshot()

    def set_manual_deadman(
        self,
        active: bool,
        *,
        caller_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        owner = _manual_owner_key(caller_id, session_id)
        with self._manual_operation_lock:
            with self._control_lock:
                self._require_manual_owner_locked(owner)
                arbiter = self._current_arbiter
            if arbiter is None:
                raise ControlTaskRejected("manual control must be acquired first")
            arbiter.set_deadman(active)
            return self.control_snapshot()

    def submit_manual_action(
        self,
        action: Any,
        *,
        caller_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        owner = _manual_owner_key(caller_id, session_id)
        with self._manual_operation_lock:
            robot_action = _robot_action(action)
            with self._control_lock:
                self._require_manual_owner_locked(owner)
                arbiter = self._current_arbiter
            if arbiter is None:
                raise ControlTaskRejected("manual control must be acquired first")
            ticket = arbiter.submit_manual(robot_action, wait=False)
            return {**self.control_snapshot(), "ticket_status": ticket.status.value}

    def _require_manual_owner_locked(self, owner: tuple[str, str]) -> None:
        if self._manual_owner is None:
            raise ControlTaskRejected("manual control must be acquired first")
        if self._manual_owner != owner:
            raise ControlTaskRejected("manual control is owned by another caller")

    def control_snapshot(self) -> dict[str, Any]:
        with self._control_lock:
            arbiter = self._current_arbiter
            estop_latched = self._estop_latched
            active_task = self._current_task_cancel is not None
        if arbiter is None:
            return {
                "status": "ok",
                "runtime_id": self.config.runtime_id,
                "robot_id": self.config.robot_id,
                "authority": AuthorityState.ESTOP_LATCHED.value if estop_latched else AuthorityState.MODEL.value,
                "active_task": False,
                "last_error": None,
                "closed": self._closed,
            }
        snapshot = arbiter.snapshot()
        return {
            "status": "ok",
            "runtime_id": self.config.runtime_id,
            "active_task": active_task,
            **snapshot,
        }

    def close(self) -> None:
        """Release process-owned robot and wireless resources."""

        with self._control_lock:
            self._closed = True
            self._manual_owner = None
            if self._current_task_cancel is not None:
                self._current_task_cancel.set()
        with self._robot_lock, self._control_lock:
            arbiter = self._current_arbiter
            robot = self._robot
            self._current_task_cancel = None
        primary_error: BaseException | None = None
        arbiter_failed = False
        if arbiter is not None:
            try:
                arbiter.close(hold=self._robot_prepared)
            except BaseException as error:
                arbiter_failed = True
                if self.device_manager is not None:
                    with contextlib.suppress(BaseException):
                        self.device_manager.mark_uncertain(
                            _resource_identity_from_config(
                                self.config,
                                _configured_resource(self.config, kind="robot"),
                            ),
                            f"robot stop/hold failed during close: {error}",
                        )
                if primary_error is None:
                    primary_error = error
        if robot is not None and self.device_manager is None and not arbiter_failed:
            try:
                robot.close()
            except BaseException as error:
                if primary_error is None:
                    primary_error = error
        with self._client_lock:
            clients = tuple(self._wireless_clients.values())
            if self._wireless_client is not None and self._wireless_client not in clients:
                clients += (self._wireless_client,)
            self._wireless_clients.clear()
            self._wireless_client = None
        for client in clients:
            try:
                _shutdown_client(client)
            except BaseException as error:
                if primary_error is None:
                    primary_error = error
        if self._recorder is not None:
            try:
                self._recorder.stop(timeout_s=1.0)
            except BaseException as error:
                if primary_error is None:
                    primary_error = error
        self._camera_observers.clear()
        observations_closed = self._shared_observations.close(timeout_s=1.0)
        if self.device_manager is not None and not arbiter_failed and observations_closed:
            try:
                self.device_manager.close()
            except BaseException as error:
                if primary_error is None:
                    primary_error = error
        if primary_error is None:
            with self._robot_lock, self._control_lock:
                self._current_arbiter = None
                self._robot = None
                self._robot_lease = None
                self._robot_prepared = False
        if primary_error is not None:
            raise primary_error

    def _inference_client(
        self,
        config: ControlServiceConfig,
        timeout_s: float,
    ) -> Any:
        if not config.inference_enabled:
            raise ControlServiceError("inference is disabled for this device service")
        if config.inference_transport != "wireless":
            return self.client_factory(config, timeout_s)
        key = (config.inference_transport, config.inference_endpoint)
        with self._client_lock:
            client = self._wireless_clients.get(key)
            if client is None:
                client = self.client_factory(config, timeout_s)
                self._wireless_clients[key] = client
                if self._wireless_client is None:
                    self._wireless_client = client
        with_timeout = getattr(client, "with_timeout", None)
        return with_timeout(timeout_s) if callable(with_timeout) else client

    def _release_inference_client(
        self,
        client: Any,
        config: ControlServiceConfig,
    ) -> None:
        if config.inference_transport != "wireless":
            _shutdown_client(client)


def create_inference_client(config: ControlServiceConfig, timeout_s: float) -> Any:
    """Select the Control-to-Inference client from static runtime config."""

    return build_inference_client(
        config.inference_transport,
        config.inference_endpoint,
        config.inference_options,
        backend=config.inference_backend,
        timeout_s=timeout_s,
    )


def _control_job_database(
    service: ControlService,
    state_dir: str | os.PathLike[str] | None,
) -> Path:
    """Choose one persistent, per-service job database under device state."""

    configured = state_dir or os.environ.get("RLINF_DEPLOY_CONTROL_STATE_DIR")
    if configured is None:
        manager = service.device_manager
        if manager is None:
            raise ControlServiceError(
                "a persistent control state directory is required when no device manager is configured"
            )
        state_path = manager.state_store.path
        configured = state_path.parent if state_path is not None else Path(manager.lock_dir)
    root = Path(configured).expanduser()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    runtime_id = service.config.runtime_id
    safe_runtime_id = (
        "".join(
            character if character.isalnum() or character in {"-", "_", "."} else "_" for character in runtime_id
        ).strip(".")
        or "service"
    )
    return root / f"{safe_runtime_id}.jobs.sqlite"


def _load_auth_policy(token_file: Path | None) -> AuthPolicy:
    """Load optional token principals without ever echoing token contents."""

    configured = token_file
    if configured is None:
        environment_path = os.environ.get("RLINF_DEPLOY_CONTROL_TOKEN_FILE")
        configured = Path(environment_path).expanduser() if environment_path else None
    if configured is None:
        return AuthPolicy()
    try:
        value = json.loads(configured.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read control token file {configured}: {error}") from error
    if not isinstance(value, Mapping):
        raise ValueError("control token file must contain a JSON object")
    if "tokens" in value:
        value = value["tokens"]
    if not isinstance(value, Mapping):
        raise ValueError("control token file tokens must contain a JSON object")
    if not value:
        raise ValueError("control token file must define at least one bound principal")
    return AuthPolicy(value)


def _configured_recorder(
    store: Any,
    config: ControlServiceConfig,
    recording_dir: Path | None,
    recording_id: str | None,
) -> ObservationRecorder | None:
    configured_dir = recording_dir
    if configured_dir is None:
        environment_path = os.environ.get("RLINF_DEPLOY_CONTROL_RECORDING_DIR")
        configured_dir = Path(environment_path).expanduser() if environment_path else None
    if configured_dir is None:
        return None
    configured_id = recording_id or os.environ.get("RLINF_DEPLOY_CONTROL_RECORDING_ID", config.runtime_id)
    expected_frames = tuple(item.name for item in config.inputs)
    return ObservationRecorder(
        store,
        configured_dir,
        configured_id,
        expected_frame_names=expected_frames,
    )


def _definitions(
    config: ControlServiceConfig,
) -> tuple[BindingDefinition, RobotDefinition]:
    try:
        binding = binding_definition(config.binding_kind)
    except (KeyError, TypeError):
        raise ControlServiceError(f"binding {config.binding_kind!r} is not available") from None
    if binding.robot_kind != config.robot_kind:
        raise ControlServiceError(f"binding {config.binding_kind!r} does not target {config.robot_kind!r}")
    try:
        robot = robot_definition(config.robot_kind)
    except (KeyError, TypeError):
        raise ControlServiceError(f"robot type {config.robot_kind!r} is not available") from None
    return binding, robot


def _binding_description(config: ControlServiceConfig) -> dict[str, Any]:
    """Describe the configured binding's action surface without loading it.

    ``describe`` must stay side-effect free, so this reads only the static
    binding definition.  Agent callers use the feature names and the maximum
    chunk length to build a bounded ``execute`` request without importing the
    robot or model packages.
    """

    try:
        binding = binding_definition(config.binding_kind)
    except (KeyError, TypeError):
        # Device-only and other binding-less services still report a stable
        # shape so callers do not need a special case.
        return {
            "kind": config.binding_kind,
            "maximum_chunk_steps": None,
            "action_feature_names": [],
        }
    adapter = binding.adapter_config or {}
    features = adapter.get("action_feature_names")
    names = [str(name) for name in features] if isinstance(features, (list, tuple)) else []
    return {
        "kind": binding.kind,
        "maximum_chunk_steps": binding.maximum_chunk_steps,
        "action_feature_names": names,
    }


def _connect_robot(robot: Any, *, prepare: bool) -> None:
    """Call the optional lifecycle-aware adapter API with legacy fallback."""

    connect = getattr(robot, "connect", None)
    if not callable(connect):
        raise ControlServiceError("robot adapter does not implement connect")
    try:
        parameters = inspect.signature(connect).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "prepare" in parameters:
        connect(prepare=prepare)
    elif prepare:
        connect()
    else:
        raise ControlServiceError("robot adapter does not expose a side-effect-free passive connection")


def _supports_passive_connect(robot: Any) -> bool:
    """Return whether the adapter exposes the explicit passive boundary."""

    connect = getattr(robot, "connect", None)
    if not callable(connect):
        return False
    try:
        return "prepare" in inspect.signature(connect).parameters
    except (TypeError, ValueError):
        return False


def _supports_prepare_operation(robot: Any) -> bool:
    """Return whether ``prepare`` is implemented beyond the base stub."""

    method = getattr(type(robot), "prepare", None)
    return callable(method) and method is not RobotAdapter.prepare


def _prepare_adapter(value: Any) -> None:
    """Prepare a connected adapter, preserving the legacy explicit-connect path."""

    if getattr(value, "_embodirun_prepared_on_connect", False):
        return
    prepare = getattr(value, "prepare", None)
    if not callable(prepare):
        raise ControlServiceError(f"{type(value).__name__} does not expose explicit prepare")
    prepare()


def _connect_and_return(robot: Any) -> Any:
    try:
        _connect_robot(robot, prepare=False)
    except BaseException:
        with contextlib.suppress(BaseException):
            robot.close()
        raise
    return robot


def _camera_capture_profile(item: SensorInput) -> tuple[str, str]:
    """Normalize capture-affecting options for one physical camera source."""

    options = {
        str(key): value
        for key, value in item.options.items()
        if key not in {"device", "path", "serial", "serial_number"}
    }
    return item.kind, json.dumps(options, sort_keys=True, default=str)


def _configured_resource(
    config: ControlServiceConfig,
    *,
    kind: str | None = None,
    identity: ResourceIdentity | None = None,
) -> Mapping[str, Any] | None:
    for item in config.device_resources:
        if kind is not None and item.get("kind") != kind:
            continue
        if identity is not None:
            if item.get("identity") == identity.key:
                return item
            node = item.get("node", config.node_id)
            value = item.get("value")
            if isinstance(node, str) and isinstance(value, str):
                try:
                    if (
                        canonical_resource_identity(
                            node,
                            str(item.get("kind", "sensor")),
                            value,
                        ).key
                        == identity.key
                    ):
                        return item
                except ValueError:
                    pass
            continue
        return item
    return None


def _configured_resources(
    config: ControlServiceConfig,
    *,
    kind: str,
) -> tuple[Mapping[str, Any], ...]:
    """Return all descriptors for one physical resource kind in config order."""

    return tuple(item for item in config.device_resources if item.get("kind") == kind)


def _robot_component_identities(
    config: ControlServiceConfig,
) -> tuple[ResourceIdentity, ...]:
    """Return physical robot components attached to the primary descriptor."""

    resources = _configured_resources(config, kind="robot")
    primary = _resource_identity_from_config(config, resources[0] if resources else None)
    identities: dict[str, ResourceIdentity] = {}
    for item in resources[1:]:
        identity = _resource_identity_from_config(config, item)
        if identity.key != primary.key:
            identities[identity.key] = identity
    if config.robot_kind == "lerobot.bi_so101":
        ports = (
            config.robot_options.get("left_port"),
            config.robot_options.get("right_port"),
        )
    else:
        ports = (config.robot_options.get("port"),)
    for port in ports:
        if not isinstance(port, str) or not port.strip():
            continue
        identity = canonical_resource_identity(primary.node, "robot", port)
        if identity.key != primary.key:
            identities.setdefault(identity.key, identity)
    return tuple(identities[key] for key in sorted(identities))


def _resource_identity_from_config(
    config: ControlServiceConfig,
    resource: Mapping[str, Any] | None,
) -> ResourceIdentity:
    if resource is None:
        source = config.robot_options.get("port") or config.robot_options.get("host")
        if not isinstance(source, str) or not source.strip():
            source = config.robot_id
        return canonical_resource_identity(config.node_id, "robot", source)
    node = resource.get("node", config.node_id)
    kind = resource.get("kind", "robot")
    value = resource.get("value")
    if not all(isinstance(item, str) and item.strip() for item in (node, kind, value)):
        identity_value = resource.get("identity")
        if not isinstance(identity_value, str) or identity_value.count(":") < 2:
            raise ControlServiceError("device resource identity is malformed")
        node, kind, value = identity_value.split(":", 2)
    return canonical_resource_identity(str(node), str(kind), str(value))


def _sensor_identity(
    config: ControlServiceConfig,
    input_value: SensorInput,
) -> ResourceIdentity:
    # An explicit device/serial is already a physical selector.  Keep this
    # path for legacy static configs and for drivers whose source is not
    # represented in Host resource metadata.
    source = input_value.options.get("device") or input_value.options.get("path")
    if source is None:
        source = input_value.options.get("serial") or input_value.options.get("serial_number")
    if isinstance(source, str) and source.strip():
        return canonical_resource_identity(config.node_id, "sensor", source)

    resources = tuple(item for item in config.device_resources if item.get("kind") == "sensor")
    matches: list[Mapping[str, Any]] = []
    for resource in resources:
        aliases = resource.get("sensor_ids")
        if isinstance(aliases, str):
            aliases = (aliases,)
        elif not isinstance(aliases, Sequence) or isinstance(aliases, (bytes, bytearray)):
            aliases = ()
        candidate_ids = tuple(
            value for value in (*aliases, resource.get("sensor_id")) if isinstance(value, str) and value.strip()
        )
        if input_value.sensor_id in candidate_ids:
            matches.append(resource)
    if len(matches) == 1:
        return _resource_identity_from_config(config, matches[0])
    if len(matches) > 1:
        raise ControlServiceError(f"sensor input {input_value.sensor_id!r} matches multiple resources")
    if not resources:
        # A legacy config without Host resource metadata uses the logical
        # sensor id as its stable source identity.
        return canonical_resource_identity(config.node_id, "sensor", input_value.sensor_id)
    if len(resources) == 1:
        # Preserve the old fallback only for genuinely aliasless static
        # configurations.  Generated Host metadata names its logical sensor
        # explicitly; silently binding an unmatched input to that one physical
        # resource would route the wrong camera.
        resource = resources[0]
        if "sensor_id" in resource or "sensor_ids" in resource:
            raise ControlServiceError(
                f"sensor input {input_value.sensor_id!r} does not identify the configured sensor resource"
            )
        return _resource_identity_from_config(config, resource)
    raise ControlServiceError(
        f"sensor input {input_value.sensor_id!r} does not identify one of {len(resources)} configured sensor resources"
    )


def _camera_group_key(config: ControlServiceConfig) -> str:
    identities = sorted(f"{_sensor_identity(config, item).key}={item.name}" for item in config.inputs)
    return "|".join(identities)


def _runtime_camera_requests(
    config: ControlServiceConfig,
) -> tuple[tuple[str, str], ...]:
    """Map configured runtime names to physical source IDs in input order."""

    return tuple((_sensor_identity(config, item).key, item.name) for item in config.inputs)


def _observation_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return dict(value)
    values = getattr(value, "values", None)
    metadata = getattr(value, "metadata", None)
    timestamp = getattr(value, "timestamp_s", None)
    if values is not None and timestamp is not None:
        return {
            "timestamp_s": timestamp,
            "values": _json_safe(values),
            "metadata": _json_safe(metadata or {}),
        }
    return value


def _json_safe(value: Any) -> Any:
    """Convert immutable mapping views in adapter snapshots to JSON data."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _shutdown_client(client: Any) -> None:
    shutdown = getattr(client, "shutdown", None)
    if callable(shutdown):
        shutdown()


def _package_version() -> str:
    """Report installed package metadata without inventing a source hash."""

    for distribution in ("embodirun", "rlinf-deploy"):
        try:
            return package_version(distribution)
        except PackageNotFoundError:  # editable/source checkouts may lack metadata
            continue
    return "unknown"


def _payload_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} payload must be an object")
    return value


def _robot_action(value: object) -> RobotAction:
    root = _payload_mapping(value, "manual action")
    timestamp_s = root.get("timestamp_s", time.time())
    if isinstance(timestamp_s, bool) or not isinstance(timestamp_s, (int, float)):
        raise ValueError("manual action timestamp_s must be numeric")
    values = _payload_mapping(root.get("values"), "manual action values")
    metadata = root.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("manual action metadata must be an object")
    return RobotAction(float(timestamp_s), dict(values), dict(metadata))


__all__ = [
    "ControlService",
    "ControlServiceError",
    "ControlTaskRejected",
    "create_inference_client",
]
