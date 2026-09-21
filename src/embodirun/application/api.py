"""Thin application semantics over ControlService, the arbiter, and jobs.

The application layer owns request identity, authorization, observation
freshness, and response shaping.  It does not open hardware, implement IK, or
duplicate adapter limits.  Direct actions use the existing arbiter's
``agent``/``replay`` sources, while legacy model tasks use the same
``JobRegistry`` and preserve their synchronous ``TaskResult`` response.
"""

from __future__ import annotations

import base64
import inspect
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

from embodirun.devices.execution.arbitration import (
    CommandSource,
    RobotControlArbiter,
)
from embodirun.devices.observations.values import ObservationSnapshot
from embodirun.robots import RobotAction

from .auth import AuthPolicy, Role
from .contracts import TaskRequest, TaskResult
from .direct_execution import (
    DirectExecutionRunner,
    action_payload,
)
from .jobs import (
    JobConflict,
    JobHandle,
    JobNotFound,
    JobRecord,
    JobRegistry,
    PhysicalStatus,
)


class ApplicationError(RuntimeError):
    """Base error for application operations."""


class ApplicationUnsupported(ApplicationError):
    """The requested operation is not available from the configured owner."""


class ApplicationProposalFailed(ApplicationError):
    """The non-executing proposal request could not be completed."""


class ApplicationStaleObservation(ApplicationError):
    """The referenced shared snapshot is unavailable or fails freshness checks."""


class ApplicationInvalidRequest(ValueError, ApplicationError):
    """The application payload is malformed or outside its bounded shape."""


class ControlApplication:
    """Expose stable application operations without owning a second driver."""

    def __init__(
        self,
        service: Any,
        *,
        registry: JobRegistry | None = None,
        arbiter_provider: Callable[[], RobotControlArbiter] | None = None,
        observation_store: Any | None = None,
        auth_policy: AuthPolicy | None = None,
        direct_runner: DirectExecutionRunner | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        max_steps: int = 50,
        default_control_hz: float = 10.0,
        max_segment_duration_s: float = 30.0,
        max_observation_age_ns: int = 500_000_000,
        max_observation_skew_ns: int | None = 100_000_000,
        max_media_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        if service is None:
            raise TypeError("service must be provided")
        if registry is not None and not isinstance(registry, JobRegistry):
            raise TypeError("registry must be a JobRegistry")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")
        if (
            isinstance(default_control_hz, bool)
            or not isinstance(default_control_hz, (int, float))
            or not math.isfinite(default_control_hz)
            or default_control_hz <= 0
        ):
            raise ValueError("default_control_hz must be finite and positive")
        if (
            isinstance(max_segment_duration_s, bool)
            or not isinstance(max_segment_duration_s, (int, float))
            or not math.isfinite(max_segment_duration_s)
            or max_segment_duration_s <= 0
        ):
            raise ValueError("max_segment_duration_s must be finite and positive")
        if (
            isinstance(max_observation_age_ns, bool)
            or not isinstance(max_observation_age_ns, int)
            or max_observation_age_ns < 0
        ):
            raise ValueError("max_observation_age_ns must be a non-negative integer")
        if max_observation_skew_ns is not None and (
            isinstance(max_observation_skew_ns, bool)
            or not isinstance(max_observation_skew_ns, int)
            or max_observation_skew_ns < 0
        ):
            raise ValueError("max_observation_skew_ns must be a non-negative integer or None")
        if isinstance(max_media_bytes, bool) or not isinstance(max_media_bytes, int) or max_media_bytes <= 0:
            raise ValueError("max_media_bytes must be a positive integer")
        self.service = service
        self.registry = registry or JobRegistry()
        self._arbiter_provider = arbiter_provider
        self.observation_store = observation_store
        self.auth = auth_policy or AuthPolicy()
        self.clock_ns = clock_ns
        self.max_steps = max_steps
        self.default_control_hz = float(default_control_hz)
        self.max_segment_duration_s = float(max_segment_duration_s)
        self.max_observation_age_ns = max_observation_age_ns
        self.max_observation_skew_ns = max_observation_skew_ns
        self.max_media_bytes = max_media_bytes
        self._direct_runner = direct_runner
        self._runner_lock = threading.Lock()

    def describe(
        self,
        *,
        caller_id: str,
        session_id: str,
        token: str | None = None,
    ) -> dict[str, Any]:
        self._scope(caller_id, session_id)
        self._authorize(token, Role.OBSERVER, caller_id=caller_id, session_id=session_id)
        base = dict(self.service.describe())
        capabilities = base.setdefault("capabilities", {})
        capabilities = dict(capabilities) if isinstance(capabilities, Mapping) else {}
        capabilities["model_execute"] = bool(capabilities.get("execute", False))
        capabilities["direct_execute"] = self._can_direct_execute()
        base["capabilities"] = capabilities
        base["application"] = {
            "describe": True,
            "observe": True,
            "propose": bool(capabilities.get("propose", False)),
            "execute": self._can_direct_execute(),
            "inspect": True,
            "cancel": True,
            "stop": True,
            "media": self.observation_store is not None,
        }
        base["application_authentication"] = (
            "token" if self.auth.token_authentication_enabled else "trusted_ssh_loopback"
        )
        return base

    def observe(
        self,
        *,
        caller_id: str,
        session_id: str,
        token: str | None = None,
        runtime_id: str | None = None,
        observation_id: str | None = None,
        include_robot: bool = False,
        max_age_ns: int | None = None,
        max_skew_ns: int | None = None,
    ) -> dict[str, Any]:
        self._scope(caller_id, session_id)
        self._authorize(token, Role.OBSERVER, caller_id=caller_id, session_id=session_id)
        if observation_id is None:
            # The service selects the runtime's camera view and performs its
            # passive observer lease.  It still returns the shared snapshot
            # identity, so callers can pin it for a later action.
            return self._service_observe(runtime_id=runtime_id, include_robot=include_robot)
        snapshot = self._snapshot(observation_id)
        if snapshot is not None:
            age_ns, fresh = self._freshness(
                snapshot,
                max_age_ns=max_age_ns,
                max_skew_ns=max_skew_ns,
                required=False,
            )
            get_snapshot = getattr(self.service, "get_snapshot", None)
            if callable(get_snapshot):
                payload = dict(
                    get_snapshot(
                        observation_id,
                        runtime_id=runtime_id,
                        include_robot=include_robot,
                    )
                )
            else:
                payload = _snapshot_payload(snapshot)
            if "observation_id" not in payload and isinstance(payload.get("snapshot_id"), str):
                payload["observation_id"] = payload["snapshot_id"]
            payload["age_ns"] = age_ns
            payload["fresh"] = fresh
            # A shared observation ID is a consistency boundary.  Do not mix it
            # with a newly captured robot state from the service.
            if include_robot and "robot" not in payload:
                payload["robot"] = _json_value(snapshot.state)
            return payload
        if observation_id is not None:
            raise ApplicationStaleObservation(f"observation {observation_id!r} is not available in the shared store")
        return self._service_observe(runtime_id=runtime_id, include_robot=include_robot)

    def propose(
        self,
        *,
        caller_id: str,
        session_id: str,
        request_id: str,
        observation_id: str,
        instruction: str,
        runtime_id: str | None = None,
        timeout_s: float = 10.0,
        token: str | None = None,
        max_age_ns: int | None = None,
        max_skew_ns: int | None = None,
    ) -> dict[str, Any]:
        """Generate one mapped proposal from an existing shared snapshot.

        Proposal generation is observer-authorized and has no job/arbiter
        side effect.  The returned actions remain proposals until a later
        ``execute`` request submits them with a new request ID.
        """

        self._scope(caller_id, session_id)
        self._authorize(token, Role.OBSERVER, caller_id=caller_id, session_id=session_id)
        if not isinstance(request_id, str) or not request_id.strip():
            raise ApplicationInvalidRequest("request_id must be a non-empty string")
        if not isinstance(observation_id, str) or not observation_id.strip():
            raise ApplicationInvalidRequest("observation_id must be a non-empty string")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ApplicationInvalidRequest("instruction must be a non-empty string")
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ApplicationInvalidRequest("timeout_s must be finite and positive")
        if self.observation_store is None:
            raise ApplicationUnsupported("proposal requires a shared observation store")
        snapshot = self._snapshot(observation_id)
        if snapshot is None:
            raise ApplicationStaleObservation(f"observation {observation_id!r} is not available")
        self._freshness(
            snapshot,
            max_age_ns=max_age_ns,
            max_skew_ns=max_skew_ns,
            required=True,
        )
        from .proposals import generate_proposal

        try:
            return generate_proposal(
                self.service,
                observation_id=observation_id,
                instruction=instruction,
                runtime_id=runtime_id,
                request_id=request_id,
                timeout_s=float(timeout_s),
            )
        except AttributeError as error:
            raise ApplicationUnsupported("configured control service does not support proposals") from error
        except ApplicationError:
            raise
        except Exception as error:
            # A proposal failure is a non-executing request failure.  Do not
            # let backend/parser or transient transport exceptions look like
            # an unsupported capability.
            raise ApplicationProposalFailed(f"proposal generation failed: {error}") from error

    def media(
        self,
        *,
        caller_id: str,
        session_id: str,
        observation_id: str,
        frame_name: str | None = None,
        runtime_id: str | None = None,
        include_data: bool = False,
        token: str | None = None,
    ) -> dict[str, Any]:
        """Return bounded shared-snapshot media references, never raw SDK reads."""

        self._scope(caller_id, session_id)
        self._authorize(token, Role.OBSERVER, caller_id=caller_id, session_id=session_id)
        snapshot = self._snapshot(observation_id)
        if snapshot is None:
            raise ApplicationUnsupported("shared observation media is not configured")
        frames = [frame for frame in snapshot.cameras if frame_name is None or frame.name == frame_name]
        get_media = getattr(self.service, "get_media", None)
        if frame_name is not None and callable(get_media):
            try:
                frame = get_media(
                    observation_id,
                    frame_name,
                    runtime_id=runtime_id,
                )
            except TypeError:
                frame = get_media(observation_id, frame_name)
            if frame not in frames:
                frames = [frame]
        if frame_name is not None and not frames:
            raise ApplicationStaleObservation(f"frame {frame_name!r} is not present in observation {observation_id!r}")
        media = []
        for frame in frames:
            if include_data and len(frame.data) > self.max_media_bytes:
                raise ApplicationUnsupported(f"frame {frame.name!r} exceeds the bounded media response")
            reference = _media_reference(snapshot, frame)
            if include_data:
                reference["data_base64"] = base64.b64encode(frame.data).decode("ascii")
            media.append(reference)
        return {
            "observation_id": snapshot.observation_id,
            "media": media,
        }

    def recording_start(
        self,
        *,
        caller_id: str,
        session_id: str,
        token: str | None = None,
    ) -> dict[str, Any]:
        self._scope(caller_id, session_id)
        self._authorize(token, Role.CONTROLLER, caller_id=caller_id, session_id=session_id)
        method = getattr(self.service, "start_recording", None)
        if not callable(method):
            raise ApplicationUnsupported("recording is not configured")
        return _json_value(method())

    def recording_stop(
        self,
        *,
        caller_id: str,
        session_id: str,
        timeout_s: float = 1.0,
        token: str | None = None,
    ) -> dict[str, Any]:
        self._scope(caller_id, session_id)
        self._authorize(token, Role.CONTROLLER, caller_id=caller_id, session_id=session_id)
        method = getattr(self.service, "stop_recording", None)
        if not callable(method):
            raise ApplicationUnsupported("recording is not configured")
        return _json_value(method(timeout_s=timeout_s))

    def recording_status(
        self,
        *,
        caller_id: str,
        session_id: str,
        token: str | None = None,
    ) -> dict[str, Any]:
        self._scope(caller_id, session_id)
        self._authorize(token, Role.OBSERVER, caller_id=caller_id, session_id=session_id)
        method = getattr(self.service, "recording_status", None)
        if not callable(method):
            raise ApplicationUnsupported("recording is not configured")
        return _json_value(method())

    def recording_get(
        self,
        *,
        caller_id: str,
        session_id: str,
        observation_id: str,
        token: str | None = None,
    ) -> dict[str, Any] | None:
        self._scope(caller_id, session_id)
        self._authorize(token, Role.OBSERVER, caller_id=caller_id, session_id=session_id)
        method = getattr(self.service, "get_record", None)
        if not callable(method):
            raise ApplicationUnsupported("recording is not configured")
        value = method(observation_id)
        return None if value is None else _json_value(value)

    def execute(
        self,
        *,
        caller_id: str,
        session_id: str,
        request_id: str,
        action: RobotAction | Mapping[str, Any] | Sequence[Any],
        source: CommandSource | str = CommandSource.AGENT,
        observation_id: str | None = None,
        runtime_id: str | None = None,
        steps: int = 1,
        control_hz: float | None = None,
        wait: bool = False,
        timeout_s: float | None = None,
        token: str | None = None,
        max_age_ns: int | None = None,
        max_skew_ns: int | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._scope(caller_id, session_id)
        self._authorize(token, Role.CONTROLLER, caller_id=caller_id, session_id=session_id)
        if not isinstance(request_id, str) or not request_id.strip():
            raise ApplicationInvalidRequest("request_id must be a non-empty string")
        if runtime_id is not None:
            runtime_id = self._validate_runtime(runtime_id)
        action_values = _actions(action, steps=steps, max_steps=self.max_steps)
        frequency = self.default_control_hz if control_hz is None else control_hz
        _validate_hz(frequency)
        source_value = _source(source)
        if parameters is not None and not isinstance(parameters, Mapping):
            raise ApplicationInvalidRequest("parameters must be an object")

        # Request identity is checked before selecting or validating a fresh
        # snapshot.  A retry must be able to inspect the original result after
        # that snapshot expires, and an omitted observation ID must not become
        # a new request merely because a newer snapshot is now latest.
        self._validate_requested_observation_metadata(action_values, observation_id)
        requested_identity = _identity_parameters(
            action_values,
            source=source_value,
            control_hz=float(frequency),
            observation_id=observation_id,
            runtime_id=runtime_id,
            selected_observation_id=None,
            parameters=parameters,
        )
        existing = self._existing_identity(
            caller_id,
            session_id,
            request_id,
            requested_identity,
        )
        if existing is not None:
            handle = self.registry.get_handle(caller_id, session_id, request_id)
            if wait:
                handle.wait(timeout_s=timeout_s)
            return _handle_payload(handle)

        selected_observation_id = self._validate_action_observation(
            observation_id,
            max_age_ns=max_age_ns,
            max_skew_ns=max_skew_ns,
        )
        if selected_observation_id is not None:
            action_values = tuple(_with_observation_id(value, selected_observation_id) for value in action_values)
        runner = self._runner()
        identity_parameters = _identity_parameters(
            action_values,
            source=source_value,
            control_hz=float(frequency),
            observation_id=observation_id,
            runtime_id=runtime_id,
            selected_observation_id=selected_observation_id,
            parameters=parameters,
        )
        task = _DirectTask(
            runner,
            action_values,
            source_value,
            float(frequency),
            selected_observation_id,
            preflight=lambda: self._validate_action_observation(
                selected_observation_id,
                max_age_ns=max_age_ns,
                max_skew_ns=max_skew_ns,
            ),
        )
        handle = self.registry.submit(
            caller_id,
            session_id,
            request_id,
            identity_parameters,
            task.execute,
            task.cancel,
            wait=False,
            inference_requested=False,
        )
        if wait:
            handle.wait(timeout_s=timeout_s)
        return _handle_payload(handle)

    def _existing_identity(
        self,
        caller_id: str,
        session_id: str,
        request_id: str,
        requested_identity: Mapping[str, Any],
    ) -> JobRecord | None:
        try:
            record = self.registry.inspect(caller_id, session_id, request_id)
        except JobNotFound:
            return None
        if _canonical_identity(record.parameters) != _canonical_identity(requested_identity):
            raise JobConflict(f"request_id {request_id!r} already has different parameters")
        return record

    def _validate_runtime(self, runtime_id: str) -> str:
        """Refuse an execute request for a runtime this owner does not serve."""

        if not isinstance(runtime_id, str) or not runtime_id.strip():
            raise ApplicationInvalidRequest("runtime_id must be a non-empty string")
        config = getattr(self.service, "config", None)
        if config is None:
            raise ApplicationUnsupported("control service does not expose its served runtime identities")
        known: set[str] = set()
        primary = getattr(config, "runtime_id", None)
        if isinstance(primary, str) and primary:
            known.add(primary)
        profiles = getattr(config, "runtime_profiles", None)
        if isinstance(profiles, Mapping):
            known.update(str(name) for name in profiles)
        if runtime_id not in known:
            raise ApplicationInvalidRequest(f"runtime {runtime_id!r} is not served by this control service")
        return runtime_id

    def inspect(
        self,
        *,
        caller_id: str,
        session_id: str,
        request_id: str,
        token: str | None = None,
    ) -> dict[str, Any]:
        self._scope(caller_id, session_id)
        self._authorize(token, Role.OBSERVER, caller_id=caller_id, session_id=session_id)
        return _record_payload(self.registry.inspect(caller_id, session_id, request_id))

    def cancel(
        self,
        *,
        caller_id: str,
        session_id: str,
        request_id: str,
        token: str | None = None,
    ) -> dict[str, Any]:
        self._scope(caller_id, session_id)
        self._authorize(token, Role.CONTROLLER, caller_id=caller_id, session_id=session_id)
        return _record_payload(self.registry.cancel(caller_id, session_id, request_id))

    def stop(
        self,
        *,
        caller_id: str,
        session_id: str,
        request_id: str,
        token: str | None = None,
    ) -> dict[str, Any]:
        """Request stop for exactly one job; never stop a later owner."""

        result = self.cancel(
            caller_id=caller_id,
            session_id=session_id,
            request_id=request_id,
            token=token,
        )
        result["stop_requested"] = True
        return result

    def legacy_task(
        self,
        request: TaskRequest,
        *,
        caller_id: str,
        session_id: str,
        token: str | None = None,
        wait: bool = True,
        timeout_s: float | None = None,
    ) -> TaskResult | JobHandle | dict[str, Any]:
        """Keep the old TaskResult shape while routing acceptance through jobs."""

        self._scope(caller_id, session_id)
        self._authorize(token, Role.CONTROLLER, caller_id=caller_id, session_id=session_id)
        if not isinstance(request, TaskRequest):
            raise TypeError("request must be a TaskRequest")
        result = self.registry.submit_task(
            request,
            caller_id=caller_id,
            session_id=session_id,
            execute=lambda cancel_event: self._execute_legacy(request, cancel_event),
            cancel=self._cancel_legacy_task,
            wait=wait,
            timeout_s=timeout_s,
        )
        if wait:
            return result
        if isinstance(result, JobHandle):
            return _handle_payload(result)
        return result

    def close(self, *, wait_s: float = 0.1) -> bool:
        """Close the registry after bounded callback/worker draining."""

        return self.registry.close(wait_s=wait_s)

    def _cancel_legacy_task(self, cancel_event: threading.Event) -> Mapping[str, Any]:
        """Cancel only the registry-owned legacy task token.

        A server integration may expose ``cancel_task(cancel_event)`` and use
        this exact event when it calls ``begin_model_task``.  An older service
        that cannot accept that hook is reported as unconfirmed; looking up a
        process-global current task here could cancel a newer owner.
        """

        cancel_event.set()
        callback = getattr(self.service, "cancel_task", None)
        if not callable(callback):
            callback = getattr(self.service, "cancel_automatic_task", None)
        if not callable(callback):
            return {
                "physical_status": PhysicalStatus.STOP_UNCONFIRMED.value,
                "stop_confirmed": False,
                "error": "service does not expose task-scoped cancellation",
            }
        try:
            value = callback(cancel_event)
            if isinstance(value, Mapping):
                return dict(value)
            accepted = bool(value)
        except Exception as error:
            return {
                "physical_status": PhysicalStatus.STOP_UNCONFIRMED.value,
                "stop_confirmed": False,
                "error": str(error),
            }
        return {
            "physical_status": (
                PhysicalStatus.STOP_REQUESTED.value if accepted else PhysicalStatus.STOP_UNCONFIRMED.value
            ),
            "stop_confirmed": False,
        }

    def _execute_legacy(self, request: TaskRequest, cancel_event: threading.Event) -> Any:
        """Call legacy services with the optional task-scoped hook."""

        execute = self.service.execute
        try:
            parameters = inspect.signature(execute).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "cancel_event" in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        ):
            return execute(request, cancel_event=cancel_event)
        return execute(request)

    def _runner(self) -> DirectExecutionRunner:
        if self._direct_runner is not None:
            return self._direct_runner
        with self._runner_lock:
            if self._direct_runner is None:
                self._direct_runner = DirectExecutionRunner(
                    self._resolve_arbiter,
                    max_steps=self.max_steps,
                    max_duration_s=self.max_segment_duration_s,
                    simulated=self._service_is_simulated(),
                )
            return self._direct_runner

    def _service_is_simulated(self) -> bool:
        config = getattr(self.service, "config", None)
        robot_kind = getattr(config, "robot_kind", "")
        return isinstance(robot_kind, str) and robot_kind.startswith("simulated.")

    def _resolve_arbiter(self) -> RobotControlArbiter:
        if self._arbiter_provider is not None:
            arbiter = self._arbiter_provider()
        else:
            provider = getattr(self.service, "get_control_arbiter", None)
            arbiter = provider() if callable(provider) else getattr(self.service, "_current_arbiter", None)
        if arbiter is None or any(
            not callable(getattr(arbiter, name, None))
            for name in (
                "begin_automatic_task",
                "submit_agent",
                "submit_replay",
                "cancel_automatic_work",
            )
        ):
            raise ApplicationUnsupported("direct execution requires a prepared control arbiter")
        return arbiter

    def _can_direct_execute(self) -> bool:
        # describe() is a read-only capability query.  Calling a provider here
        # could open hardware or allocate an owner before execute() asks for
        # it.  An injected runner/provider or an explicit service provider is
        # enough to advertise that the owner can be resolved later.
        if self._direct_runner is not None or self._arbiter_provider is not None:
            return True
        return callable(getattr(self.service, "get_control_arbiter", None))

    def _snapshot(self, observation_id: str | None) -> ObservationSnapshot | None:
        if self.observation_store is None:
            return None
        try:
            if observation_id is None:
                return self.observation_store.latest()
            return self.observation_store.get(observation_id)
        except Exception as error:
            raise ApplicationStaleObservation(str(error)) from error

    def _freshness(
        self,
        snapshot: ObservationSnapshot,
        *,
        max_age_ns: int | None,
        max_skew_ns: int | None,
        required: bool,
    ) -> tuple[int | None, bool]:
        age_limit = self.max_observation_age_ns if max_age_ns is None else max_age_ns
        skew_limit = self.max_observation_skew_ns if max_skew_ns is None else max_skew_ns
        if isinstance(age_limit, bool) or not isinstance(age_limit, int) or age_limit < 0:
            raise ApplicationInvalidRequest("max_age_ns must be a non-negative integer")
        if skew_limit is not None and (
            isinstance(skew_limit, bool) or not isinstance(skew_limit, int) or skew_limit < 0
        ):
            raise ApplicationInvalidRequest("max_skew_ns must be a non-negative integer or None")
        now_ns = self.clock_ns()
        age = snapshot.age_ns(now_ns)
        fresh = snapshot.is_fresh(
            now_ns=now_ns,
            now_clock_domain="host_monotonic_ns",
            max_age_ns=age_limit,
            max_skew_ns=skew_limit,
        )
        if required and not fresh:
            raise ApplicationStaleObservation(f"observation {snapshot.observation_id!r} is stale or timing is unknown")
        return age, fresh

    def _validate_action_observation(
        self,
        observation_id: str | None,
        *,
        max_age_ns: int | None,
        max_skew_ns: int | None,
    ) -> str | None:
        if self.observation_store is None:
            return observation_id
        if observation_id is None and not self._observation_required_for_action():
            # A configured device-only robot may have no camera/state source.
            # Its direct action still enters the prepared arbiter; there is no
            # absent snapshot to call stale in that explicitly supported mode.
            return None
        snapshot = self._snapshot(observation_id)
        if snapshot is None:
            raise ApplicationStaleObservation("no shared observation is available")
        self._freshness(
            snapshot,
            max_age_ns=max_age_ns,
            max_skew_ns=max_skew_ns,
            required=True,
        )
        return snapshot.observation_id

    def _observation_required_for_action(self) -> bool:
        config = getattr(self.service, "config", None)
        inputs = getattr(config, "inputs", None)
        if inputs is None:
            # Generic injected services may use the store without exposing
            # static input metadata, so preserve the strict shared-snapshot
            # behavior for that composition.
            return True
        return bool(inputs)

    @staticmethod
    def _validate_requested_observation_metadata(
        actions: Sequence[RobotAction],
        observation_id: str | None,
    ) -> None:
        if observation_id is None:
            return
        for action in actions:
            requested = action.metadata.get("observation_id")
            if requested is not None and requested != observation_id:
                raise ApplicationInvalidRequest(
                    "action metadata observation_id conflicts with the request observation_id"
                )

    def _service_observe(self, *, runtime_id: str | None, include_robot: bool) -> dict[str, Any]:
        value = self.service.observe(runtime_id=runtime_id, include_robot=include_robot)
        if not isinstance(value, Mapping):
            raise ApplicationError("control service returned an invalid observation")
        payload = dict(value)
        # ControlService historically names the shared consistency token
        # ``snapshot_id``.  The application API exposes the Agent-facing
        # ``observation_id`` while retaining the old field for compatibility.
        if "observation_id" not in payload and isinstance(payload.get("snapshot_id"), str):
            payload["observation_id"] = payload["snapshot_id"]
        # The live service path returns the current immutable snapshot but
        # historically omitted the application-level freshness fields that
        # are present when an observation ID is supplied.  Resolve that same
        # snapshot here so every Agent-facing observe response has one
        # freshness contract; never infer freshness from HTTP receipt time.
        payload.setdefault("observation_id", None)
        observation_id = payload.get("observation_id")
        if isinstance(observation_id, str) and "fresh" not in payload:
            try:
                snapshot = self._snapshot(observation_id)
            except ApplicationStaleObservation:
                snapshot = None
            if snapshot is not None:
                age_ns, fresh = self._freshness(
                    snapshot,
                    max_age_ns=None,
                    max_skew_ns=None,
                    required=False,
                )
                payload["age_ns"] = age_ns
                payload["fresh"] = fresh
                payload["freshness"] = "fresh" if fresh else "stale"
                if not fresh:
                    payload["freshness_reason"] = "snapshot is stale or timing is unknown"
            else:
                payload["age_ns"] = None
                payload["fresh"] = False
                payload["freshness"] = "unknown"
                payload["freshness_reason"] = "live observation is not retained in the shared store"
        elif "fresh" not in payload:
            # A service that cannot publish a shared snapshot must never be
            # treated as fresh merely because its HTTP request succeeded.
            payload["age_ns"] = None
            payload["fresh"] = False
            payload["freshness"] = "unknown"
            payload["freshness_reason"] = "service returned no shared observation identity"
        return payload

    def _authorize(
        self,
        token: str | None,
        role: Role,
        *,
        caller_id: str,
        session_id: str,
    ) -> None:
        self.auth.authorize_scope(
            token,
            role,
            caller_id=caller_id,
            session_id=session_id,
        )

    @staticmethod
    def _scope(caller_id: str, session_id: str) -> None:
        for name, value in (("caller_id", caller_id), ("session_id", session_id)):
            if not isinstance(value, str) or not value.strip():
                raise ApplicationInvalidRequest(f"{name} must be a non-empty string")


class _DirectTask:
    def __init__(
        self,
        runner: DirectExecutionRunner,
        actions: tuple[RobotAction, ...],
        source: CommandSource,
        control_hz: float,
        observation_id: str | None,
        preflight: Callable[[], object] | None = None,
    ) -> None:
        self.runner = runner
        self.actions = actions
        self.source = source
        self.control_hz = control_hz
        self.observation_id = observation_id
        self.preflight = preflight

    def execute(self, cancel_event: threading.Event) -> Mapping[str, Any]:
        if self.preflight is not None:
            self.preflight()
        return self.runner.run(
            self.actions,
            source=self.source,
            cancel_event=cancel_event,
            control_hz=self.control_hz,
            observation_id=self.observation_id,
        ).to_dict()

    def cancel(self, cancel_event: threading.Event) -> Mapping[str, Any]:
        return self.runner.cancel(cancel_event)


def _actions(
    value: RobotAction | Mapping[str, Any] | Sequence[Any],
    *,
    steps: int,
    max_steps: int,
) -> tuple[RobotAction, ...]:
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ApplicationInvalidRequest("steps must be a positive integer")
    if steps > max_steps:
        raise ApplicationInvalidRequest(f"steps must not exceed {max_steps}")
    if isinstance(value, (RobotAction, Mapping)):
        action = _action(value)
        return tuple(action for _ in range(steps))
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ApplicationInvalidRequest("action must be an object or action list")
    if len(value) == 0 or len(value) > max_steps:
        raise ApplicationInvalidRequest(f"action list must contain 1..{max_steps} actions")
    actions = tuple(_action(item) for item in value)
    if steps != 1 and steps != len(actions):
        raise ApplicationInvalidRequest("steps must match an explicit action list")
    return actions


def _action(value: RobotAction | Mapping[str, Any]) -> RobotAction:
    if isinstance(value, RobotAction):
        action_payload(value)
        return value
    if not isinstance(value, Mapping):
        raise ApplicationInvalidRequest("each action must be an object")
    timestamp = value.get("timestamp_s", 0.0)
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
        raise ApplicationInvalidRequest("action timestamp_s must be finite")
    values = value.get("values")
    metadata = value.get("metadata", {})
    if not isinstance(values, Mapping) or not isinstance(metadata, Mapping):
        raise ApplicationInvalidRequest("action values and metadata must be objects")
    action = RobotAction(timestamp_s=float(timestamp), values=dict(values), metadata=dict(metadata))
    action_payload(action)
    return action


def _with_observation_id(action: RobotAction, observation_id: str) -> RobotAction:
    """Carry the selected shared snapshot into recorder-facing action metadata."""

    existing = action.metadata.get("observation_id")
    if existing is not None and existing != observation_id:
        raise ApplicationInvalidRequest("action metadata observation_id conflicts with the request observation_id")
    if existing == observation_id:
        return action
    metadata = dict(action.metadata)
    metadata["observation_id"] = observation_id
    return RobotAction(
        timestamp_s=action.timestamp_s,
        values=action.values,
        metadata=metadata,
    )


def _source(value: CommandSource | str) -> CommandSource:
    try:
        source = value if isinstance(value, CommandSource) else CommandSource(value)
    except (TypeError, ValueError) as error:
        raise ApplicationInvalidRequest("source must be 'agent' or 'replay'") from error
    if source not in {CommandSource.AGENT, CommandSource.REPLAY}:
        raise ApplicationInvalidRequest("source must be 'agent' or 'replay'")
    return source


def _validate_hz(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ApplicationInvalidRequest("control_hz must be finite and positive")


def _handle_payload(handle: JobHandle) -> dict[str, Any]:
    return _record_payload(handle.inspect())


def _identity_parameters(
    actions: Sequence[RobotAction],
    *,
    source: CommandSource,
    control_hz: float,
    observation_id: str | None,
    runtime_id: str | None,
    selected_observation_id: str | None,
    parameters: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build ordinary request values used by the registry's exact comparison.

    ``observation_id`` is the caller's binding.  The selected ID is retained
    separately for diagnostics and execution metadata, so a retry without an
    explicit binding remains the same request when the latest snapshot moves.
    """

    identity: dict[str, Any] = {
        "action": [action_payload(value) for value in actions],
        "source": source.value,
        "steps": len(actions),
        "control_hz": control_hz,
        "observation_id": observation_id,
        "observation_id_explicit": observation_id is not None,
        "runtime_id": runtime_id,
        "selected_observation_id": selected_observation_id,
    }
    if parameters is not None:
        identity["parameters"] = dict(parameters)
    return identity


def _canonical_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize generated observation metadata without hashing request data."""

    normalized = _json_value(value)
    if not isinstance(normalized, dict):
        return {"value": normalized}
    normalized.setdefault("runtime_id", None)
    explicit = normalized.get("observation_id_explicit")
    if not isinstance(explicit, bool):
        explicit = normalized.get("observation_id") is not None
    requested = normalized.get("observation_id")
    normalized.pop("selected_observation_id", None)
    actions = normalized.get("action")
    if isinstance(actions, list):
        for item in actions:
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata")
            if not isinstance(metadata, dict):
                continue
            if explicit:
                metadata["observation_id"] = requested
            else:
                metadata.pop("observation_id", None)
    if not explicit:
        normalized["observation_id"] = None
    normalized["observation_id_explicit"] = explicit
    return normalized


def _record_payload(record: JobRecord) -> dict[str, Any]:
    return {
        "caller_id": record.caller_id,
        "session_id": record.session_id,
        "request_id": record.request_id,
        "run_id": record.run_id,
        "parameters": _json_value(record.parameters),
        "status": record.status.value,
        "dispatch_status": record.dispatch_status.value,
        "inference_status": record.inference_status.value,
        "physical_status": record.physical_status.value,
        "cancel_requested": record.cancel_requested,
        "result": _json_value(record.result),
        "error": record.error,
        "events": [
            {
                "name": event.name,
                "at_s": event.at_s,
                "duration_s": event.duration_s,
                "detail": event.detail,
            }
            for event in record.events
        ],
        "events_truncated": record.events_truncated,
        "created_at_s": record.created_at_s,
        "updated_at_s": record.updated_at_s,
        "started_at_s": record.started_at_s,
        "finished_at_s": record.finished_at_s,
    }


def _snapshot_payload(snapshot: ObservationSnapshot) -> dict[str, Any]:
    return {
        "status": "ok",
        "observation_id": snapshot.observation_id,
        "service_instance_id": snapshot.service_instance_id,
        "generation": snapshot.generation,
        "sequence": snapshot.sequence,
        "state": _json_value(snapshot.state),
        "metadata": _json_value(snapshot.metadata),
        "errors": _json_value(snapshot.errors),
        "stale": snapshot.stale,
        "available": snapshot.available,
        "captured_timestamp_ns": snapshot.captured_timestamp_ns,
        "received_timestamp_ns": snapshot.received_timestamp_ns,
        "published_timestamp_ns": snapshot.published_timestamp_ns,
        "skew_ns": snapshot.skew_ns,
        "media": [_media_reference(snapshot, frame) for frame in snapshot.cameras],
    }


def _media_reference(snapshot: ObservationSnapshot, frame: Any) -> dict[str, Any]:
    return {
        "name": frame.name,
        "mime_type": frame.mime_type,
        "bytes": len(frame.data),
        "captured_timestamp_ns": frame.captured_timestamp_ns,
        "clock_domain": frame.clock_domain,
        "media_ref": (
            f"observation://{snapshot.service_instance_id}/{snapshot.generation}/{snapshot.observation_id}/{frame.name}"
        ),
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "ApplicationError",
    "ApplicationInvalidRequest",
    "ApplicationProposalFailed",
    "ApplicationStaleObservation",
    "ApplicationUnsupported",
    "ControlApplication",
]
