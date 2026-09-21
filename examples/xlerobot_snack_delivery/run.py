#!/usr/bin/env python3
"""Run the XLeRobot chips pickup and handover recipe.

The runner is an orchestration layer. It uses pre-configured Deploy Control
services for observations, VLA proposals, and bounded action segments. It does
not open robot hardware and it deliberately has no credential argument.

Two scoped Control endpoints are required: one that owns the mobile base and
one that owns the arms. Each endpoint is a separate Control service attached to
the same single serial owner; the recipe never sends a base action to an arms
scope (or the reverse) and never asks an owner to widen its scope implicitly.

Use ``--mode plan`` to validate a deployment package, ``--mode dry-run`` to
exercise the state machine without hardware, or ``--mode hardware`` after the
owner and Control services have been started by the deployment operator.
"""

from __future__ import annotations

import argparse
import base64
import importlib
import json
import math
import signal
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

try:  # imported as ``examples.xlerobot_snack_delivery.run``
    from .evidence import (
        ConfigEvidenceNormalizer,
        EvidenceError,
        EvidenceNormalizer,
        EvidenceValue,
        lookup_path,
    )
except ImportError:  # executed directly as ``python .../run.py``
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evidence import (  # type: ignore[no-redef]
        ConfigEvidenceNormalizer,
        EvidenceError,
        EvidenceNormalizer,
        EvidenceValue,
        lookup_path,
    )


FORBIDDEN_KEYS = {"token", "token_file", "secret", "password", "api_key"}

#: Physical statuses that still permit the state machine to advance.
TERMINAL_PHYSICAL_STATUSES = frozenset({"stopped"})

#: Physical statuses that must never advance a segment to the next stage.
BLOCKED_PHYSICAL_STATUSES = frozenset({"unknown", "stop_unconfirmed"})

DEFAULT_PLANNING_MODE = "recorded_route"

_ACTION_CONTRACT: Any | None = None


class RecipeError(RuntimeError):
    """The recipe cannot continue while its evidence or configuration is unsafe."""


class Runtime(Protocol):
    """One scope-bound public Control boundary used by the recipe."""

    simulation: bool
    runtime_id: str
    scope: str

    def observe(self) -> dict[str, Any]: ...

    def propose(
        self,
        *,
        request_id: str,
        observation_id: str,
        instruction: str,
        timeout_s: float,
    ) -> dict[str, Any]: ...

    def execute(
        self,
        action: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        *,
        request_id: str,
        observation_id: str,
        control_hz: float,
    ) -> dict[str, Any]: ...

    def stop(self, request_id: str) -> dict[str, Any]: ...


class RuntimeSet(Protocol):
    """The base and arm owners used by the state machine."""

    simulation: bool
    base: Runtime
    manipulation: Runtime


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RecipeError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RecipeError(f"JSON file {path} must contain an object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _reject_credentials(value: Any, location: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in FORBIDDEN_KEYS or any(
                key_text.endswith(suffix) for suffix in ("_token", "_secret", "_password")
            ):
                raise RecipeError(
                    f"{location}.{key} looks like a credential; configure credentials "
                    "at the deployment/owner boundary, never in a recipe config"
                )
            _reject_credentials(item, f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _reject_credentials(item, f"{location}[{index}]")


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecipeError(f"{name} must be a non-empty string")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecipeError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise RecipeError(f"{name} must be finite")
    return result


def _positive(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result <= 0:
        raise RecipeError(f"{name} must be positive")
    return result


def _contract() -> Any:
    """Load the shared XLeRobot action contract lazily.

    ``plan`` mode must not need the installed package, so the contract is
    imported only when an action is built or validated.
    """

    global _ACTION_CONTRACT
    if _ACTION_CONTRACT is None:
        try:
            from embodirun.robots.lerobot.xlerobot import units as contract
        except ImportError as exc:  # pragma: no cover - bad install
            raise RecipeError(
                "install this repository or set PYTHONPATH=src so the XLeRobot action contract is importable"
            ) from exc
        _ACTION_CONTRACT = contract
    return _ACTION_CONTRACT


def _scope_of(values: Mapping[str, Any], name: str) -> str:
    try:
        return str(_contract().scope_for_values(values))
    except ValueError as exc:
        raise RecipeError(f"{name}: {exc}") from exc


def _action(value: Mapping[str, Any], name: str, *, scope: str) -> dict[str, Any]:
    """Validate one externally supplied action against the canonical contract.

    Externally supplied actions (a recorded route chunk or a VLA proposal) must
    already declare ``action_space`` and the canonical unit of every value.
    The recipe never guesses a unit for an action it did not author.
    """

    if not isinstance(value, Mapping):
        raise RecipeError(f"{name} must be an action object")
    timestamp = _finite(value.get("timestamp_s", 0.0), f"{name}.timestamp_s")
    values = value.get("values")
    metadata = value.get("metadata", {})
    if not isinstance(values, Mapping) or not values:
        raise RecipeError(f"{name}.values must be a non-empty object")
    if not isinstance(metadata, Mapping):
        raise RecipeError(f"{name}.metadata must be an object")
    try:
        _contract().validate_action(values, metadata, scope=scope)
    except ValueError as exc:
        raise RecipeError(f"{name}: {exc}") from exc
    for key, item in values.items():
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise RecipeError(f"{name}.values.{key} must be finite numeric")
        if not math.isfinite(float(item)):
            raise RecipeError(f"{name}.values.{key} must be finite numeric")
    return {
        "timestamp_s": timestamp,
        "values": dict(values),
        "metadata": dict(metadata),
    }


def _action_from_values(values: Mapping[str, Any], stage: str, *, scope: str) -> dict[str, Any]:
    """Author a recipe-owned action with canonical action space and units."""

    if not isinstance(values, Mapping) or not values:
        raise RecipeError(f"{stage} action values must be a non-empty object")
    try:
        metadata = _contract().stamped_metadata(
            values, {"source": "xlerobot.snack_delivery", "stage": stage}, scope=scope
        )
    except ValueError as exc:
        raise RecipeError(f"{stage}: {exc}") from exc
    return _action(
        {"timestamp_s": 0.0, "values": dict(values), "metadata": metadata},
        stage,
        scope=scope,
    )


def _request_id(stage: str) -> str:
    return f"snack-{stage}-{uuid.uuid4().hex}"


def _actions_of(
    action: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(action, Mapping):
        return (action,)
    if isinstance(action, (str, bytes)) or not isinstance(action, Sequence):
        raise RecipeError("action must be an action object or a list of action objects")
    return tuple(action)


def _endpoint_spec(config: Mapping[str, Any], role: str) -> tuple[str, str, str]:
    control = config.get("control")
    if not isinstance(control, Mapping):
        raise RecipeError("control configuration is required")
    endpoints = control.get("endpoints")
    if not isinstance(endpoints, Mapping):
        raise RecipeError("control.endpoints must configure a base and a manipulation service")
    spec = endpoints.get(role)
    if not isinstance(spec, Mapping):
        raise RecipeError(f"control.endpoints.{role} is required")
    endpoint = _require_string(spec.get("endpoint"), f"control.endpoints.{role}.endpoint")
    runtime_id = _require_string(spec.get("runtime_id"), f"control.endpoints.{role}.runtime_id")
    scope = _require_string(spec.get("scope"), f"control.endpoints.{role}.scope")
    expected = "base" if role == "base" else "arms"
    if scope != expected:
        raise RecipeError(
            f"control.endpoints.{role}.scope must be {expected!r}, not {scope!r}; "
            "the recipe never widens a scope implicitly"
        )
    return endpoint, runtime_id, scope


class ControlEndpointRuntime:
    """Adapter for one scope-bound, versioned public Control HTTP client."""

    simulation = False

    def __init__(self, config: Mapping[str, Any], *, role: str) -> None:
        try:
            from embodirun.client import ControlClient
        except ImportError as exc:  # pragma: no cover - exercised by a bad install
            raise RecipeError("install this repository or set PYTHONPATH=src before hardware mode") from exc
        control = config.get("control")
        if not isinstance(control, Mapping):
            raise RecipeError("control configuration is required")
        endpoint, runtime_id, scope = _endpoint_spec(config, role)
        caller_id = _require_string(control.get("caller_id"), "control.caller_id")
        session_value = control.get("session_id", "generated-at-start")
        session_id = (
            uuid.uuid4().hex
            if session_value == "generated-at-start"
            else _require_string(session_value, "control.session_id")
        )
        timeout_s = _positive(control.get("timeout_s", 30), "control.timeout_s")
        self.role = role
        self.runtime_id = runtime_id
        self.scope = scope
        # Deliberately omit token=. Authentication belongs to the already
        # configured local Control/owner boundary, never to this recipe.
        self.client = ControlClient(
            endpoint,
            caller_id=caller_id,
            session_id=session_id,
            timeout_s=timeout_s,
        )

    def check(self) -> dict[str, Any]:
        description = self.client.describe()
        if description.get("runtime_id") != self.runtime_id:
            raise RecipeError(f"{self.role} endpoint has the wrong primary runtime")
        if description.get("control_scope") != self.scope:
            raise RecipeError(f"{self.role} endpoint does not declare scope={self.scope}")
        caps = description.get("application", {})
        required = ["observe", "execute", "stop"]
        if self.role == "manipulation":
            required += ["propose", "media"]
        if any(caps.get(key) is not True for key in required):
            raise RecipeError(f"{self.role} endpoint lacks {required}")
        return description

    def images(self, observation_id: str) -> list[dict[str, Any]]:
        response = self.client.media(observation_id, runtime_id=self.runtime_id, include_data=True)
        if response.get("observation_id") != observation_id:
            raise RecipeError("media does not match the reviewed observation")
        return [
            {"name": item["name"], "mime_type": item["mime_type"], "data": item["data_base64"]}
            for item in response.get("media", [])
        ]

    def observe(self) -> dict[str, Any]:
        observation = self.client.observe(runtime_id=self.runtime_id)
        return {
            "observation_id": observation.observation_id,
            "payload": dict(observation.payload),
        }

    def propose(
        self,
        *,
        request_id: str,
        observation_id: str,
        instruction: str,
        timeout_s: float,
    ) -> dict[str, Any]:
        return self.client.propose(
            request_id=request_id,
            observation_id=observation_id,
            instruction=instruction,
            runtime_id=self.runtime_id,
            timeout_s=timeout_s,
        )

    def execute(
        self,
        action: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        *,
        request_id: str,
        observation_id: str,
        control_hz: float,
    ) -> dict[str, Any]:
        self.require_scope(action)
        return self.client.execute(
            action,
            request_id=request_id,
            observation_id=observation_id,
            runtime_id=self.runtime_id,
            control_hz=control_hz,
            wait=True,
            source="agent",
        )

    def require_scope(self, action: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> None:
        """Refuse an action that does not belong to this endpoint's scope."""

        for index, item in enumerate(_actions_of(action)):
            values = item.get("values") if isinstance(item, Mapping) else None
            if not isinstance(values, Mapping) or not values:
                raise RecipeError(f"action[{index}] has no values object")
            found = _scope_of(values, f"action[{index}]")
            if found != self.scope:
                raise RecipeError(
                    f"refusing to send a {found} action to the {self.scope}-scope "
                    f"Control endpoint ({self.runtime_id}); configure a matching "
                    "scoped endpoint"
                )

    def stop(self, request_id: str) -> dict[str, Any]:
        return self.client.stop(request_id)


class ControlRuntimes:
    """Two scope-bound Control endpoints attached to one physical owner."""

    simulation = False

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.base = ControlEndpointRuntime(config, role="base")
        self.manipulation = ControlEndpointRuntime(config, role="manipulation")


class DryRunRuntime:
    """Deterministic orchestration fixture; it never represents hardware success."""

    simulation = True

    def __init__(self, config: Mapping[str, Any], *, role: str) -> None:
        endpoint, runtime_id, scope = _endpoint_spec(config, role)
        del endpoint  # a fixture never opens a connection
        dry_run = config.get("dry_run", {})
        if not isinstance(dry_run, Mapping):
            raise RecipeError("dry_run must be an object")
        self.role = role
        self.runtime_id = runtime_id
        self.scope = scope
        self.counter = 0
        self._grasp_values = dry_run.get("grasp_values", {"right_arm_gripper.pos": 25.0})
        self._grasp_evidence = bool(dry_run.get("grasp_evidence", True))
        self._arrival_evidence = bool(dry_run.get("arrival_evidence", True))

    def observe(self) -> dict[str, Any]:
        self.counter += 1
        payload: dict[str, Any] = {
            "status": "ok",
            "fresh": True,
            "fixture": True,
            "safety": {
                "base_control_ready": True,
                "stopped": True,
                "stop_confirmed": True,
            },
        }
        if self._arrival_evidence:
            payload["navigation"] = {"arrived": True, "zero_velocity": True}
        if self._grasp_evidence:
            payload["task_evidence"] = {"grasp_confirmed": True}
        payload["robot"] = {
            "metadata": {key: payload[key] for key in ("safety", "navigation", "task_evidence") if key in payload}
        }
        return {
            "observation_id": f"dry-run:{self.role}:{self.counter}",
            "payload": payload,
        }

    def propose(self, **kwargs: Any) -> dict[str, Any]:
        if not isinstance(self._grasp_values, Mapping) or not self._grasp_values:
            raise RecipeError("dry_run.grasp_values must be a non-empty object")
        return {
            "status": "proposed",
            "request_id": kwargs["request_id"],
            "observation_id": kwargs["observation_id"],
            "actions": [_action_from_values(self._grasp_values, "dry-run-grasp", scope=self.scope)],
            "fixture": True,
        }

    def execute(self, action: Any, **kwargs: Any) -> dict[str, Any]:
        self._require_scope(action)
        return {
            "status": "completed",
            "dispatch_status": "completed",
            "accepted": True,
            "success": None,
            "physical_status": "simulation_fixture_only",
            "request_id": kwargs["request_id"],
            "fixture": True,
        }

    def _require_scope(self, action: Any) -> None:
        for index, item in enumerate(_actions_of(action)):
            values = item.get("values") if isinstance(item, Mapping) else None
            if not isinstance(values, Mapping) or not values:
                raise RecipeError(f"action[{index}] has no values object")
            found = _scope_of(values, f"action[{index}]")
            if found != self.scope:
                raise RecipeError(f"dry-run {self.scope}-scope endpoint refused a {found}-scope action")

    def stop(self, request_id: str) -> dict[str, Any]:
        return {
            "status": "stopped",
            "physical_status": "simulation_fixture_only",
            "stop_confirmed": False,
            "request_id": request_id,
            "fixture": True,
        }


class DryRunRuntimes:
    """Fixture runtimes for the plan/dry-run modes."""

    simulation = True

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.base = DryRunRuntime(config, role="base")
        self.manipulation = DryRunRuntime(config, role="manipulation")


class SnackDelivery:
    """State machine for the complete chips pickup and handover task."""

    def __init__(
        self,
        config: Mapping[str, Any],
        runtimes: RuntimeSet,
        output: Path,
        *,
        auto_confirm: bool,
        evidence: EvidenceNormalizer | None = None,
    ) -> None:
        self.config = config
        self.runtimes = runtimes
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.auto_confirm = auto_confirm
        self.evidence: EvidenceNormalizer = evidence or ConfigEvidenceNormalizer.from_config(config)
        self.state: dict[str, Any] = {
            "recipe": "xlerobot.snack_delivery",
            "status": "prepared",
            "stage": "to_pickup",
            "simulation": runtimes.simulation,
            "history": [],
            "route_arrival": {},
        }
        self.last_request_id: str | None = None
        self.last_request_runtime: Runtime | None = None
        self.last_stop_receipt: dict[str, Any] | None = None
        self._last_payload: dict[str, Any] | None = None
        self._agent: Any = None

    @property
    def base_runtime(self) -> str:
        return self.runtimes.base.runtime_id

    @property
    def manipulation_runtime(self) -> str:
        return self.runtimes.manipulation.runtime_id

    def save_state(self) -> None:
        self.state["updated_ns"] = time.time_ns()
        _write_json(self.output / "status.json", self.state)

    def event(self, name: str, **data: Any) -> None:
        record = {"event": name, "stage": self.state["stage"], "timestamp_ns": time.time_ns(), **data}
        self.state["history"].append(record)
        with (self.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.save_state()

    def confirm(self, kind: str, message: str, **data: Any) -> None:
        detail = {"kind": kind, "message": message, **data}
        self.event("operator_wait", **detail)
        if self.auto_confirm:
            if not self.runtimes.simulation:
                raise RecipeError("hardware mode always requires a person; --auto-confirm is dry-run only")
            self.event("operator_confirmed", kind=kind, mode="dry-run")
            return
        print(json.dumps({"waiting": kind, **detail}, ensure_ascii=False), flush=True)
        answer = input("Type 'yes' after the physical check: ").strip().lower()
        if answer != "yes":
            raise RecipeError(f"operator did not confirm {kind}")
        self.event("operator_confirmed", kind=kind, mode="interactive")

    def observe(self, runtime: Runtime) -> dict[str, Any]:
        requested_ns = time.monotonic_ns()
        value = runtime.observe()
        if not isinstance(value, Mapping):
            raise RecipeError("runtime returned an invalid observation")
        observation_id = value.get("observation_id")
        payload = value.get("payload")
        if not isinstance(observation_id, str) or not observation_id:
            raise RecipeError("observation has no observation_id")
        if not isinstance(payload, Mapping):
            raise RecipeError("observation has no payload object")
        # A live observation must report freshness explicitly.  A missing or
        # non-boolean field fails closed; the recipe never assumes it is fresh.
        if not runtime.simulation and payload.get("fixture") is True:
            raise RecipeError("hardware endpoint returned fixture observations")
        if payload.get("fresh") is not True:
            raise RecipeError(
                f"fresh observation is required before this recipe can act (runtime={runtime.runtime_id!r})"
            )
        self._last_payload = dict(payload)
        age_ns = payload.get("age_ns")
        skew_ns = payload.get("timestamps", {}).get("skew_ns")
        # Remote monotonic timestamps have a different epoch. Translate the
        # reported elapsed age using our request start, conservatively including
        # the entire round trip and source skew, so every source is newer.
        capture_lower_bound_ns = (
            requested_ns - age_ns - skew_ns
            if all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (age_ns, skew_ns))
            and age_ns + skew_ns <= requested_ns
            else None
        )
        return {
            "observation_id": observation_id,
            "payload": dict(payload),
            "capture_lower_bound_ns": capture_lower_bound_ns,
        }

    def post_observe(self, runtime: Runtime, previous_id: str, *, after_ns: int | None = None) -> dict[str, Any]:
        deadline = time.monotonic() + 2.0
        while True:
            observed = self.observe(runtime)
            state_time = observed.get("capture_lower_bound_ns")
            after_execution = after_ns is None or (
                isinstance(state_time, int) and not isinstance(state_time, bool) and state_time >= after_ns
            )
            if observed["observation_id"] != previous_id and after_execution:
                return observed
            if time.monotonic() >= deadline:
                raise RecipeError("no new post-execution observation")
            time.sleep(0.02)

    def evaluate_evidence(self, payload: Mapping[str, Any], name: str, *, label: str) -> EvidenceValue:
        value = self.evidence.evaluate(name, payload)
        if value.confirmed:
            return value
        if self.evidence.manual_confirmation_allowed(name):
            self.confirm(
                f"manual_{name}",
                f"Owner/sensor evidence for {label} ({name}) is unavailable. Confirm the physical state manually.",
                evidence=value.to_dict(),
            )
            manual = EvidenceValue(
                name=name,
                confirmed=True,
                source="manual_confirmation",
                path=value.path,
                value=value.value,
                present=value.present,
            )
            self.event(
                "evidence_manual_confirmation",
                evidence=manual.to_dict(),
                label=label,
            )
            return manual
        raise RecipeError(
            f"{label} is not confirmed by the owner observation "
            f"({name} at {'.'.join(value.path)}); the recipe stops here"
        )

    def require_evidence(self, payload: Mapping[str, Any], name: str, *, label: str) -> EvidenceValue:
        value = self.evaluate_evidence(payload, name, label=label)
        self.event("evidence_ok", evidence=value.to_dict(), label=label)
        return value

    def execute(
        self,
        runtime: Runtime,
        action: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        observation_id: str,
        *,
        stage: str,
        hz: float,
    ) -> dict[str, Any]:
        request_id = _request_id(stage)
        self.last_request_id = request_id
        self.last_request_runtime = runtime
        result = runtime.execute(
            action,
            request_id=request_id,
            observation_id=observation_id,
            control_hz=hz,
        )
        if not isinstance(result, Mapping):
            raise RecipeError(f"{stage} returned an invalid execution receipt")
        if result.get("accepted") is False or result.get("success") is False:
            raise RecipeError(f"{stage} was rejected: {result}")
        status = result.get("status")
        dispatch = result.get("dispatch_status")
        if status != "completed" or dispatch != "completed":
            raise RecipeError(
                f"{stage} is not a completed dispatch (status={status!r}, "
                f"dispatch_status={dispatch!r}); the recipe does not advance"
            )
        physical = result.get("physical_status")
        if physical == "stop_requested":
            # The public runner queues scoped cleanup. Completion of dispatch
            # is not completion of that cleanup: wait for later sensor evidence.
            finished_ns = time.monotonic_ns()
            deadline = time.monotonic() + 2.0
            while True:
                post = self.post_observe(runtime, observation_id, after_ns=finished_ns)
                names = ("stopped", "stop_confirmed")
                if runtime.scope == "arms":
                    names += ("arms_stop_confirmed",)
                checks = [self.evidence.evaluate(name, post["payload"]) for name in names]
                if all(item.confirmed for item in checks):
                    self.event(
                        "stop_observed",
                        observation_id=post["observation_id"],
                        evidence=[item.to_dict() for item in checks],
                    )
                    break
                if time.monotonic() >= deadline:
                    raise RecipeError("scoped cleanup has no confirmed stop feedback")
                time.sleep(0.02)
        if physical in BLOCKED_PHYSICAL_STATUSES:
            raise RecipeError(
                f"{stage} reported physical_status={physical!r}; the recipe does not advance without a confirmed hold"
            )
        if physical == "simulation_fixture_only":
            if not runtime.simulation:
                raise RecipeError(f"{stage} reported simulation_fixture_only on a hardware runtime")
        elif physical not in TERMINAL_PHYSICAL_STATUSES and physical != "stop_requested":
            raise RecipeError(f"{stage} did not report a terminal physical status: {physical!r}")
        self.event(
            "execution_receipt",
            stage=stage,
            request_id=request_id,
            runtime_id=runtime.runtime_id,
            scope=runtime.scope,
            receipt=dict(result),
        )
        return dict(result)

    # --- route artifacts -------------------------------------------------

    def _route_path(self, route_ref: str) -> Path:
        route_path = Path(route_ref)
        if route_path.is_absolute():
            return route_path
        candidates = (
            Path(self.config.get("_config_dir", ".")) / route_path,
            Path.cwd() / route_path,
            Path(__file__).resolve().parent / route_path,
        )
        return next((candidate for candidate in candidates if candidate.exists()), candidates[0])

    def load_route(self, route_name: str) -> tuple[list[Any], dict[str, Any]]:
        """Load one route artifact and return its chunks plus provenance."""

        navigation = self.config.get("navigation")
        if not isinstance(navigation, Mapping):
            raise RecipeError("navigation configuration is required")
        routes = navigation.get("routes")
        if not isinstance(routes, Mapping):
            raise RecipeError("navigation.routes is required")
        route_ref = routes.get(route_name)
        if isinstance(route_ref, str):
            route: Any = _load_json(self._route_path(route_ref))
        elif isinstance(route_ref, (list, Mapping)):
            route = route_ref
        else:
            raise RecipeError(f"navigation.routes.{route_name} is missing")
        if not isinstance(route, Mapping):
            raise RecipeError(f"route {route_name} must be an object")

        provenance = {
            "schema": route.get("schema"),
            "source": route.get("source"),
            "fixture": route.get("fixture"),
            "verified": route.get("verified"),
            "hardware_access": route.get("hardware_access"),
            "provider": route.get("provider"),
        }
        if isinstance(route.get("route"), Mapping):
            inner = route.get("route")
            if not isinstance(inner, Mapping):
                raise RecipeError(f"route artifact {route_name} must contain a route object")
            route = inner
        chunks = route.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise RecipeError(f"route {route_name} has no action chunks; record this room first")
        return chunks, provenance

    def _check_route_provenance(self, route_name: str, provenance: Mapping[str, Any]) -> None:
        planning = self.config.get("planning", {})
        mode = planning.get("mode", DEFAULT_PLANNING_MODE) if isinstance(planning, Mapping) else DEFAULT_PLANNING_MODE
        source = provenance.get("source")
        if source is not None:
            if source not in {"recorded_route", "diagram"}:
                raise RecipeError(
                    f"route {route_name} declares source={source!r}; 'source' is now "
                    "the provenance enum (recorded_route or diagram) produced by the "
                    "RPent route adapter. Move free-form prose to 'description'."
                )
            if source != mode:
                raise RecipeError(
                    f"route {route_name} was produced by {source!r} but planning.mode is "
                    f"{mode!r}; regenerate it with the configured mode"
                )
        if self.runtimes.simulation:
            return
        if provenance.get("fixture") is True:
            raise RecipeError(
                f"route {route_name} is the checked-in fixture; replace it with a room route before hardware mode"
            )

    def _arrival_requirements(self) -> tuple[str, ...]:
        navigation = self.config.get("navigation")
        if not isinstance(navigation, Mapping):
            raise RecipeError("navigation configuration is required")
        raw = navigation.get("arrival_evidence", ["route_arrived", "stopped", "stop_confirmed"])
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise RecipeError("navigation.arrival_evidence must be a list of evidence names")
        names = list(dict.fromkeys([*raw, "route_arrived", "stopped", "stop_confirmed", "zero_velocity"]))
        if navigation.get("require_zero_velocity", True) and "zero_velocity" not in names:
            names.append("zero_velocity")
        return tuple(names)

    def route(self, route_name: str, stage: str) -> None:
        chunks, provenance = self.load_route(route_name)
        self._check_route_provenance(route_name, provenance)
        navigation = self.config.get("navigation")
        assert isinstance(navigation, Mapping)
        hz = _positive(navigation.get("control_hz", 15), "navigation.control_hz")
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, Mapping):
                raise RecipeError(f"route {route_name} chunk {index} must be an object")
            observed = self.observe(self.runtimes.base)
            self.require_evidence(observed["payload"], "base_control_ready", label="base control readiness")
            actions = chunk.get("actions", chunk.get("action"))
            if isinstance(actions, Mapping):
                command: Mapping[str, Any] | Sequence[Mapping[str, Any]] = _action(
                    actions, f"{route_name}[{index}].action", scope="base"
                )
            elif isinstance(actions, list) and actions:
                command = [
                    _action(item, f"{route_name}[{index}].actions[{j}]", scope="base") for j, item in enumerate(actions)
                ]
            else:
                raise RecipeError(f"{route_name}[{index}] must provide action or actions")
            self.execute(
                self.runtimes.base,
                command,
                observed["observation_id"],
                stage=f"{stage}-{index:03d}",
                hz=hz,
            )
            self.post_observe(self.runtimes.base, observed["observation_id"])

        # A completed HTTP receipt only proves that the chunks were dispatched.
        # Arrival and zero-velocity evidence is a separate, configurable check.
        arrival = self.observe(self.runtimes.base)
        results = {
            name: (
                self.require_evidence(arrival["payload"], name, label=f"{route_name} arrival").to_dict()
                if self.evidence.manual_confirmation_allowed(name)
                else self.evidence.evaluate(name, arrival["payload"]).to_dict()
            )
            for name in self._arrival_requirements()
        }
        verified = bool(results) and all(item["confirmed"] for item in results.values())
        self.state["route_arrival"][route_name] = {
            "verified": verified,
            "observation_id": arrival["observation_id"],
            "evidence": results,
        }
        self.event(
            "route_complete",
            route=route_name,
            chunks=len(chunks),
            arrival="verified" if verified else "unverified",
            observation_id=arrival["observation_id"],
            evidence=results,
            route_source=provenance.get("source"),
            fixture=provenance.get("fixture"),
        )

    def _require_route_verified(self, route_name: str, *, target: str) -> None:
        entry = self.state["route_arrival"].get(route_name)
        if not isinstance(entry, Mapping) or entry.get("verified") is not True:
            raise RecipeError(f"route {route_name} arrival is unverified; refusing to enter {target}")

    def ensure_hardware_inputs(self) -> None:
        """Reject unverified fixtures before asking an operator to move hardware."""

        for route_name in ("outbound", "return"):
            chunks, provenance = self.load_route(route_name)
            self._check_route_provenance(route_name, provenance)
            for index, chunk in enumerate(chunks):
                if not isinstance(chunk, Mapping):
                    raise RecipeError(f"route {route_name} chunk {index} must be an object")
                actions = _actions_of(chunk.get("actions", chunk.get("action")))
                if not actions or len(actions) > 100:
                    raise RecipeError("each route chunk must contain 1..100 actions")
                for item in actions:
                    _action(item, f"{route_name}[{index}]", scope="base")

    def agent(self) -> Any:
        if self._agent is None:
            spec = self.config.get("agent", {})
            factory = _require_string(spec.get("factory"), "agent.factory")
            module, separator, name = factory.partition(":")
            if not separator:
                raise RecipeError("agent.factory must be module:callable")
            self._agent = getattr(importlib.import_module(module), name)(spec.get("options", {}))
        return self._agent

    def prepare_routes(self) -> None:
        planning = self.config.get("planning", {})
        if planning.get("mode") != "diagram":
            return
        path = self._route_path(planning["route_diagram"])
        segments = planning.get("segments", {})
        if not isinstance(segments, Mapping) or not segments:
            raise RecipeError("diagram mode requires annotated, recorded planning.segments")
        packet = {
            "segments": {name: item["description"] for name, item in segments.items()},
            "images": [
                {
                    "mime_type": "image/png" if path.suffix.lower() == ".png" else "image/jpeg",
                    "data": base64.b64encode(path.read_bytes()).decode(),
                }
            ],
        }
        decision = self.agent().plan_routes(packet)
        if not isinstance(decision, Mapping) or decision.get("decision") != "proceed":
            raise RecipeError("RPent did not approve route selection")
        for route_name in ("outbound", "return"):
            selected = decision.get(route_name)
            if not isinstance(selected, list) or not 1 <= len(selected) <= 32:
                raise RecipeError("RPent must select 1..32 recorded segments per route")
            if any(not isinstance(name, str) or name not in segments for name in selected):
                raise RecipeError("RPent selected an unknown route segment")
        self._write_artifact("rpent-route-decision.json", decision)
        for route_name in ("outbound", "return"):
            chunks = []
            for name in decision[route_name]:
                if name not in segments:
                    raise RecipeError("RPent selected an unknown route segment")
                segment = _load_json(self._route_path(segments[name]["path"]))
                if segment.get("fixture") is not False:
                    raise RecipeError("diagram route segments must explicitly be non-fixture recordings")
                chunks.extend(segment["chunks"])
            self.config["navigation"]["routes"][route_name] = {
                "source": "diagram",
                "fixture": False,
                "chunks": chunks,
            }

    def grasp(self) -> None:
        self._require_route_verified("outbound", target="grasp")
        grasp = self.config["grasp"]
        self.confirm("ready_grasp", "At the pickup point: chips are in the grasp area and the base is stopped.")
        max_steps = grasp.get("max_steps", 8)
        max_rounds = grasp.get("max_rounds", 1)
        for round_index in range(max_rounds):
            observed = self.observe(self.runtimes.manipulation)
            instruction = _require_string(grasp.get("instruction"), "grasp.instruction")
            steps = max_steps
            if not self.runtimes.simulation:
                packet = {
                    "observation_id": observed["observation_id"],
                    "observation": observed["payload"],
                    "instruction": instruction,
                    "max_steps": max_steps,
                    "images": self.runtimes.manipulation.images(observed["observation_id"]),
                }
                decision = self.agent().before_grasp(packet)
                self._write_artifact(f"rpent-grasp-{round_index:03d}.json", decision)
                if decision.get("observation_id") != observed["observation_id"]:
                    raise RecipeError("RPent decision has the wrong observation_id")
                if decision.get("decision") != "proceed":
                    raise RecipeError(f"RPent held grasp: {decision.get('reason')}")
                steps = decision.get("max_steps")
                if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= max_steps:
                    raise RecipeError("RPent exceeded the action prefix bound")
                instruction = _require_string(decision.get("instruction"), "RPent instruction")
                # RPent selects a skill request. It cannot authorize a VLA
                # proposal based on the scene it reviewed seconds earlier.
                observed = self.post_observe(
                    self.runtimes.manipulation, observed["observation_id"], after_ns=time.monotonic_ns()
                )
            for name in ("stopped", "stop_confirmed"):
                self.require_evidence(observed["payload"], name, label="base stopped before grasp")
            proposal = self.runtimes.manipulation.propose(
                request_id=_request_id("proposal"),
                observation_id=observed["observation_id"],
                instruction=instruction,
                timeout_s=_positive(grasp.get("timeout_s", 120), "grasp.timeout_s"),
            )
            if proposal.get("observation_id", observed["observation_id"]) != observed["observation_id"]:
                raise RecipeError("VLA proposal references a different observation")
            actions = proposal.get("actions")
            if not isinstance(actions, list) or not actions:
                raise RecipeError("VLA returned no executable actions")
            commands = [
                _action(item, f"VLA actions[{index}]", scope="arms") for index, item in enumerate(actions[:steps])
            ]
            self._write_artifact(f"grasp-proposal-{round_index:03d}.json", proposal)
            # Check the same snapshot again after inference. Never relabel an
            # old proposal with a newer observation to bypass freshness.
            if not self.runtimes.simulation:
                pinned = self.runtimes.manipulation.client.observe(
                    observation_id=observed["observation_id"], runtime_id=self.manipulation_runtime
                )
                if pinned.fresh is not True:
                    raise RecipeError("VLA proposal observation expired during inference; shorten model latency")
            self.execute(
                self.runtimes.manipulation,
                commands,
                observed["observation_id"],
                stage="grasp-vla",
                hz=_positive(grasp.get("control_hz", 15), "grasp.control_hz"),
            )
            after = self.post_observe(self.runtimes.manipulation, observed["observation_id"])
            for name in ("stopped", "stop_confirmed"):
                self.require_evidence(after["payload"], name, label="stop after grasp")
            evidence = self.evidence.evaluate("grasp_confirmed", after["payload"])
            if evidence.confirmed or self.evidence.manual_confirmation_allowed("grasp_confirmed"):
                evidence = self.require_evidence(after["payload"], "grasp_confirmed", label="grasp")
                self.event("grasp_confirmed", evidence=evidence.to_dict(), observation_id=after["observation_id"])
                self.confirm(
                    "ready_return", "Confirm the chips are held, the arm is clear, and the route back is safe."
                )
                return
        raise RecipeError("grasp is not confirmed after the bounded VLA rounds")

    def handover(self) -> None:
        self._require_route_verified("return", target="handover")
        handover = self.config.get("handover")
        if not isinstance(handover, Mapping):
            raise RecipeError("handover configuration is required")
        pose = handover.get("forward_pose")
        opening = handover.get("gripper_opening")
        if not isinstance(pose, Mapping) or not pose:
            raise RecipeError("handover.forward_pose must be a calibrated non-empty pose")
        if not isinstance(opening, Mapping) or not opening:
            raise RecipeError("handover.gripper_opening must be a calibrated action")
        if "right_arm_gripper.pos" not in opening:
            raise RecipeError("handover.gripper_opening must include right_arm_gripper.pos")
        if any(not str(name).startswith("right_arm_") for name in (*pose.keys(), *opening.keys())):
            raise RecipeError("handover may address right-arm public action fields only")
        self.confirm("receiver_ready", "At the handover point: the recipient is in position and can support the chips.")
        observed = self.observe(self.runtimes.manipulation)
        self.require_evidence(observed["payload"], "stopped", label="base stopped before handover")
        self.require_evidence(observed["payload"], "stop_confirmed", label="stop before handover")
        hz = _positive(handover.get("control_hz", 15), "handover.control_hz")
        self.execute(
            self.runtimes.manipulation,
            _action_from_values(pose, "handover-extension", scope="arms"),
            observed["observation_id"],
            stage="handover-extension",
            hz=hz,
        )
        after_extension = self.post_observe(self.runtimes.manipulation, observed["observation_id"])
        self.require_evidence(after_extension["payload"], "stopped", label="base stopped before opening")
        # Hold the calibrated extension while loosening the gripper. This
        # mirrors the owner-side move_fields behavior and avoids sending a
        # gripper-only action that could leave the other joints unspecified.
        opening_action = dict(pose)
        opening_action.update(opening)
        self.execute(
            self.runtimes.manipulation,
            _action_from_values(opening_action, "handover-opening", scope="arms"),
            after_extension["observation_id"],
            stage="handover-opening",
            hz=hz,
        )
        after_opening = self.post_observe(self.runtimes.manipulation, after_extension["observation_id"])
        self.require_evidence(after_opening["payload"], "stopped", label="stop after handover")
        self.require_evidence(after_opening["payload"], "stop_confirmed", label="stop after handover")
        self.state["last_handover_payload"] = dict(after_opening["payload"])
        self.event("handover_motion_complete")
        self.confirm("delivery_received", "Confirm that the person has taken the chips before finishing.")

    def _write_artifact(self, name: str, value: Any) -> None:
        if isinstance(value, Mapping):
            _write_json(self.output / name, value)
        else:
            _write_json(self.output / name, {"value": value})

    def completion_evidence(self) -> dict[str, Any]:
        """Report task/physical success only from configured evidence.

        A hardware run never writes ``physical_success=true`` because it got to
        the end of the state machine.  Without a configured, owner-published
        success field the result stays ``unverified``/``null``.
        """

        raw = self.config.get("evidence", {})
        completion = raw.get("completion", {}) if isinstance(raw, Mapping) else {}
        if not isinstance(completion, Mapping):
            raise RecipeError("evidence.completion must be an object")
        payload = self.state.get("last_handover_payload")
        result: dict[str, Any] = {
            "task_success": "unverified",
            "physical_success": None,
            "evidence": {},
        }
        for name in ("task_success", "physical_success"):
            path = completion.get(name)
            if not path:
                continue
            if isinstance(path, (str, bytes)) or not isinstance(path, Sequence):
                raise RecipeError(f"evidence.completion.{name} must be a field path")
            present, value = lookup_path(payload if isinstance(payload, Mapping) else {}, path)
            result["evidence"][name] = {
                "path": list(path),
                "present": present,
                "value": value,
                "confirmed": present and value is True,
            }
            if present and value is True:
                result[name] = True
        if self.runtimes.simulation:
            # Fixture evidence is never a physical-success claim.
            result["task_success"] = "unverified"
            result["physical_success"] = None
        return result

    def request_stop_after_failure(self) -> dict[str, Any] | None:
        """Stop the exact request that is still in flight, on its own runtime.

        The request may belong to the base endpoint or the manipulation
        endpoint; stopping a different scope would leave the moving owner
        untouched.  The stop receipt is recorded, never raised, so the original
        failure survives, and an unconfirmed stop stays explicit.
        """

        if self.last_request_id is None:
            return None
        runtime = self.last_request_runtime or self.runtimes.manipulation
        try:
            receipt = runtime.stop(self.last_request_id)
        except Exception as exc:  # preserve the original failure; require inspection
            self.last_stop_receipt = {
                "stop_confirmed": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
            self.event(
                "stop_request_failed",
                request_id=self.last_request_id,
                runtime_id=runtime.runtime_id,
                scope=runtime.scope,
                error=self.last_stop_receipt["error"],
                stop_unconfirmed=True,
            )
            return self.last_stop_receipt
        value = dict(receipt) if isinstance(receipt, Mapping) else {"receipt": receipt}
        self.last_stop_receipt = value
        confirmed = value.get("stop_confirmed") is True
        self.event(
            "stop_requested_after_failure",
            request_id=self.last_request_id,
            runtime_id=runtime.runtime_id,
            scope=runtime.scope,
            receipt=value,
            stop_confirmed=confirmed,
            stop_unconfirmed=not confirmed,
        )
        return value

    def _abort(self, status: str, error: BaseException) -> None:
        receipt = self.request_stop_after_failure()
        stop_confirmed = receipt.get("stop_confirmed") if isinstance(receipt, Mapping) else None
        self.state.update(
            {
                "status": status,
                "error": f"{type(error).__name__}: {error}",
                "task_success": "unverified",
                "physical_success": None,
                "stop_requested": receipt is not None,
                "stop_confirmed": stop_confirmed,
                "stop_unconfirmed": receipt is None or stop_confirmed is not True,
            }
        )
        self.event(
            status,
            error=self.state["error"],
            request_id=self.last_request_id,
            stop_confirmed=stop_confirmed,
            stop_unconfirmed=self.state["stop_unconfirmed"],
        )

    def plan(self) -> dict[str, Any]:
        self.ensure_hardware_inputs()
        navigation = self.config.get("navigation", {})
        routes = navigation.get("routes", {}) if isinstance(navigation, Mapping) else {}
        planning = self.config.get("planning", {})
        summary = {
            "recipe": "xlerobot.snack_delivery",
            "sequence": [
                "handover point → pickup point",
                "VLA grasp",
                "pickup point → handover point",
                "right-arm handover",
            ],
            "base_runtime": self.base_runtime,
            "manipulation_runtime": self.manipulation_runtime,
            "control_endpoints": {
                "base": {"runtime_id": self.base_runtime, "scope": self.runtimes.base.scope},
                "manipulation": {
                    "runtime_id": self.manipulation_runtime,
                    "scope": self.runtimes.manipulation.scope,
                },
            },
            "routes": list(routes.keys()) if isinstance(routes, Mapping) else [],
            "route_source": planning.get("mode", DEFAULT_PLANNING_MODE)
            if isinstance(planning, Mapping)
            else DEFAULT_PLANNING_MODE,
            "route_provider": "RPent" if planning.get("mode") == "diagram" else "recording",
            "arrival_evidence": list(self._arrival_requirements()),
            "model_instruction": self.config.get("grasp", {}).get("instruction"),
            "validation": "local config and action files only",
            "unchecked": ["Control endpoints", "RPent credentials", "model availability", "physical calibration"],
            "credentials": "deployment boundary only",
            "task_success": "unverified",
            "physical_success": None,
        }
        self.output.mkdir(parents=True, exist_ok=True)
        self._write_artifact("plan.json", summary)
        return summary

    def run(self) -> dict[str, Any]:
        self.output.mkdir(parents=True, exist_ok=True)
        self.save_state()
        try:
            if not self.runtimes.simulation:
                for runtime in (self.runtimes.base, self.runtimes.manipulation):
                    self._write_artifact(f"{runtime.scope}-description.json", runtime.check())
                agent = self.agent()
                if not callable(getattr(agent, "before_grasp", None)):
                    raise RecipeError("RPent factory must provide before_grasp")
                check = getattr(agent, "check", None)
                if callable(check):
                    check()
                self.prepare_routes()
            self.ensure_hardware_inputs()
            self.confirm(
                "ready_start", "Start at the handover point with the robot owner connected and all people clear."
            )
            self.route("outbound", "to-pickup")
            self.state["stage"] = "grasp"
            self.save_state()
            self.grasp()
            self.state["stage"] = "to_handover"
            self.save_state()
            self.route("return", "to-handover")
            self.state["stage"] = "handover"
            self.save_state()
            self.handover()
            completion = self.completion_evidence()
            self.state.update(
                {
                    "stage": "done",
                    "status": "completed",
                    "task_success": completion["task_success"],
                    "physical_success": completion["physical_success"],
                    "completion_evidence": completion["evidence"],
                }
            )
            self.event(
                "completed",
                task_success=completion["task_success"],
                physical_success=completion["physical_success"],
                completion_evidence=completion["evidence"],
            )
        except KeyboardInterrupt as exc:
            # An operator interrupt must stop the same in-flight request and
            # keep the unconfirmed-stop fact on the record before re-raising.
            self._abort("interrupted", exc)
            raise
        except Exception as exc:
            self._abort("failed", exc)
            raise
        return self.state


def _validate_config(config: Mapping[str, Any]) -> None:
    _reject_credentials(config)
    control = config.get("control")
    if not isinstance(control, Mapping):
        raise RecipeError("control configuration is required")
    if "endpoint" in control:
        raise RecipeError("use control.endpoints.base and control.endpoints.manipulation")
    _require_string(control.get("caller_id"), "control.caller_id")
    for role in ("base", "manipulation"):
        endpoint, _, _ = _endpoint_spec(config, role)
        parsed = _urlsplit(endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise RecipeError(
                f"control.endpoints.{role}.endpoint must be an HTTP(S) URL without "
                "credentials, query, or fragment; tokens belong to the deployment boundary"
            )
    planning = config.get("planning", {})
    if not isinstance(planning, Mapping):
        raise RecipeError("planning must be an object")
    planning_mode = planning.get("mode", DEFAULT_PLANNING_MODE)
    if planning_mode not in {"recorded_route", "diagram"}:
        raise RecipeError("planning.mode must be recorded_route or diagram")
    if planning_mode == "diagram":
        _require_string(planning.get("route_diagram"), "planning.route_diagram")
    navigation = config.get("navigation")
    if not isinstance(navigation, Mapping):
        raise RecipeError("navigation configuration is required")
    _positive(navigation.get("control_hz", 15), "navigation.control_hz")
    ConfigEvidenceNormalizer.from_config(config)
    grasp = config.get("grasp", {})
    for key, limit, default in (("max_steps", 15, 8), ("max_rounds", 100, 1)):
        value = grasp.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= limit:
            raise RecipeError(f"grasp.{key} must be in 1..{limit}")
    handover = config.get("handover")
    if not isinstance(handover, Mapping):
        raise RecipeError("handover configuration is required")
    for key in ("forward_pose", "gripper_opening"):
        value = handover.get(key)
        if not isinstance(value, Mapping) or not value:
            raise RecipeError(f"handover.{key} must be a calibrated non-empty object")
        for field, target in value.items():
            if not str(field).startswith("right_arm_"):
                raise RecipeError(f"handover.{key} contains a non-right-arm field: {field}")
            _finite(target, f"handover.{key}.{field}")
    if "right_arm_gripper.pos" not in handover["gripper_opening"]:
        raise RecipeError("handover.gripper_opening must include right_arm_gripper.pos")


def _urlsplit(endpoint: str) -> Any:
    from urllib.parse import urlsplit

    return urlsplit(endpoint)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("plan", "dry-run", "hardware"), default="plan")
    parser.add_argument(
        "--auto-confirm",
        action="store_true",
        help="auto-confirm only dry-run operator gates; hardware always requires a person",
    )
    args = parser.parse_args(argv)
    try:
        config = _load_json(args.config)
        _validate_config(config)
        config = dict(config)
        config["_config_dir"] = str(args.config.parent.resolve())
        runtimes: RuntimeSet = DryRunRuntimes(config) if args.mode in {"plan", "dry-run"} else ControlRuntimes(config)
        runner = SnackDelivery(
            config,
            runtimes,
            args.output,
            auto_confirm=bool(args.auto_confirm) and args.mode == "dry-run",
        )
        result = runner.plan() if args.mode == "plan" else runner.run()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (RecipeError, EvidenceError, OSError, ValueError, EOFError) as exc:
        print(f"recipe failed: {exc}", file=sys.stderr)
        return 2


def interrupted(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
