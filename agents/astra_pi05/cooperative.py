"""A small Astra–π0.5 cooperative loop over the public HTTP client.

This module demonstrates the Astra–π0.5 ordering at the public boundary:
observe one shared snapshot, validate one 50x12 proposal, review it, execute a
bounded prefix or an explicitly mapped correction, then observe again.  An
external caller such as RPent supplies the proposal and reviewer; this adapter
is not a model-serving API or a second scheduler.
"""

from __future__ import annotations

import inspect
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from agents.astra_pi05.decision import validate_decision, validate_proposal
from embodirun.client import (
    ControlClient,
    ControlHTTPError,
    ControlTransportError,
    Observation,
    UnsupportedOperation,
)

ReviewPacketBuilder = Callable[[Observation, "Proposal", Mapping[str, Any]], Mapping[str, Any]]
RequestStarted = Callable[[str], None]
ActionEncoder = Callable[..., Mapping[str, Any]]


class CorrectionUnsupported(RuntimeError):
    """No robot-specific waypoint-to-action mapper was supplied."""


@dataclass(frozen=True, slots=True)
class Proposal:
    """A detached, validated 50-step proposal from an external model.

    ``observation_id`` is copied from the snapshot used to produce the
    proposal. It is checked on construction so a callback cannot silently
    relabel a proposal for a newer observation.
    """

    proposal_id: str
    observation_id: str
    actions: tuple[tuple[float, ...], ...]
    metadata: Mapping[str, Any]

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        observation_id: str,
        proposal_id: str | None = None,
    ) -> Proposal:
        """Validate a proposal while preserving the caller's snapshot ID."""

        if not isinstance(payload, Mapping):
            raise ValueError("proposal must be an object")
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("proposal.metadata must be an object")
        actions = validate_proposal(payload.get("actions"), metadata)
        declared_observation_id = payload.get("observation_id")
        if declared_observation_id is not None and declared_observation_id != observation_id:
            raise ValueError("proposal.observation_id does not match the observation")
        value = proposal_id if proposal_id is not None else payload.get("proposal_id")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("proposal_id must be a non-empty string")
        return cls(value, observation_id, actions, dict(metadata))

    def to_payload(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "observation_id": self.observation_id,
            "actions": [list(row) for row in self.actions],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CooperativeResult:
    """Facts from one software cooperative round.

    ``status`` describes the public control outcome. It does not assert a
    business task succeeded; that remains an external caller's decision based
    on the new observation and robot feedback. ``pending``, ``unknown``,
    ``cancelled`` and ``manual_takeover`` retain their control meaning and
    keep ``request_id`` for later inspection.
    """

    status: str
    observation_id: str
    proposal_id: str
    decision: str | None = None
    executed_steps: int = 0
    discarded_steps: int = 0
    execution: Mapping[str, Any] | None = None
    post_observation: Observation | None = None
    error: str | None = None
    request_id: str | None = None
    execution_evidence_unknown: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "observation_id": self.observation_id,
            "proposal_id": self.proposal_id,
            "decision": self.decision,
            "executed_steps": self.executed_steps,
            "discarded_steps": self.discarded_steps,
            "execution": dict(self.execution) if self.execution is not None else None,
            "post_observation_id": (
                self.post_observation.observation_id if self.post_observation is not None else None
            ),
            "error": self.error,
            "request_id": self.request_id,
            "execution_evidence_unknown": self.execution_evidence_unknown,
        }


def rows_to_actions(
    rows: Sequence[Sequence[float]],
    *,
    feature_names: Sequence[str],
    timestamp_step_s: float = 0.1,
    metadata: Mapping[str, Any] | None = None,
    action_encoder: ActionEncoder | None = None,
) -> list[dict[str, Any]]:
    """Map rows only when the proposal declares explicit binding feature names.

    The mapping is positional and unit-preserving.  It is suitable for a
    binding that publicly declares the twelve BiSO101 joint names; it is not a
    generic 6D IK conversion and refuses missing or duplicate names.
    """

    names = tuple(feature_names)
    if len(names) != 12 or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("feature_names must contain 12 non-empty names")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique")
    if isinstance(timestamp_step_s, bool) or not math.isfinite(timestamp_step_s) or timestamp_step_s <= 0:
        raise ValueError("timestamp_step_s must be finite and positive")
    result = []
    shared_metadata = dict(metadata or {})
    for index, row in enumerate(rows):
        if len(row) != len(names) or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in row
        ):
            raise ValueError("action rows must contain finite values matching feature_names")
        timestamp_s = index * float(timestamp_step_s)
        if action_encoder is None:
            result.append(
                {
                    "timestamp_s": timestamp_s,
                    "values": {name: float(value) for name, value in zip(names, row)},
                    "metadata": shared_metadata,
                }
            )
            continue
        encoded = action_encoder(
            tuple(float(value) for value in row),
            timestamp_s=timestamp_s,
            metadata=shared_metadata,
        )
        if not isinstance(encoded, Mapping):
            raise ValueError("action_encoder must return a public action mapping")
        action = dict(encoded)
        declared_timestamp = action.get("timestamp_s", timestamp_s)
        if (
            isinstance(declared_timestamp, bool)
            or not isinstance(declared_timestamp, (int, float))
            or not math.isfinite(float(declared_timestamp))
            or not math.isclose(float(declared_timestamp), timestamp_s, abs_tol=1e-9)
        ):
            raise ValueError("action_encoder timestamp_s must match the public schedule")
        values = action.get("values")
        if not isinstance(values, Mapping):
            raise ValueError("action_encoder must return public values")
        encoded_metadata = action.get("metadata", {})
        if not isinstance(encoded_metadata, Mapping):
            raise ValueError("action_encoder metadata must be a mapping")
        action["timestamp_s"] = timestamp_s
        merged_metadata = dict(shared_metadata)
        merged_metadata.update(dict(encoded_metadata))
        action["metadata"] = merged_metadata
        result.append(action)
    return result


class CooperativeLoop:
    """Execute one externally planned observe/review/short-segment round.

    ``control_hz`` is the action stream rate in hertz and is passed to the
    public execute request. After a completed execution,
    ``post_observation_timeout_s`` is a bounded timeout in seconds: the loop
    polls for an observation with a new ID and reports
    ``post_observation_pending`` if the server still returns the old snapshot.
    It never treats that old snapshot as post-action evidence.
    """

    def __init__(
        self,
        client: ControlClient,
        *,
        proposal_provider: Callable[[Observation], Mapping[str, Any]],
        reviewer: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        correction_mapper: Callable[..., Sequence[Sequence[float]]] | None = None,
        feature_names: Sequence[str] | None = None,
        action_encoder: ActionEncoder | None = None,
        control_hz: float | None = None,
        post_observation_timeout_s: float = 1.0,
        review_packet_builder: ReviewPacketBuilder | None = None,
        request_started: RequestStarted | None = None,
    ) -> None:
        if not isinstance(client, ControlClient):
            raise TypeError("client must be a ControlClient")
        if not callable(proposal_provider) or not callable(reviewer):
            raise TypeError("proposal_provider and reviewer must be callable")
        if correction_mapper is not None and not callable(correction_mapper):
            raise TypeError("correction_mapper must be callable")
        if action_encoder is not None and not callable(action_encoder):
            raise TypeError("action_encoder must be callable")
        if (
            isinstance(post_observation_timeout_s, bool)
            or not isinstance(post_observation_timeout_s, (int, float))
            or not math.isfinite(float(post_observation_timeout_s))
            or post_observation_timeout_s <= 0
        ):
            raise ValueError("post_observation_timeout_s must be finite and positive")
        self.client = client
        self.proposal_provider = proposal_provider
        self.reviewer = reviewer
        self.correction_mapper = correction_mapper
        self.feature_names = tuple(feature_names) if feature_names is not None else None
        self.action_encoder = action_encoder
        if control_hz is not None and (
            isinstance(control_hz, bool)
            or not isinstance(control_hz, (int, float))
            or not math.isfinite(float(control_hz))
            or control_hz <= 0
        ):
            raise ValueError("control_hz must be finite and positive")
        self.control_hz = control_hz
        self.post_observation_timeout_s = float(post_observation_timeout_s)
        if review_packet_builder is not None and not callable(review_packet_builder):
            raise TypeError("review_packet_builder must be callable")
        if request_started is not None and not callable(request_started):
            raise TypeError("request_started must be callable")
        self.review_packet_builder = review_packet_builder
        self.request_started = request_started

    def run_round(self, *, prompt: str = "", context: Mapping[str, Any] | None = None) -> CooperativeResult:
        """Run observe, proposal validation, review, execute, and fresh read.

        The proposal and reviewer are callbacks supplied by the external
        algorithm. A held, unsupported, pending, cancelled, or unknown result
        is returned with its discarded-step count and request ID; only a
        completed execution attempts the bounded fresh-observation read.
        """

        observation = self.client.observe(include_robot=True)
        raw = self.proposal_provider(observation)
        proposal = Proposal.from_payload(raw, observation_id=observation.observation_id)
        packet: dict[str, Any] = {
            "prompt": prompt,
            "context": dict(context or {}),
            "observation": dict(observation.payload),
            "proposal": proposal.to_payload(),
        }
        if self.review_packet_builder is not None:
            extra = self.review_packet_builder(observation, proposal, packet)
            if not isinstance(extra, Mapping):
                raise ValueError("review_packet_builder must return a mapping")
            packet.update(dict(extra))
        raw_decision = self.reviewer(packet)
        if inspect.isawaitable(raw_decision):
            raise RuntimeError("reviewer returned an awaitable; wrap async reviewers with a synchronous adapter")
        decision = validate_decision(
            raw_decision,
            proposal_id=proposal.proposal_id,
            observation_id=proposal.observation_id,
        )
        kind = decision["decision"]
        if kind == "hold":
            return CooperativeResult(
                "held",
                proposal.observation_id,
                proposal.proposal_id,
                decision=kind,
                discarded_steps=len(proposal.actions),
                error=decision["reason"],
            )

        request_id = f"astra-pi05-cooperative:{uuid4().hex}"
        try:
            if kind == "execute_prefix":
                rows = proposal.actions[: decision["execute_steps"]]
                names = self.feature_names or _declared_feature_names(proposal.metadata)
                actions = rows_to_actions(
                    rows,
                    feature_names=names,
                    timestamp_step_s=(1.0 / self.control_hz if self.control_hz else 0.1),
                    metadata={
                        "proposal_id": proposal.proposal_id,
                        "joint_position_unit": proposal.metadata.get("joint_position_unit"),
                    },
                    action_encoder=self.action_encoder,
                )
            else:
                if self.correction_mapper is None:
                    raise CorrectionUnsupported("correction requires an explicit robot-specific mapper")
                mapped = _call_correction_mapper(
                    self.correction_mapper,
                    decision["corrections"],
                    observation=observation,
                )
                mapper_metadata: Mapping[str, Any] = {}
                mapper_names: Sequence[str] | None = None
                if isinstance(mapped, Mapping):
                    mapped_rows = mapped.get("actions")
                    raw_names = mapped.get("feature_names")
                    if raw_names is not None:
                        if not isinstance(raw_names, Sequence) or isinstance(raw_names, (str, bytes)):
                            raise ValueError("correction mapper feature_names must be a sequence")
                        mapper_names = tuple(raw_names)
                    raw_metadata = mapped.get("metadata")
                    if raw_metadata is not None:
                        if not isinstance(raw_metadata, Mapping):
                            raise ValueError("correction mapper metadata must be a mapping")
                        mapper_metadata = dict(raw_metadata)
                    mapped = mapped_rows
                if (
                    isinstance(mapped, Sequence)
                    and not isinstance(mapped, (str, bytes))
                    and all(isinstance(item, Mapping) for item in mapped)
                ):
                    if not 1 <= len(mapped) <= 5:
                        raise ValueError("correction mapper must return 1..5 action payloads")
                    actions = []
                    for item in mapped:
                        action = dict(item)
                        if not isinstance(action.get("values"), Mapping):
                            raise ValueError("mapped correction action must contain public values")
                        if mapper_metadata:
                            metadata_value = dict(action.get("metadata", {}))
                            metadata_value.update(mapper_metadata)
                            action["metadata"] = metadata_value
                        actions.append(action)
                else:
                    if not isinstance(mapped, Sequence) or isinstance(mapped, (str, bytes)):
                        raise ValueError("correction mapper must return action payloads or rows")
                    rows = tuple(tuple(float(value) for value in row) for row in mapped)
                    if not 1 <= len(rows) <= 5:
                        raise ValueError("correction mapper must return 1..5 action rows")
                    names = self.feature_names or mapper_names or _declared_feature_names(proposal.metadata)
                    actions = rows_to_actions(
                        rows,
                        feature_names=names,
                        timestamp_step_s=(1.0 / self.control_hz if self.control_hz else 0.1),
                        metadata={
                            "proposal_id": proposal.proposal_id,
                            "correction": True,
                            **dict(mapper_metadata),
                        },
                        action_encoder=self.action_encoder,
                    )
            execution_control_hz = _resolve_execution_control_hz(actions, requested=self.control_hz)
            if self.request_started is not None:
                self.request_started(request_id)
            execution = self.client.execute(
                actions,
                request_id=request_id,
                observation_id=observation.observation_id,
                steps=len(actions),
                control_hz=execution_control_hz,
                wait=True,
            )
        except CorrectionUnsupported as error:
            return CooperativeResult(
                "unsupported",
                proposal.observation_id,
                proposal.proposal_id,
                decision=kind,
                request_id=request_id,
                discarded_steps=len(proposal.actions),
                error=str(error),
            )
        except UnsupportedOperation as error:
            return CooperativeResult(
                "unsupported",
                proposal.observation_id,
                proposal.proposal_id,
                decision=kind,
                request_id=request_id,
                discarded_steps=len(proposal.actions),
                error=str(error),
            )
        except ControlTransportError as error:
            return CooperativeResult(
                "unknown",
                proposal.observation_id,
                proposal.proposal_id,
                decision=kind,
                request_id=request_id,
                error=str(error),
            )
        except ControlHTTPError as error:
            payload_status = error.payload.get("status") if isinstance(error.payload, Mapping) else None
            payload_code = error.payload.get("code") if isinstance(error.payload, Mapping) else None
            if payload_code == "job_cancelled" or payload_status == "cancelled":
                error_status = "cancelled"
            elif payload_status in {"stopped", "manual_takeover"}:
                error_status = payload_status
            elif error.unknown or payload_status in {"unknown", "uncertain"}:
                error_status = "unknown"
            else:
                error_status = "failed"
            return CooperativeResult(
                error_status,
                proposal.observation_id,
                proposal.proposal_id,
                decision=kind,
                request_id=request_id,
                error=str(error),
            )

        status = str(execution.get("status", "accepted"))
        if status != "completed":
            pending = status in {"accepted", "running", "queued", "pending"}
            terminal_status = (
                status
                if status
                in {
                    "unknown",
                    "uncertain",
                    "cancelled",
                    "stopped",
                    "manual_takeover",
                    "failed",
                    "unsupported",
                }
                else "failed"
            )
            return CooperativeResult(
                "pending" if pending else terminal_status,
                proposal.observation_id,
                proposal.proposal_id,
                decision=kind,
                request_id=request_id,
                discarded_steps=(len(proposal.actions) if kind == "correct" else len(proposal.actions) - len(actions)),
                execution=execution,
                error=str(execution.get("error", "execution did not complete")),
            )
        result_payload = execution.get("result")
        actual_steps = result_payload.get("executed_steps") if isinstance(result_payload, Mapping) else None
        execution_evidence_unknown = (
            isinstance(actual_steps, bool)
            or not isinstance(actual_steps, int)
            or actual_steps < 0
            or actual_steps > len(actions)
        )
        if execution_evidence_unknown:
            actual_steps = 0
        observation_error: str | None = None
        try:
            post_observation = self._new_observation()
        except (ControlTransportError, ControlHTTPError) as error:
            post_observation = None
            observation_error = f"post-execution observation failed: {error}"
        outcome_status = "completed"
        outcome_error = (
            "execution evidence did not include a valid executed_steps count" if execution_evidence_unknown else None
        )
        if post_observation is None:
            outcome_status = "post_observation_pending"
            outcome_error = observation_error or (
                "execution completed but no newer observation was available "
                f"within {self.post_observation_timeout_s:g}s"
            )
        return CooperativeResult(
            outcome_status,
            proposal.observation_id,
            proposal.proposal_id,
            decision=kind,
            request_id=request_id,
            executed_steps=actual_steps,
            discarded_steps=(len(proposal.actions) if kind == "correct" else len(proposal.actions) - actual_steps),
            execution=execution,
            post_observation=post_observation,
            error=outcome_error,
            execution_evidence_unknown=execution_evidence_unknown,
        )

    def _new_observation(self) -> Observation | None:
        """Wait for a publication after the first post-execution snapshot.

        The first read establishes the observation point because a server may
        publish a snapshot while the action is still completing. A later read
        must have a newer ID and, when the server exposes a top-level
        publication sequence or timestamp, a newer value. No clocks from
        another host are compared.
        """

        baseline = self.client.observe(include_robot=True)
        deadline = time.monotonic() + self.post_observation_timeout_s
        while True:
            current = self.client.observe(include_robot=True)
            if _observation_is_after(baseline, current):
                return current
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            time.sleep(min(0.05, remaining))


def _observation_is_after(baseline: Observation, current: Observation) -> bool:
    """Return whether ``current`` is a later same-service publication.

    These are the Control observation response's top-level publication fields.
    Nested camera or business metadata is intentionally ignored because it is
    not a comparable service-wide publication marker.
    """

    if current.observation_id == baseline.observation_id:
        return False
    baseline_sequence = baseline.payload.get("sequence")
    current_sequence = current.payload.get("sequence")
    if (
        baseline_sequence is not None
        and current_sequence is not None
        and (
            isinstance(baseline_sequence, (int, float))
            and not isinstance(baseline_sequence, bool)
            and isinstance(current_sequence, (int, float))
            and not isinstance(current_sequence, bool)
        )
    ):
        return current_sequence > baseline_sequence
    baseline_timestamp = baseline.payload.get("published_timestamp_ns")
    current_timestamp = current.payload.get("published_timestamp_ns")
    if (
        baseline_timestamp is not None
        and current_timestamp is not None
        and (
            isinstance(baseline_timestamp, (int, float))
            and not isinstance(baseline_timestamp, bool)
            and isinstance(current_timestamp, (int, float))
            and not isinstance(current_timestamp, bool)
        )
    ):
        return current_timestamp > baseline_timestamp
    return True


def _call_correction_mapper(
    mapper: Callable[..., Any],
    corrections: Sequence[Mapping[str, Any]],
    *,
    observation: Observation,
) -> Any:
    """Call a named observation-aware mapper without hiding old test doubles.

    New adapters receive the complete detached ``Observation`` so they can
    inspect the declared robot state and control generation.  The ``state``
    keyword is retained only for the existing software fixtures; it is not a
    generic end-effector conversion path.
    """

    try:
        signature = inspect.signature(mapper)
    except (TypeError, ValueError):
        signature = None
    if signature is not None and "observation" in signature.parameters:
        return mapper(corrections, observation=observation)
    if signature is not None and "state" in signature.parameters:
        return mapper(corrections, state=observation.robot)
    return mapper(corrections, observation=observation)


def _declared_feature_names(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    value = metadata.get("feature_names")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("proposal metadata must declare feature_names for the binding action space")
    return tuple(value)


def _resolve_execution_control_hz(actions: Sequence[Mapping[str, Any]], *, requested: float | None) -> float | None:
    """Honor an explicitly declared public fixed-rate correction schedule."""

    declared: list[float] = []
    for action in actions:
        metadata = action.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        if metadata.get("duration_semantics") != "public_control_hz_step":
            continue
        rate = metadata.get("control_hz")
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(float(rate)) or rate <= 0:
            raise ValueError("correction action must declare a finite positive control_hz")
        declared.append(float(rate))
    if not declared:
        return requested
    if any(not math.isclose(rate, declared[0], rel_tol=1e-9, abs_tol=1e-9) for rate in declared[1:]):
        raise ValueError("correction actions declare conflicting control_hz values")
    if requested is not None and not math.isclose(float(requested), declared[0], rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("session control_hz conflicts with correction action control_hz")
    return declared[0]


__all__ = [
    "CooperativeLoop",
    "CooperativeResult",
    "CorrectionUnsupported",
    "ActionEncoder",
    "Proposal",
    "ReviewPacketBuilder",
    "RequestStarted",
    "rows_to_actions",
]
