"""Repeat the Astra/π0.5 loop through Deploy's public Agent client.

This module owns upper-layer orchestration only.  It obtains proposals with
``ControlClient.propose``, submits selected actions with ``ControlClient`` and
records detached evidence.  Robot drivers, cameras, SSH sessions, and
kinematics are deliberately outside the module.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from uuid import uuid4

from agents.astra_pi05.cooperative import (
    ActionEncoder,
    CooperativeLoop,
    CooperativeResult,
    Proposal,
    ReviewPacketBuilder,
)
from agents.astra_pi05.recording import SessionRecorder
from embodirun.client import ControlClient, Observation


def _to_list(value: Any) -> Any:
    """Detach tensor-like fixture values without importing a model runtime."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _finite_row(value: Any, *, label: str) -> list[float]:
    value = _to_list(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 12:
        raise ValueError(f"{label} must contain 12 values")
    result = [float(item) for item in value]
    if any(not math.isfinite(item) for item in result):
        raise ValueError(f"{label} contains a non-finite value")
    return result


def _metadata_from(payload: Mapping[str, Any], nested: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for source in (nested, payload):
        candidate = source.get("metadata")
        if candidate is not None:
            if not isinstance(candidate, Mapping):
                raise ValueError("proposal metadata must be a mapping")
            _merge_metadata(metadata, candidate)
        candidate = source.get("action_metadata")
        if candidate is not None:
            if not isinstance(candidate, Mapping):
                raise ValueError("proposal action_metadata must be a mapping")
            _merge_metadata(metadata, candidate)
    return metadata


def _merge_metadata(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if key in target and target[key] != value:
            raise ValueError(f"conflicting proposal metadata for {key}")
        target[key] = value


def _action_source(payload: Mapping[str, Any], nested: Mapping[str, Any]) -> Any:
    actions = payload.get("actions")
    if actions is None:
        actions = nested.get("actions")
    if actions is None:
        proposal_value = payload.get("proposal")
        if isinstance(proposal_value, Mapping):
            actions = proposal_value.get("actions")
    if isinstance(actions, Mapping):
        actions = actions.get("data", actions.get("actions"))
    return _to_list(actions)


def _normalize_rows(
    payload: Mapping[str, Any],
    *,
    feature_names: Sequence[str] | None,
    action_decoder: Callable[[Mapping[str, Any]], Sequence[float]] | None = None,
) -> tuple[list[list[float]], tuple[str, ...] | None]:
    nested = payload.get("proposal")
    nested = nested if isinstance(nested, Mapping) else {}
    actions = _action_source(payload, nested)
    # The Deploy policy payload commonly wraps a 50x12 chunk as one action's
    # ``values.data``.  Unwrap that container first.
    if (
        isinstance(actions, Sequence)
        and not isinstance(actions, (str, bytes))
        and len(actions) == 1
        and isinstance(actions[0], Mapping)
    ):
        item = actions[0]
        values = item.get("values", item)
        if isinstance(values, Mapping):
            chunk = values.get("data", values.get("actions"))
            if chunk is not None:
                actions = _to_list(chunk)

    names: tuple[str, ...] | None = tuple(feature_names) if feature_names is not None else None
    if names is None:
        raw_names = payload.get("feature_names", nested.get("feature_names"))
        if raw_names is None:
            metadata = _metadata_from(payload, nested)
            raw_names = metadata.get("feature_names")
        if isinstance(raw_names, Sequence) and not isinstance(raw_names, (str, bytes)):
            names = tuple(str(name) for name in raw_names)

    actions = _to_list(actions)
    if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)) or len(actions) != 50:
        raise ValueError("Deploy proposal must contain exactly 50 action rows")

    rows: list[list[float]] = []
    inferred_names: tuple[str, ...] | None = names
    for index, item in enumerate(actions):
        item = _to_list(item)
        if isinstance(item, Mapping):
            values = item.get("values", item.get("action", item))
            values = _to_list(values)
            if isinstance(values, Mapping):
                data = values.get("data", values.get("joint_positions"))
                if data is not None:
                    values = _to_list(data)
                elif action_decoder is not None and {
                    "type",
                    "left",
                    "right",
                }.issubset(values):
                    values = action_decoder(values)
                elif inferred_names is not None:
                    try:
                        values = [values[name] for name in inferred_names]
                    except KeyError as error:
                        raise ValueError(f"proposal action {index} is missing a declared feature") from error
                else:
                    keys = tuple(values)
                    if len(keys) == 12 and all(isinstance(key, str) for key in keys):
                        inferred_names = keys
                        values = [values[key] for key in keys]
                    else:
                        raise ValueError("mapped proposal actions require explicit feature_names")
            rows.append(_finite_row(values, label=f"proposal.actions[{index}]"))
        else:
            rows.append(_finite_row(item, label=f"proposal.actions[{index}]"))
    if inferred_names is not None and len(inferred_names) != 12:
        raise ValueError("feature_names must contain exactly 12 names")
    return rows, inferred_names


def normalize_public_proposal(
    payload: Mapping[str, Any],
    *,
    observation_id: str,
    feature_names: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    action_decoder: Callable[[Mapping[str, Any]], Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Convert one public ``/v1/propose`` response into the review contract."""

    if not isinstance(payload, Mapping):
        raise ValueError("Deploy proposal response must be a mapping")
    nested = payload.get("proposal")
    nested = nested if isinstance(nested, Mapping) else {}
    declared_observation = payload.get("observation_id", nested.get("observation_id"))
    if declared_observation is not None and declared_observation != observation_id:
        raise ValueError("Deploy proposal observation_id does not match observation")
    rows, names = _normalize_rows(payload, feature_names=feature_names, action_decoder=action_decoder)
    proposal_metadata = _metadata_from(payload, nested)
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise ValueError("proposal metadata must be a mapping")
        # Caller metadata supplies missing declarations (for example when an
        # older public response omits feature_names); it may not relabel a
        # model-declared unit or action encoding.
        for key, value in metadata.items():
            if key in proposal_metadata and proposal_metadata[key] != value:
                raise ValueError(f"conflicting proposal metadata for {key}")
            proposal_metadata[key] = value
    if names is not None:
        proposal_metadata.setdefault("feature_names", list(names))
    proposal_id = payload.get(
        "proposal_id",
        nested.get("proposal_id", payload.get("request_id", f"proposal-{uuid4().hex}")),
    )
    if not isinstance(proposal_id, str) or not proposal_id.strip():
        raise ValueError("Deploy proposal must have a non-empty proposal_id")
    normalized = {
        "proposal_id": proposal_id,
        "observation_id": observation_id,
        "actions": rows,
        "metadata": proposal_metadata,
    }
    # Validate before the session invokes a reviewer.  This keeps a public
    # proposal with missing units or semantics from being treated as a valid
    # π0.5 packet by any caller.
    Proposal.from_payload(normalized, observation_id=observation_id)
    return normalized


def _sync_review(reviewer: Any, packet: Mapping[str, Any]) -> Mapping[str, Any]:
    callback = getattr(reviewer, "review", reviewer)
    result = callback(packet)
    if inspect.isawaitable(result):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            result = asyncio.run(result)
        else:
            raise RuntimeError("an async reviewer cannot run inside an active event loop; use an async session wrapper")
    if not isinstance(result, Mapping):
        raise ValueError("reviewer must return a mapping")
    return result


class PublicCooperativeSession:
    """Run bounded repeated rounds against the public Deploy client."""

    def __init__(
        self,
        client: ControlClient,
        *,
        reviewer: Any,
        instruction: str,
        proposal_provider: Callable[[Observation], Mapping[str, Any]] | None = None,
        proposal_metadata: Mapping[str, Any] | None = None,
        correction_mapper: Callable[..., Any] | None = None,
        feature_names: Sequence[str] | None = None,
        action_encoder: ActionEncoder | None = None,
        action_decoder: Callable[[Mapping[str, Any]], Sequence[float]] | None = None,
        control_hz: float | None = None,
        max_rounds: int = 1,
        post_observation_timeout_s: float = 1.0,
        review_packet_builder: ReviewPacketBuilder | None = None,
        recorder: SessionRecorder | None = None,
        supported_skills: Sequence[str] = ("grasp",),
    ) -> None:
        if not isinstance(client, ControlClient):
            raise TypeError("client must be a ControlClient")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds <= 0:
            raise ValueError("max_rounds must be a positive integer")
        if (
            not isinstance(supported_skills, Sequence)
            or isinstance(supported_skills, (str, bytes))
            or not supported_skills
            or any(not isinstance(skill, str) or not skill.strip() for skill in supported_skills)
        ):
            raise ValueError("supported_skills must contain at least one non-empty name")
        if not callable(getattr(reviewer, "review", reviewer)):
            raise TypeError("reviewer must be callable or provide review()")
        if proposal_provider is not None and not callable(proposal_provider):
            raise TypeError("proposal_provider must be callable")
        self.client = client
        self.reviewer = reviewer
        self.instruction = instruction
        self.proposal_provider = proposal_provider
        self.proposal_metadata: dict[str, Any] = {}
        if action_encoder is not None:
            default_metadata = getattr(action_encoder, "proposal_metadata", {})
            if callable(default_metadata):
                default_metadata = default_metadata()
            if default_metadata:
                if not isinstance(default_metadata, Mapping):
                    raise TypeError("action_encoder proposal_metadata must be a mapping")
                self.proposal_metadata.update(dict(default_metadata))
        if proposal_metadata is not None:
            self.proposal_metadata.update(dict(proposal_metadata))
        self.feature_names = tuple(feature_names) if feature_names is not None else None
        encoder_names = getattr(action_encoder, "feature_names", None) if action_encoder is not None else None
        if encoder_names is not None and self.feature_names is not None and tuple(encoder_names) != self.feature_names:
            raise ValueError("action_encoder feature_names must match session feature_names")
        self.max_rounds = max_rounds
        self.recorder = recorder
        self._active_request_id: str | None = None
        self._active_lock = threading.Lock()
        self.supported_skills = frozenset(supported_skills)
        if action_decoder is None and action_encoder is not None:
            candidate = getattr(action_encoder, "decode", None)
            if callable(candidate):
                action_decoder = candidate
        if action_decoder is not None and not callable(action_decoder):
            raise TypeError("action_decoder must be callable")
        self.action_decoder = action_decoder
        self._round = 0
        self._loop = CooperativeLoop(
            client,
            proposal_provider=self._proposal,
            reviewer=lambda packet: _sync_review(self.reviewer, packet),
            correction_mapper=correction_mapper,
            feature_names=self.feature_names,
            action_encoder=action_encoder,
            control_hz=control_hz,
            post_observation_timeout_s=post_observation_timeout_s,
            review_packet_builder=review_packet_builder,
            request_started=self._set_active_request,
        )

    @property
    def active_request_id(self) -> str | None:
        with self._active_lock:
            return self._active_request_id

    def _set_active_request(self, request_id: str) -> None:
        with self._active_lock:
            self._active_request_id = request_id
        if self.recorder is not None:
            self.recorder.record("execute_submitted", request_id=request_id, round=self._round)

    def _clear_active_request(self) -> None:
        with self._active_lock:
            self._active_request_id = None

    def _proposal(self, observation: Observation) -> Mapping[str, Any]:
        if self.proposal_provider is not None:
            return self.proposal_provider(observation)
        request_id = f"rpent-cooperative:{uuid4().hex}:proposal:{self._round + 1}"
        response = self.client.propose(
            request_id=request_id,
            observation_id=observation.observation_id,
            instruction=self.instruction,
        )
        return normalize_public_proposal(
            response,
            observation_id=observation.observation_id,
            feature_names=self.feature_names,
            metadata=self.proposal_metadata,
            action_decoder=self.action_decoder,
        )

    def run_round(
        self,
        *,
        prompt: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> CooperativeResult:
        self._round += 1
        # A new round owns a fresh request identity.  A previous rejected or
        # held round must never leave a stale cancellation target behind.
        self._clear_active_request()
        result = self._loop.run_round(
            prompt=self.instruction if prompt is None else prompt,
            context={"round": self._round, **dict(context or {})},
        )
        if self.recorder is not None:
            self.recorder.record("round_result", round=self._round, result=result.to_dict())
        if result.status not in {"pending", "unknown", "uncertain"}:
            self._clear_active_request()
        return result

    def run(
        self,
        *,
        prompt: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        rounds: list[dict[str, Any]] = []
        if self.recorder is not None:
            self.recorder.record(
                "session_started",
                instruction=self.instruction,
                max_rounds=self.max_rounds,
            )
        try:
            for _ in range(self.max_rounds):
                result = self.run_round(prompt=prompt, context=context)
                rounds.append(result.to_dict())
                if result.status != "completed" or result.execution_evidence_unknown:
                    break
            status = (
                "completed"
                if rounds and rounds[-1]["status"] == "completed"
                else (rounds[-1]["status"] if rounds else "failed")
            )
            summary = {
                "status": status,
                "task_success": "unverified",
                "rounds": rounds,
                "round_count": len(rounds),
            }
            if self.recorder is not None:
                self.recorder.finalize(summary)
            return summary
        except BaseException as error:
            if self.recorder is not None:
                self.recorder.record("session_error", error=repr(error))
                self.recorder.finalize(
                    {
                        "status": "failed",
                        "task_success": "unverified",
                        "rounds": rounds,
                        "error": str(error),
                    }
                )
            raise

    def run_skill(
        self,
        skill: str,
        prompt: str,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compatibility name for the old RPent session entrypoint."""

        if skill not in self.supported_skills:
            raise ValueError(f"unsupported skill {skill!r}; configure supported_skills explicitly")
        return self.run(prompt=prompt, context=context)

    def cancel(self) -> dict[str, Any]:
        request_id = self.active_request_id
        if request_id is None:
            raise RuntimeError("no public execute request is active")
        response = self.client.cancel(request_id)
        if self.recorder is not None:
            self.recorder.record("cancel_requested", request_id=request_id, response=response)
        return response

    def stop(self) -> dict[str, Any]:
        request_id = self.active_request_id
        if request_id is None:
            raise RuntimeError("no public execute request is active")
        response = self.client.stop(request_id)
        if self.recorder is not None:
            self.recorder.record("stop_requested", request_id=request_id, response=response)
        return response


__all__ = ["PublicCooperativeSession", "normalize_public_proposal"]
