"""Single shared-snapshot policy proposals.

This module is the non-executing bridge between a retained Control snapshot,
the configured binding, and the existing session-oriented InferenceClient.
It creates one inference session, performs one mapper step, maps the returned
policy actions into detached ``RobotAction`` payloads, and closes that session
in ``finally``.  It never prepares a robot, captures a new camera frame, or
submits an action to the arbiter.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from embodirun.model_services import PolicyObservation, PolicyResult
from embodirun.robots import RobotAction


class ProposalError(RuntimeError):
    """A proposal could not be generated from the requested snapshot."""


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _policy_payload(result: PolicyResult) -> dict[str, Any]:
    return {
        "request_id": result.request_id,
        "session_id": result.session_id,
        "step_id": result.step_id,
        "session_revision": result.session_revision,
        "action_space": result.action_space,
        "policy_revision": result.policy_revision,
        "timing": dict(result.timing),
        "actions": [{"type": action.kind, "values": _json_value(action.values)} for action in result.actions],
    }


def _action_payload(action: RobotAction) -> dict[str, Any]:
    return {
        "timestamp_s": action.timestamp_s,
        "values": _json_value(action.values),
        "metadata": _json_value(action.metadata),
    }


def generate_proposal(
    service: Any,
    *,
    observation_id: str,
    instruction: str,
    runtime_id: str | None,
    request_id: str,
    timeout_s: float,
) -> dict[str, Any]:
    """Generate one mapped proposal from exactly ``observation_id``.

    ``service.proposal_context`` is a Control-owned composition seam.  It
    returns the already retained state/frame objects and selected binding; it
    must not call ``observe`` or open a device.  The returned ``actions`` are
    proposed RobotAction values only and have no execution/job status.
    """

    if not isinstance(observation_id, str) or not observation_id.strip():
        raise ProposalError("observation_id must be a non-empty string")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ProposalError("instruction must be a non-empty string")
    if not isinstance(request_id, str) or not request_id.strip():
        raise ProposalError("request_id must be a non-empty string")
    if (
        not isinstance(timeout_s, (int, float))
        or isinstance(timeout_s, bool)
        or not math.isfinite(float(timeout_s))
        or timeout_s <= 0
    ):
        raise ProposalError("timeout_s must be finite and positive")

    profile_config, binding, robot_observation, frames = service.proposal_context(
        observation_id,
        runtime_id=runtime_id,
    )
    client = service._inference_client(profile_config, float(timeout_s))
    mapper = None
    session = None
    try:
        mapper = binding.mapper_factory()
        session = client.open_session(
            robot_id=profile_config.robot_id,
            action_space=mapper.policy_action_space,
            metadata={"purpose": "proposal", "observation_id": observation_id},
        )
        policy_request = mapper.map_observation(
            robot_observation,
            session_id=session.session_id,
            request_id=request_id,
            step_id=0,
            instruction=instruction,
            frames=tuple(frames),
        )
        if not isinstance(policy_request, PolicyObservation):
            raise ProposalError("binding returned an invalid PolicyObservation")
        metadata = dict(policy_request.metadata)
        metadata.update({"observation_id": observation_id, "snapshot_id": observation_id})
        policy_request = replace(policy_request, metadata=metadata)
        result = client.step(policy_request)
        if not isinstance(result, PolicyResult):
            raise ProposalError("inference client returned an invalid PolicyResult")
        if (
            result.request_id != policy_request.request_id
            or result.session_id != policy_request.session_id
            or result.step_id != policy_request.step_id
        ):
            raise ProposalError("inference result request_id/session_id/step_id does not match the proposal")
        actions = tuple(mapper.map_result(result))
        if not actions or any(not isinstance(action, RobotAction) for action in actions):
            raise ProposalError("binding returned no valid RobotAction proposal")
        return {
            "status": "proposed",
            "request_id": request_id,
            "observation_id": observation_id,
            "runtime_id": profile_config.runtime_id,
            "proposal": _policy_payload(result),
            "actions": [_action_payload(action) for action in actions],
            "session": {
                "session_id": session.session_id,
                "closed": True,
            },
        }
    finally:
        # Close only this temporary session, then release a non-wireless
        # client through the service's existing lifecycle hook.  Wireless
        # clients are cached by that hook and are intentionally not unloaded.
        try:
            if session is not None:
                client.close(session.session_id)
        finally:
            release = getattr(service, "_release_inference_client", None)
            if callable(release):
                release(client, profile_config)


__all__ = ["ProposalError", "generate_proposal"]
