"""Narrow read-only restart evidence for a fully owned failed all-stop.

No I/O or fault clearing here. Old services lack hardware safety export: their
fallback requires successful held/arm evidence AND an explicit maintenance
operator assertion that no enable was attempted afterwards. Never infer a
missing torque-fault flag as false merely from present zero velocity.
"""

from __future__ import annotations

import math
import time

from .hardware import WHEEL_NAMES


def legacy_safety(robot, snapshot):
    assertion = snapshot.get("legacy_stop_only_review", {})
    run = assertion.get("last_arm_run", {})
    arm = run.get("arm_feedback", {})
    started, completed = run.get("started_at_ns"), run.get("completed_at_ns")
    stop_stamp = snapshot["failed_stop"].get("sent_timestamp_ns")
    if (
        assertion.get("no_enable_attempts_since") is not True
        or arm.get("armed") is not True
        or arm.get("errors") != []
        or not arm.get("writes")
        or any(w.get("status") != "accepted" for w in arm["writes"])
        or type(started) is not int
        or type(completed) is not int
        or type(stop_stamp) is not int
        or not 0 < started <= completed <= stop_stamp
    ):
        raise ValueError("legacy handoff needs reviewed successful arm and no-later-enable assertion")
    names = set(robot._motor_names_for_control())
    positions = names - set(WHEEL_NAMES)
    for key in ("status_before", "status_after"):
        held = snapshot[key].get("metadata", {}).get("holding_resume", {})
        if (
            held.get("restored") is not True
            or held.get("register_writes") != 0
            or held.get("stop_confirmed") is not True
            or held.get("errors") != []
            or set(held.get("motors", [])) != positions
            or len(held.get("motors", [])) != len(positions)
            or held.get("cold_motors") != []
            or held.get("wheel_torque") != dict.fromkeys(WHEEL_NAMES, 1)
            or any(type(v) is not int for v in held["wheel_torque"].values())
        ):
            raise ValueError("legacy ownership must be proven by the prior full powered held restore")
    goals = {w["motor"]: w["value"] for w in snapshot["failed_stop"]["writes"] if w["field"] == "Goal_Position"}
    return {
        "stop_unconfirmed": True,
        "torque_ownership_uncertain": False,
        "owned_motors": sorted(names),
        "held_goals": goals,
        "gripper_goals": {n: v for n, v in goals.items() if n.endswith("_gripper")},
    }


def validate_snapshot(robot, snapshot):
    if not isinstance(snapshot, dict) or snapshot.get("source") != "physical":
        raise ValueError("physical fault snapshot required")
    if snapshot.get("ports") != robot._ports or snapshot.get("boot_id") != robot._boot_id():
        raise ValueError("fault handoff ports/boot mismatch")
    stamp = snapshot.get("created_monotonic_s")
    if type(stamp) not in (int, float) or not math.isfinite(stamp) or not 0 <= time.monotonic() - stamp <= 120:
        raise ValueError("fault handoff expired")
    names = set(robot._motor_names_for_control())
    if not set(WHEEL_NAMES) < names:
        raise ValueError("fault handoff requires the configured wheels and position motors")
    stop = snapshot.get("failed_stop", {})
    if (
        stop.get("scope") != "all"
        or stop.get("stop_confirmed") is not False
        or stop.get("command_accepted") is not True
        or stop.get("errors") != []
        or not stop.get("samples")
    ):
        raise ValueError("full failed all-stop with accepted writes and feedback required")
    writes = stop.get("writes", [])
    if len(writes) != len(names) or {w.get("motor") for w in writes} != names:
        raise ValueError("failed all-stop must cover every configured motor exactly once")
    goals = {}
    for write in writes:
        name, value = write["motor"], write.get("value")
        field = "Goal_Velocity" if name in WHEEL_NAMES else "Goal_Position"
        if (
            write.get("status") != "accepted"
            or write.get("field") != field
            or type(value) is not int
            or (value != 0 if name in WHEEL_NAMES else not 0 <= value <= 4095)
        ):
            raise ValueError("invalid failed-stop write evidence")
        if name not in WHEEL_NAMES:
            goals[name] = value
    for key in ("status_before", "status_after"):
        status = snapshot.get(key, {})
        if (
            status.get("connected") is not True
            or status.get("armed") is not False
            or status.get("control_owner") is not None
            or status.get("control_state") != {"arms": False, "base": False}
            or status.get("stop_unconfirmed") is not True
            or status.get("recording") is not False
            or status.get("error")
            or status.get("observation_errors")
            or status.get("feedback") != stop
        ):
            raise ValueError("prior service must be unowned/inactive with the same latched failed stop")
    samples, previous = snapshot.get("samples", []), -1
    if not isinstance(samples, list) or len(samples) != 3:
        raise ValueError("three fresh observations required; they do not clear the fault")
    for sample in samples:
        stamp, raw = sample.get("state_timestamp_ns"), sample.get("raw", {})
        if (
            type(stamp) is not int
            or stamp <= previous
            or sample.get("state_cached") is not False
            or sample.get("errors") != []
            or set(raw) != names
            or any(
                type(v.get(k)) is not int
                for v in raw.values()
                for k in ("Present_Position", "Present_Velocity", "Moving")
            )
        ):
            raise ValueError("fault observations incomplete, cached or invalid")
        previous = stamp
    before = snapshot["status_before"].get("hardware_safety")
    after = snapshot["status_after"].get("hardware_safety")
    if before is None and after is None:
        state = legacy_safety(robot, snapshot)
    else:
        if not isinstance(before, dict) or before != after:
            raise ValueError("hardware safety state changed during capture")
        state = before
    if (
        state.get("stop_unconfirmed") is not True
        or type(state.get("torque_ownership_uncertain")) is not bool
        or not isinstance(state.get("owned_motors"), list)
        or len(state["owned_motors"]) != len(names)
        or set(state["owned_motors"]) != names
        or state.get("held_goals") != goals
        or state.get("gripper_goals") != {n: v for n, v in goals.items() if n.endswith("_gripper")}
        or any(type(v) is not int for v in (*state["held_goals"].values(), *state["gripper_goals"].values()))
    ):
        raise ValueError("incomplete or inconsistent ownership/hold/gripper evidence")
    # Preserve a true torque-ownership fault; a successful stop must not clear it.
    return state
