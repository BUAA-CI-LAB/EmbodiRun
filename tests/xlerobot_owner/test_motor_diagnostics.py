import copy

import pytest
from tests.xlerobot_owner.test_hardware import _robot
from tests.xlerobot_owner.test_server import HEADERS, station

from embodirun_xlerobot_owner.motor_diagnostics import read_sts3215_present_block


def word(value):
    encoded = abs(value) | (0x8000 if value < 0 else 0)
    return [encoded & 255, encoded >> 8]


class Packet:
    def __init__(self, bus):
        self.bus, self.calls, self.overrides = bus, [], {}

    def readTxRx(self, port, motor_id, address, length):
        assert port is self.bus.port_handler
        self.calls.append((motor_id, address, length))
        if address in self.overrides:
            value = self.overrides[address]
            if isinstance(value, Exception):
                raise value
            return value
        name = next(n for n, spec in self.bus.motors.items() if spec["id"] == motor_id)
        fields = self.bus.values[name]
        values = {
            3: word(777),
            0: [2, 54],
            80: [1, 20, 50],
            56: (
                word(fields["Present_Position"])
                + word(fields["Present_Velocity"])
                + [0, 0, 120, 35, 0, 0, fields["Moving"]]
            ),
            46: word(fields["Goal_Velocity"]),
            33: [fields["Operating_Mode"]],
            40: [fields["Torque_Enable"]],
            69: [12, 0],
        }
        return values[address], 0, 0


def setup(tmp_path):
    robot, buses = _robot(tmp_path, enable_base=True, wheel_directions={"left": -1, "right": 1})
    bus = buses["right"]
    bus.port_handler = object()
    bus.packet_handler = Packet(bus)
    return robot, buses, bus.packet_handler


@pytest.mark.parametrize("velocity", [0, 50, -50, 32767, -32767])
def test_present_block_sign_magnitude_and_single_transaction(tmp_path, velocity):
    robot, buses, packet = setup(tmp_path)
    try:
        buses["right"].values["base_left_wheel"]["Present_Velocity"] = velocity
        result = read_sts3215_present_block(packet, buses["right"].port_handler, 9)
        assert result["ok"] and result["fields"]["Present_Velocity"] == velocity
        assert result["raw_bytes"][2:4] == word(velocity)
        assert packet.calls == [(9, 56, 11)]
        assert result["finished_monotonic_ns"] >= result["started_monotonic_ns"]
    finally:
        robot.close()


@pytest.mark.parametrize(
    "response",
    [
        ([0] * 10, 0, 0),
        ([0] * 12, 0, 0),
        ([0] * 11, -3002, 0),
        ([0] * 11, 0, 8),
        ([0] * 11, None, 0),
        ([0] * 11, 0, None),
        ([True] * 11, 0, 0),
        ([256] * 11, 0, 0),
        ([0] * 11, False, 0),
        (None, 0, 0),
        OSError("checksum/transport failure"),
    ],
)
def test_bad_reply_never_decodes_as_zero_or_falls_back(tmp_path, response):
    robot, buses, packet = setup(tmp_path)
    try:
        packet.overrides[56] = response
        result = robot.arm("base")
        assert not result["armed"] and not result["writes"]
        record = result["preflight"]["records"]["base_left_wheel"]
        assert record["present_block"]["errors"]
        assert "Present_Velocity" not in record["fields"]
        assert all(not bus.writes for bus in buses.values())
    finally:
        robot.close()


@pytest.mark.parametrize("velocity,moving", [(-50, 1), (50, 0), (0, 1), (1, 0), (0, 2)])
def test_same_reply_nonzero_still_refuses_before_writes(tmp_path, velocity, moving):
    robot, buses, _ = setup(tmp_path)
    try:
        buses["right"].values["base_left_wheel"].update(Present_Velocity=velocity, Moving=moving)
        result = robot.arm("base")
        assert not result["armed"] and result["writes"] == []
        assert "motor is moving during preflight" in str(result["errors"])
    finally:
        robot.close()


def test_wheel_preflight_uses_block_not_interleaved_scalar_reads(tmp_path):
    robot, buses, packet = setup(tmp_path)
    bus = buses["right"]
    scalar = bus.read

    def scalar_read(field, motor, normalize=False):
        if motor.startswith("base_") and field in {"Present_Position", "Present_Velocity", "Moving"}:
            raise AssertionError("wheel snapshot must not perform interleaved scalar reads")
        return scalar(field, motor, normalize)

    try:
        bus.read = scalar_read
        _, records, errors = robot._preflight(scope="base")
        assert not errors
        assert packet.calls == [(9, 56, 11), (10, 56, 11)]
        assert records["base_left_wheel"]["present_block"]["ok"]
        assert not bus.writes
    finally:
        bus.read = scalar
        robot.close()


@pytest.mark.parametrize(
    "sequence,expected",
    [
        ([(50, 1), (0, 0), (0, 0), (0, 0)], True),
        ([(-50, 1), (0, 0), (0, 0), (50, 1), (0, 0), (0, 0), (0, 0)], True),
        ([(50, 1)] * 41, False),
        ([(50, 1), (0, 0), (0, 0)] * 14, False),
        ([(50, 1), None, (0, 0), (0, 0), (0, 0)], False),
        # Observed nine-packet prefix; only a hypothetical tenth zero completes
        # settling. This tests the limit, not a claim that real hardware did so.
        ([(-50, 1), (0, 0), (50, 1), (0, 0), (0, 0), (50, 1), (-50, 1), (0, 0), (0, 0), (0, 0)], True),
        ([(50, 1)] * 38 + [(0, 0)] * 3, True),
        ([(50, 1)] * 39 + [(0, 0)] * 2, False),
    ],
)
def test_stopped_owned_wheel_wait_retains_every_packet(tmp_path, monkeypatch, sequence, expected):
    robot, buses, packet = setup(tmp_path)
    bus = buses["right"]
    name = "base_left_wheel"
    bus.values[name]["Torque_Enable"] = 1
    robot._owned_torque_names.add(name)
    robot._stopped_scopes.add("base")
    original = packet.readTxRx
    seen = []

    def read(port, motor_id, address, length):
        if motor_id == 9 and address == 56:
            sample = sequence[min(len(seen), len(sequence) - 1)]
            seen.append(sample)
            if sample is None:
                return [], -3002, 0
            bus.values[name].update(Present_Velocity=sample[0], Moving=sample[1])
        return original(port, motor_id, address, length)

    monkeypatch.setattr(packet, "readTxRx", read)
    monkeypatch.setattr("embodirun_xlerobot_owner.hardware.time.sleep", lambda _: None)
    try:
        _, records, errors = robot._preflight(scope="base")
        trace = records[name]["stationary_wait"]
        assert trace["settled"] is expected and bool(errors) is not expected
        assert len(trace["samples"]) == len(seen) <= 41
        if not expected and None not in seen:
            assert len(seen) == 41
        assert trace["samples"][0]["fields"]["Present_Velocity"] == sequence[0][0]
        if None in seen:
            assert seen[-1] is None and not trace["samples"][-1]["ok"]
        assert all(not b.writes for b in buses.values())
    finally:
        robot.close()


@pytest.mark.parametrize("fault", ["active_arms", "nonzero_goal", "unowned", "stop_latch"])
def test_stationary_wait_never_runs_for_active_or_untrusted_state(tmp_path, fault):
    robot, buses, packet = setup(tmp_path)
    name = "base_left_wheel"
    wheel = buses["right"].values[name]
    wheel.update(Torque_Enable=1, Present_Velocity=50, Moving=1)
    robot._owned_torque_names.add(name)
    robot._stopped_scopes.add("base")
    if fault == "active_arms":
        robot._active_scopes.add("arms")
    elif fault == "nonzero_goal":
        wheel["Goal_Velocity"] = 100
    elif fault == "unowned":
        robot._owned_torque_names.clear()
    elif fault == "stop_latch":
        robot._stop_uncertain = True
    try:
        _, records, errors = robot._preflight(scope="base")
        assert errors and "stationary_wait" not in records[name]
        assert len(packet.calls) == 2
        assert all(not b.writes for b in buses.values())
    finally:
        robot._active_scopes.clear()
        robot.close()


def test_idle_diagnostics_preserve_state_fault_latches_and_raw_data(tmp_path):
    robot, buses, _ = setup(tmp_path)
    try:
        robot._stop_uncertain = True
        robot._torque_ownership_uncertain = True
        before = copy.deepcopy(robot._last_audit)
        result = robot.read_motor_diagnostics("base_left_wheel")
        assert result["ok"] and result["read_only"] and result["register_writes"] == 0
        assert len(result["transactions"]) == 8 and not result["atomic_snapshot"]
        assert result["transactions"][-1]["fields"]["Velocity_Unit_factor"] == 50
        assert robot._stop_uncertain and robot._torque_ownership_uncertain
        assert robot._last_audit == before and not robot.armed
        assert all(not b.writes for b in buses.values())
        assert not robot.arm("base")["armed"]
    finally:
        robot.close()


def test_identity_failure_aborts_diagnostics_and_optional_failure_stays_unknown(tmp_path):
    robot, _, packet = setup(tmp_path)
    try:
        packet.overrides[3] = (word(123), 0, 0)
        result = robot.read_motor_diagnostics("base_left_wheel")
        assert not result["ok"] and len(packet.calls) == 1
        packet.overrides = {80: ([], -3001, 0)}
        result = robot.read_motor_diagnostics("base_left_wheel")
        assert not result["ok"]
        assert result["transactions"][-1]["fields"] == {}
    finally:
        robot.close()


@pytest.mark.parametrize("target", ["head_motor_2", "left_arm_shoulder_pan", "9", "unknown"])
def test_only_configured_wheel_targets(tmp_path, target):
    robot, _, packet = setup(tmp_path)
    try:
        with pytest.raises(ValueError):
            robot.read_motor_diagnostics(target)
        assert packet.calls == []
    finally:
        robot.close()


@pytest.mark.parametrize("scope", ["arms", "base"])
def test_diagnostics_refuse_active_control_without_serial_reads(tmp_path, scope):
    robot, _, packet = setup(tmp_path)
    try:
        robot._active_scopes.add(scope)
        with pytest.raises(RuntimeError, match="inactive"):
            robot.read_motor_diagnostics("base_left_wheel")
        assert packet.calls == []
    finally:
        robot._active_scopes.clear()
        robot.close()


@pytest.mark.asyncio
async def test_diagnostic_api_auth_queries_and_owner_guard(tmp_path):
    robot, _, packet = setup(tmp_path)
    async with station(tmp_path / "api", robot=robot, robot_api=True) as (platform, client):
        url = "/robot/diagnostics?motor=base_left_wheel"
        assert (await client.get(url)).status == 401
        assert (await client.get(url + "&motor=base_right_wheel", headers=HEADERS)).status == 400
        assert (await client.get(url + "&count=100", headers=HEADERS)).status == 400
        response = await client.get(url, headers=HEADERS)
        assert response.status == 200 and (await response.json())["read_only"]
        calls = len(packet.calls)
        platform.robot_owners["base"] = "other"
        response = await client.get(url, headers=HEADERS)
        assert response.status == 409
        assert platform.robot_owners == {"base": "other"} and len(packet.calls) == calls
        platform.robot_owners.clear()
