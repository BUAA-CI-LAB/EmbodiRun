import pytest
from tests.xlerobot_owner.test_motor_diagnostics import setup


def forbid_wheel_scalar_reads(bus, monkeypatch):
    original = bus.read

    def read(field, name, normalize=False):
        assert not name.startswith("base_"), "no wheel scalar fallback"
        return original(field, name, normalize)

    monkeypatch.setattr(bus, "read", read)


@pytest.mark.parametrize("velocity,moving", [(0, 0), (50, 1), (-50, 1), (0, 1), (50, 0)])
def test_observe_uses_one_packet_per_wheel_without_filtering_or_waiting(tmp_path, monkeypatch, velocity, moving):
    robot, buses, packet = setup(tmp_path)
    bus = buses["right"]
    forbid_wheel_scalar_reads(bus, monkeypatch)
    bus.values["base_left_wheel"].update(Present_Position=2544, Present_Velocity=velocity, Moving=moving)
    robot._active_scopes.add("base")

    def no_wait(*args):
        raise AssertionError("observation must never enter stationary waiting")

    monkeypatch.setattr(robot, "_wait_stopped_wheel", no_wait)
    try:
        observation, _ = robot.read()
        assert packet.calls == [(9, 56, 11), (10, 56, 11)]
        assert observation["raw"]["base_left_wheel"] == {
            "Present_Position": 2544,
            "Present_Velocity": velocity,
            "Moving": moving,
        }
        assert observation["raw_fields"] == observation["raw"]
        assert observation["state"]["x.vel"] == robot._wheel_raw_to_body(velocity, 0)["x.vel"]
        assert observation["control_state"]["base"] is True
        assert not observation["errors"] and not observation["state_cached"]
        assert observation["wheel_present_blocks"]["base_left_wheel"]["ok"]
        assert len(observation["wheel_present_blocks"]["base_left_wheel"]["raw_bytes"]) == 11
        assert all(not b.writes for b in buses.values())
    finally:
        robot._active_scopes.clear()
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
        (None, 0, 0),
        OSError("bad packet"),
    ],
)
def test_bad_observe_reply_stays_unknown_without_scalar_retry(tmp_path, monkeypatch, response):
    robot, buses, packet = setup(tmp_path)
    forbid_wheel_scalar_reads(buses["right"], monkeypatch)
    packet.overrides[56] = response
    try:
        observation, _ = robot.read()
        assert packet.calls == [(9, 56, 11), (10, 56, 11)]
        assert len(observation["errors"]) == 6
        assert "x.vel" not in observation["state"] and "theta.vel" not in observation["state"]
        for name in ("base_left_wheel", "base_right_wheel"):
            assert not observation["wheel_present_blocks"][name]["ok"]
            assert observation["wheel_present_blocks"][name]["fields"] == {}
            assert all("error" in value for value in observation["raw"][name].values())
        assert all(not b.writes for b in buses.values())
    finally:
        robot.close()


@pytest.mark.parametrize("failed", [False, True])
def test_observe_cache_retains_original_packets_errors_and_freshness(tmp_path, failed):
    robot, _, packet = setup(tmp_path)
    robot._read_interval_s = 10
    if failed:
        packet.overrides[56] = ([], -3002, 0)
    try:
        first, _ = robot.read()
        cached, _ = robot.read()
        assert len(packet.calls) == 2
        assert not first["state_cached"] and cached["state_cached"]
        assert cached["state_timestamp_ns"] == first["state_timestamp_ns"]
        assert cached["wheel_present_blocks"] == first["wheel_present_blocks"]
        assert cached["errors"] == first["errors"]
        first["wheel_present_blocks"].clear()
        again, _ = robot.read()
        assert len(again["wheel_present_blocks"]) == 2
        robot._read_interval_s = 0
        packet.overrides.clear()
        fresh, _ = robot.read()
        assert len(packet.calls) == 4
        assert not fresh["state_cached"] and not fresh["errors"]
        assert fresh["state_timestamp_ns"] > cached["state_timestamp_ns"]
        for name, block in fresh["wheel_present_blocks"].items():
            assert block["started_monotonic_ns"] > cached["wheel_present_blocks"][name]["started_monotonic_ns"]
    finally:
        robot.close()
