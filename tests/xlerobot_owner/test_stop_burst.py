import pytest
from tests.xlerobot_owner.test_motor_diagnostics import setup
from tests.xlerobot_owner.test_wheel_observe_block import forbid_wheel_scalar_reads


def test_stop_one_packet_per_wheel_two_zeros_and_unchanged_write_set(tmp_path, monkeypatch):
    robot, buses, packet = setup(tmp_path)
    forbid_wheel_scalar_reads(buses["right"], monkeypatch)
    try:
        result = robot._stop_locked("operator_retry", scope="all")
        assert result["stop_confirmed"] and len(result["samples"]) == 2
        assert packet.calls == [(9, 56, 11), (10, 56, 11)] * 2
        assert len(result["writes"]) == 14
        for write in result["writes"]:
            assert write["status"] == "accepted"
            assert write["field"] == ("Goal_Velocity" if write["motor"].startswith("base_") else "Goal_Position")
        for sample in result["samples"]:
            assert set(sample["wheel_present_blocks"]) == {"base_left_wheel", "base_right_wheel"}
            assert sample["wheel_present_blocks"]["base_left_wheel"]["ok"]
    finally:
        robot.close()


@pytest.mark.parametrize(
    "reply",
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
        OSError("checksum failure"),
    ],
)
def test_one_bad_packet_cannot_be_erased_by_later_two_zero_samples(tmp_path, monkeypatch, reply):
    robot, buses, packet = setup(tmp_path)
    forbid_wheel_scalar_reads(buses["right"], monkeypatch)
    read, first = packet.readTxRx, True

    def bad_first(*args):
        nonlocal first
        if first:
            first = False
            if isinstance(reply, Exception):
                raise reply
            return reply
        return read(*args)

    monkeypatch.setattr(packet, "readTxRx", bad_first)
    try:
        result = robot._stop_locked("operator_retry", scope="base")
        assert len(result["samples"]) == 3 and result["stationary_confirmed"]
        assert not result["stop_confirmed"] and not result["feedback_available"]
        assert robot.stop_unconfirmed
        first_sample = result["samples"][0]
        assert not first_sample["wheel_present_blocks"]["base_left_wheel"]["ok"]
        assert first_sample["motors"]["base_left_wheel"]["Present_Velocity"] is None
    finally:
        robot.close()


@pytest.mark.parametrize("velocity,moving", [(50, 0), (0, 1), (-50, 1), (50, 1)])
def test_nonzero_pairs_never_pass_and_keep_point_four_deadline(tmp_path, monkeypatch, velocity, moving):
    robot, buses, packet = setup(tmp_path)
    robot._stop_timeout_s, robot._stop_poll_s = 0.4, 0.02
    # Fake clock measures the configured polling budget, not real SDK latency.
    clock = [0.0]
    monkeypatch.setattr("embodirun_xlerobot_owner.hardware.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "embodirun_xlerobot_owner.hardware.time.sleep", lambda dt: clock.__setitem__(0, clock[0] + max(dt, 0.000001))
    )
    buses["right"].values["base_left_wheel"].update(Present_Velocity=velocity, Moving=moving)
    forbid_wheel_scalar_reads(buses["right"], monkeypatch)
    try:
        result = robot._stop_locked("watchdog", scope="base")
        assert not result["stop_confirmed"] and 0.4 <= clock[0] < 0.421
        assert result["feedback_available"] and len(result["writes"]) == 2
        assert len(packet.calls) == 2 * len(result["samples"])
        assert all(s["motors"]["base_left_wheel"]["Present_Velocity"] == velocity for s in result["samples"])
    finally:
        robot.close()


def test_scoped_success_does_not_clear_prior_whole_fault(tmp_path):
    robot, _, _ = setup(tmp_path)
    try:
        robot._stop_uncertain = True
        assert robot._stop_locked("operator_retry", scope="base")["stop_confirmed"]
        assert robot.stop_unconfirmed
        assert robot._stop_locked("operator_retry", scope="all")["stop_confirmed"]
        assert not robot.stop_unconfirmed  # non-inherited behavior unchanged
    finally:
        robot.close()
