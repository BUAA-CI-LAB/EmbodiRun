from tests.xlerobot_owner.test_motor_diagnostics import setup


def _sequence_reader(packet, bus, sequence):
    """Return a read hook whose first wheel block is the preflight sample."""

    original = packet.readTxRx
    seen = []

    def read(port, motor_id, address, length):
        if motor_id == 9 and address == 56:
            sample = sequence[min(len(seen), len(sequence) - 1)]
            seen.append(sample)
            if sample is None:
                return [], -3002, 0
            bus.values["base_left_wheel"].update(Present_Velocity=sample[0], Moving=sample[1])
        return original(port, motor_id, address, length)

    return read, seen


def test_cold_torque_off_transient_velocity_waits_for_three_strict_zeros(tmp_path, monkeypatch):
    robot, buses, packet = setup(tmp_path)
    bus = buses["right"]
    read, seen = _sequence_reader(packet, bus, [(50, 1), (0, 0), (0, 0), (0, 0)])
    monkeypatch.setattr(packet, "readTxRx", read)
    monkeypatch.setattr("embodirun_xlerobot_owner.hardware.time.sleep", lambda _: None)
    try:
        _, records, errors = robot._preflight(scope="base")
        trace = records["base_left_wheel"]["stationary_wait"]
        assert not errors
        assert trace["settled"] is True
        assert len(seen) == 4
        assert bus.values["base_left_wheel"]["Torque_Enable"] == 0
        assert all(not item.writes for item in buses.values())
    finally:
        robot.close()


def test_cold_torque_off_persistent_motion_is_rejected_without_writes(tmp_path, monkeypatch):
    robot, buses, packet = setup(tmp_path)
    bus = buses["right"]
    read, seen = _sequence_reader(packet, bus, [(50, 1)] * 41)
    monkeypatch.setattr(packet, "readTxRx", read)
    monkeypatch.setattr("embodirun_xlerobot_owner.hardware.time.sleep", lambda _: None)
    try:
        _, records, errors = robot._preflight(scope="base")
        trace = records["base_left_wheel"]["stationary_wait"]
        assert trace["settled"] is False
        assert len(seen) == 41
        assert any("three consecutive strictly stationary samples" in error for error in errors)
        assert all(not item.writes for item in buses.values())
    finally:
        robot.close()


def test_cold_wheel_bad_wait_reply_is_unknown_and_rejected(tmp_path, monkeypatch):
    robot, buses, packet = setup(tmp_path)
    bus = buses["right"]
    read, seen = _sequence_reader(packet, bus, [(50, 1), None])
    monkeypatch.setattr(packet, "readTxRx", read)
    monkeypatch.setattr("embodirun_xlerobot_owner.hardware.time.sleep", lambda _: None)
    try:
        _, records, errors = robot._preflight(scope="base")
        trace = records["base_left_wheel"]["stationary_wait"]
        assert trace["settled"] is False
        assert len(seen) == 2
        assert any("stationary wait transport/device failure" in error for error in errors)
        assert all(not item.writes for item in buses.values())
    finally:
        robot.close()


def test_cold_wheel_nonzero_goal_does_not_enter_stationary_wait(tmp_path, monkeypatch):
    robot, buses, packet = setup(tmp_path)
    bus = buses["right"]
    bus.values["base_left_wheel"]["Goal_Velocity"] = 1
    read, seen = _sequence_reader(packet, bus, [(50, 1), (0, 0), (0, 0), (0, 0)])
    monkeypatch.setattr(packet, "readTxRx", read)
    monkeypatch.setattr("embodirun_xlerobot_owner.hardware.time.sleep", lambda _: None)
    try:
        _, records, errors = robot._preflight(scope="base")
        assert "stationary_wait" not in records["base_left_wheel"]
        assert len(seen) == 1
        assert any("motor is moving during preflight" in error for error in errors)
        assert all(not item.writes for item in buses.values())
    finally:
        robot.close()
