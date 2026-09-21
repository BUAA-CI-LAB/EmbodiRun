from types import SimpleNamespace

import pytest
from tests.xlerobot_owner.test_motor_diagnostics import setup
from tests.xlerobot_owner.test_stop_fault_handoff import prepared

from embodirun_xlerobot_owner import hardware


def clocked(robot, bus, monkeypatch, *, until=0.65, alternating=False):
    robot._watchdog_stop.set()
    robot._watchdog_thread.join(1)
    robot._stop_timeout_s, robot._stop_poll_s = 0.4, 0.02
    clock = SimpleNamespace(t=0.0)

    def monotonic():
        clock.t += 0.000001
        return clock.t

    def sleep(delay):
        clock.t += max(delay, 0.000001)

    monkeypatch.setattr(
        hardware, "time", SimpleNamespace(monotonic=monotonic, sleep=sleep, time_ns=lambda: int(clock.t * 1e9))
    )
    original = bus.packet_handler.readTxRx

    def read(port, motor_id, address, length):
        if address == 56:
            name = "base_left_wheel" if motor_id == 9 else "base_right_wheel"
            moving = (int(clock.t / 0.02) % 2) if alternating else int(clock.t < until)
            bus.values[name].update(Present_Velocity=50 * moving, Moving=moving)
        return original(port, motor_id, address, length)

    monkeypatch.setattr(bus.packet_handler, "readTxRx", read)
    return clock


@pytest.mark.parametrize(
    "reason,write,other,expected",
    [
        ("operator", True, False, True),
        ("operator_retry", True, False, True),
        ("watchdog", True, False, False),
        ("close", True, False, False),
        ("verified held restart", False, False, False),
        ("operator", True, True, False),
    ],
)
def test_single_packet_with_late_zeros_only_extends_explicit_idle_stop(
    tmp_path, monkeypatch, reason, write, other, expected
):
    robot, buses, packet = setup(tmp_path)
    clock = clocked(robot, buses["right"], monkeypatch)
    if other:
        robot._active_scopes.add("arms")
    try:
        result = robot._stop_locked(reason, write_commands=write, scope="base")
        assert result["stop_confirmed"] is expected
        assert (0.65 < clock.t < 0.75) if expected else (0.4 <= clock.t < 0.41)
        assert len(packet.calls) == 2 * len(result["samples"])
        assert ("arms" in robot._active_scopes) is other
        if expected:
            assert all(
                not m["Present_Velocity"] and not m["Moving"]
                for s in result["samples"][-2:]
                for m in s["motors"].values()
            )
    finally:
        robot._active_scopes.clear()
        robot.armed = False
        robot.close()


@pytest.mark.parametrize("mode", ["persistent", "alternating", "write_error"])
def test_extended_window_still_rejects_and_errors_do_not_extend(tmp_path, monkeypatch, mode):
    robot, buses, _ = setup(tmp_path)
    clock = clocked(robot, buses["right"], monkeypatch, until=10, alternating=mode == "alternating")
    if mode == "write_error":
        buses["right"].fail_field, buses["right"].fail_motor = "Goal_Velocity", "base_left_wheel"
    try:
        result = robot._stop_locked("operator_retry", scope="base")
        assert not result["stop_confirmed"] and robot.stop_unconfirmed
        assert (0.4 <= clock.t < 0.41) if mode == "write_error" else (1.5 <= clock.t < 1.51)
    finally:
        robot.close()


@pytest.mark.parametrize("bad_first", [False, True])
def test_fault_resume_plus_burst_plus_extended_window_requires_valid_consecutive_zeros(
    tmp_path, monkeypatch, bad_first
):
    robot, buses, _ = prepared(tmp_path, monkeypatch)
    robot.connect()
    packet = buses["right"].packet_handler
    clock = clocked(robot, buses["right"], monkeypatch)
    if bad_first:
        read = packet.readTxRx
        first = True

        def failed(*args):
            nonlocal first
            if first:
                first = False
                return [], -3002, 0
            return read(*args)

        monkeypatch.setattr(packet, "readTxRx", failed)
    try:
        result = robot.stop_all()
        assert result["stop_confirmed"] is not bad_first
        assert robot.stop_unconfirmed is bad_first
        assert result["stationary_confirmed"] and 0.65 < clock.t < 0.75
        assert len(result["writes"]) == 15
        assert all(w["status"] == "accepted" for w in result["writes"])
    finally:
        robot.close()
