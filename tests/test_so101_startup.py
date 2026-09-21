import pytest

from embodirun.robots.lerobot.so101.adapter import (
    _MOTOR_IDS,
    SO101AdapterError,
    _FeetechSO101Controller,
    _MotorCalibration,
)
from embodirun.robots.lerobot.so101.config import SO101Config


def controller(failure=None, present=2000):
    c = object.__new__(_FeetechSO101Controller)
    c.config = SO101Config(port="/dev/fake")
    c.calibration = {name: _MotorCalibration(i, 0, 0, 1000, 3000) for name, i in _MOTOR_IDS.items()}
    c.torque = dict.fromkeys(range(1, 7), 0)
    c.writes = []
    c._read_register = lambda motor, register, **kw: present if register[0] == 56 else 0
    failed = False

    def write(motor, register, value, **kwargs):
        nonlocal failed
        c.writes.append((motor, register[0], value))
        if not failed and failure == (motor, register[0], value):
            failed = True
            raise SO101AdapterError("injected write failure")
        if register[0] == 42:
            c.torque[motor] = 1  # Account for the partial activation observed after goal writes.
        if register[0] == 40:
            c.torque[motor] = value

    c._write_register = write
    return c


def test_goal_write_failure_disables_every_motor():
    c = controller(failure=(3, 42, 2000))
    with pytest.raises(SO101AdapterError, match="injected"):
        c._configure()
    assert c.torque == dict.fromkeys(range(1, 7), 0)
    assert (6, 40, 0) in c.writes[-12:]


def test_partial_enable_failure_disables_every_motor():
    c = controller(failure=(4, 40, 1))
    with pytest.raises(SO101AdapterError, match="injected"):
        c._configure()
    assert c.torque == dict.fromkeys(range(1, 7), 0)


def test_small_encoder_overshoot_uses_valid_bounded_goal():
    c = controller(present=3004)
    c._configure()
    assert [(m, v) for m, r, v in c.writes if r == 42] == [(m, 3000) for m in range(1, 7)]


def test_large_overshoot_sends_no_position_goals():
    c = controller(present=3500)
    with pytest.raises(SO101AdapterError, match="configured step limit"):
        c._configure()
    assert not any(r == 42 for m, r, v in c.writes)
    assert c.torque == dict.fromkeys(range(1, 7), 0)


def test_shutdown_attempts_remaining_motors_after_one_failure():
    c = controller(failure=(1, 40, 0))
    c.torque = dict.fromkeys(range(1, 7), 1)
    with pytest.raises(SO101AdapterError, match="shutdown incomplete"):
        c._set_torque(False)
    assert all(c.torque[i] == 0 for i in range(2, 7))
    assert (6, 55, 0) in c.writes


def test_initial_goals_precede_torque_enable():
    c = controller()
    c._configure()
    last_goal = max(i for i, (_, register, _) in enumerate(c.writes) if register == 42)
    first_enable = next(i for i, (_, register, value) in enumerate(c.writes) if register == 40 and value == 1)
    assert last_goal < first_enable


def test_late_invalid_joint_prevents_all_goal_writes():
    c = controller()
    c._read_register = lambda motor, register, **kw: (3500 if motor == 5 else 2000) if register[0] == 56 else 0
    with pytest.raises(SO101AdapterError, match="configured step limit"):
        c._configure()
    assert not any(register == 42 for _, register, _ in c.writes)


def test_cleanup_failure_preserves_startup_error():
    c = controller()
    failed = False
    original = c._write_register

    def write(motor, register, value, **kwargs):
        nonlocal failed
        if register[0] == 42:
            failed = True
            raise SO101AdapterError("original goal write failure")
        if failed and motor == 1 and register[0] == 40:
            raise SO101AdapterError("disable failed")
        original(motor, register, value, **kwargs)

    c._write_register = write
    with pytest.raises(SO101AdapterError, match="original goal write failure") as caught:
        c._configure()
    assert "torque shutdown incomplete" in str(caught.value.__cause__)
    assert (6, 55, 0) in c.writes
