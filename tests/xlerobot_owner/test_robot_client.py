import io
import json

import pytest

from embodirun_xlerobot_owner.robot import RemoteRobot


class RecordingOpener:
    def __init__(self, *, fail: Exception | None = None):
        self.fail = fail
        self.calls: list[tuple[str, float]] = []

    def open(self, request, *, timeout):
        endpoint = request.full_url.rsplit("/", 1)[-1]
        self.calls.append((endpoint, timeout))
        if self.fail is not None:
            raise self.fail
        payload = {"armed": True} if endpoint == "arm" else {"stop_confirmed": True}
        return io.BytesIO(json.dumps(payload).encode())


def test_control_transitions_get_long_timeout_without_slowing_regular_requests():
    robot = RemoteRobot("http://127.0.0.1:8766", "test-token", timeout=0.25)
    opener = RecordingOpener()
    robot.opener = opener

    robot._request("status")
    robot.arm()
    robot.stop()
    robot.stop_all()

    assert opener.calls == [
        ("status", 0.25),
        ("arm", 15.0),
        ("stop", 15.0),
        ("stop_all", 15.0),
    ]


def test_timeout_identifies_the_agx_operation_and_effective_limit():
    robot = RemoteRobot("http://127.0.0.1:8766", "test-token", timeout=0.25)
    robot.opener = RecordingOpener(fail=TimeoutError("timed out"))

    with pytest.raises(TimeoutError, match=r"AGX arm timed out after 15s"):
        robot.arm()
