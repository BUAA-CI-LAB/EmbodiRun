import json
import time

import pytest

pytest.importorskip("PIL")

from tests.xlerobot_owner.test_dualsense import FakeRobot, sample

from embodirun_xlerobot_owner.dualsense_drive import ZERO, BaseSession
from embodirun_xlerobot_owner.dualsense_recording import DemoRecording, input_record
from embodirun_xlerobot_owner.robot import DemoRobot


class Observer:
    def __init__(self):
        self.demo = DemoRobot()
        _, self.images = self.demo.read()

    def connect(self):
        pass

    def read(self):
        now = time.time_ns()
        return {
            "metadata": {"source": "physical"},
            "state": self.demo.state,
            "source_timestamp_ns": now,
            "state_timestamp_ns": now,
            "camera_timestamps_ns": dict.fromkeys(self.images, now),
            "raw": {"base_left_wheel": {"Present_Position": 2001}},
        }, self.images


def recorder(tmp_path, observer=None):
    return DemoRecording(
        tmp_path,
        "reference demo",
        "http://127.0.0.1:8766",
        "secret",
        {"source": "physical"},
        observer=observer or Observer(),
    )


def test_camera_raw_and_exact_command_feedback_are_saved_without_observer_control(tmp_path):
    recording = recorder(tmp_path)
    recording.wait_ready()
    robot = FakeRobot()
    session = BaseSession(robot, recording)
    session.enable()
    session.send({"x.vel": 0.02, "theta.vel": 0}, input_record(sample(["r1"])))
    session.stop()
    result = recording.close()
    assert result["frame_count"] >= 1
    frames = [json.loads(line) for line in (recording.path / "frames.jsonl").read_text().splitlines()]
    assert frames[0]["observation"]["raw"]["base_left_wheel"]["Present_Position"] == 2001
    assert frames[0]["action"] is None
    assert set(frames[0]["images"]) == {"front", "left_wrist", "right_wrist"}
    assert all((recording.path / p).is_file() for p in frames[0]["images"].values())
    events = [json.loads(line) for line in (recording.path / "control.jsonl").read_text().splitlines()]
    command = next(e for e in events if e["kind"] == "command_requested")
    ack = next(e for e in events if e["kind"] == "command_feedback")
    assert command["command_index"] == ack["command_index"] == 1
    assert command["action"] == ack["feedback"]["applied_action"]
    assert command["input_sample"]["buttons"] == ["r1"]
    assert events[-1]["kind"] == "release_result"
    assert robot.stops == 1
    assert "secret" not in (recording.path / "control.jsonl").read_text()


def test_missing_camera_fails_before_control_and_keeps_partial_record(tmp_path):
    observer = Observer()
    observer.images.pop("front")
    recording = recorder(tmp_path, observer)
    with pytest.raises(RuntimeError, match="观测不完整"):
        recording.wait_ready()
    with pytest.raises(RuntimeError):
        recording.close()
    assert (recording.path / "metadata.json").is_file()


def test_release_precedes_failed_logging(tmp_path):
    recording = recorder(tmp_path)
    recording.wait_ready()
    robot = FakeRobot()
    session = BaseSession(robot, recording)
    session.enable()
    recording.error = "disk failed"
    with pytest.raises(RuntimeError, match="disk failed"):
        session.send(ZERO)
    session.stop()
    assert robot.stops == 1 and not robot.actions
    with pytest.raises(RuntimeError):
        recording.close()


def test_no_camera_or_input_data_can_enable_robot_by_itself(tmp_path):
    recording = recorder(tmp_path)
    recording.wait_ready()
    robot = FakeRobot()
    session = BaseSession(robot, recording)
    recording.emit("input", input_sample=input_record(sample(["r1"])))
    session.stop()
    recording.close()
    assert not session.owns and not robot.actions and robot.stops == 0
