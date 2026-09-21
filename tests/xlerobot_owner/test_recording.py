"""Recorder contract tests and fake-LeRobot adapter tests.

The export tests are deliberately named as adapter tests: they inject a small
``LeRobotDataset`` stand-in and do not claim that the installed LeRobot
package was exercised.  The dependency-error test covers the real-package
absence path.
"""

from __future__ import annotations

import io
import json
import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest

from embodirun_xlerobot_owner.recording import (
    CAMERA_ROLES,
    DEFAULT_JOINT_NAMES,
    EpisodeCorruptionError,
    EpisodeRecorder,
    LeRobotExportError,
    export_lerobot,
    validate_episode,
)

MINIMAL_JPEG = b"\xff\xd8minimal-test-jpeg\xff\xd9"


def _jpeg() -> bytes:
    try:
        from PIL import Image
    except ImportError:
        return MINIMAL_JPEG
    payload = io.BytesIO()
    Image.new("RGB", (4, 3), (40, 100, 180)).save(payload, format="JPEG", quality=90)
    return payload.getvalue()


def _metadata(*, source: str = "physical", collection_mode: str = "teleoperation", fps: float = 20.0) -> dict:
    return {
        "source": source,
        "collection_mode": collection_mode,
        "camera_roles_confirmed": True,
        "cameras": list(CAMERA_ROLES),
        "joint_names": list(DEFAULT_JOINT_NAMES),
        "joint_unit": "degrees",
        "gripper_unit": "range_0_100",
        "fps": fps,
        "max_skew_ms": 100,
        "max_gap_ms": 150,
    }


def _observation(timestamp_ns: int, *, base_velocity: dict | None = None) -> dict:
    value = {
        "state": {name: float(index) for index, name in enumerate(DEFAULT_JOINT_NAMES)},
        "source_timestamp_ns": timestamp_ns,
        "state_timestamp_ns": timestamp_ns,
        "camera_timestamps_ns": dict.fromkeys(CAMERA_ROLES, timestamp_ns),
        "received_timestamp_ns": timestamp_ns + 1,
    }
    if base_velocity is not None:
        value["base_velocity"] = base_velocity
    return value


def _action() -> dict:
    return {name: float(index + 10) for index, name in enumerate(DEFAULT_JOINT_NAMES)}


def _feedback(action: dict, timestamp_ns: int) -> dict:
    return {
        "applied_action": dict(action),
        "sent_timestamp_ns": timestamp_ns,
        # A command write is known; physical execution is intentionally still
        # unknown until measured by the subsequent observation.
        "physical_outcome": "unknown",
    }


def _append_frame(
    recorder: EpisodeRecorder,
    timestamp_ns: int,
    *,
    action: dict | None = None,
    feedback: dict | None = None,
    observation: dict | None = None,
) -> None:
    if observation is None:
        observation = _observation(timestamp_ns)
    recorder.append(
        observation=observation,
        action=action,
        images={role: _jpeg() for role in CAMERA_ROLES},
        input_sample={"pose": [1.0, 2.0, 3.0]} if action is not None else None,
        feedback=feedback,
    )


def _physical_episode(root: Path, *, fps: float = 20.0) -> Path:
    recorder = EpisodeRecorder(root, fps=fps)
    path = recorder.start("pick the object", _metadata(fps=fps))
    action = _action()
    _append_frame(recorder, 1_000_000_000, action=action, feedback=_feedback(action, 1_000_000_000))
    _append_frame(
        recorder,
        1_000_000_000 + round(1_000_000_000 / fps),
        action=action,
        feedback=_feedback(action, 1_000_000_000 + round(1_000_000_000 / fps)),
    )
    recorder.finish(success=True)
    return path


def _install_fake_lerobot(monkeypatch: pytest.MonkeyPatch):
    class FakeDataset:
        instances: ClassVar[list[FakeDataset]] = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.frames: list[dict] = []
            self.episodes: list[list[dict]] = []
            self.finalized = False
            type(self).instances.append(self)

        @classmethod
        def create(cls, **kwargs):
            return cls(**kwargs)

        def add_frame(self, frame):
            self.frames.append(frame)

        def save_episode(self):
            self.episodes.append(self.frames)
            self.frames = []

        def finalize(self):
            self.finalized = True

    root_module = types.ModuleType("lerobot")
    datasets_module = types.ModuleType("lerobot.datasets")
    dataset_module = types.ModuleType("lerobot.datasets.lerobot_dataset")
    dataset_module.LeRobotDataset = FakeDataset
    root_module.datasets = datasets_module
    datasets_module.lerobot_dataset = dataset_module
    monkeypatch.setitem(sys.modules, "lerobot", root_module)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets_module)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", dataset_module)
    return FakeDataset


@pytest.fixture
def export_dependencies() -> None:
    """Require the numeric/image adapters only for tests that execute export.

    The fake LeRobot module isolates the writer contract, but the production
    export path still converts frame arrays through NumPy and decodes JPEGs
    through Pillow.  Keep this requirement on the four tests that exercise
    that path so core-only pytest still covers recording validation and the
    missing-LeRobot error test below.
    """

    pytest.importorskip("numpy", reason="export adapter tests require NumPy")
    pytest.importorskip("PIL.Image", reason="export adapter tests require Pillow")


def test_recorder_is_lazy_and_writes_original_images_and_separate_input(tmp_path: Path) -> None:
    root = tmp_path / "episodes"
    recorder = EpisodeRecorder(root, fps=20.0)
    assert not root.exists()
    assert recorder.active is False
    assert recorder.path is None

    path = recorder.start(" pick object ", _metadata())
    action = _action()
    raw_images = {role: _jpeg() for role in CAMERA_ROLES}
    recorder.append(
        observation=_observation(1_000_000_000),
        action=action,
        images=raw_images,
        input_sample={"pose": {"x": 0.25}, "source": "quest"},
        feedback=_feedback(action, 1_000_000_000),
    )

    frame = json.loads((path / "frames.jsonl").read_text(encoding="utf-8"))
    assert frame["observation"]["state"] != frame["action"]
    assert frame["action"] == action
    assert frame["input_sample"]["pose"] == {"x": 0.25}
    assert frame["feedback"]["physical_outcome"] == "unknown"
    assert frame["source_timestamp_ns"] == 1_000_000_000
    for role, payload in raw_images.items():
        assert (path / frame["images"][role]).read_bytes() == payload

    result = recorder.finish(success=True)
    assert recorder.active is False
    assert recorder.path == path
    assert result["frame_count"] == 1
    assert (path / "metadata.json").is_file()
    assert (path / "results.json").is_file()


def test_start_rejects_blank_or_active_and_allocates_ordinary_unique_directories(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(tmp_path / "episodes")
    with pytest.raises(ValueError, match="non-blank"):
        recorder.start(" ", _metadata())
    first = recorder.start("one", _metadata())
    with pytest.raises(RuntimeError, match="already active"):
        recorder.start("two", _metadata())
    recorder.finish(reason="operator")
    second = recorder.start("two", _metadata())
    assert first != second
    recorder.force_interrupted_close()


def test_physical_command_receipt_unknown_outcome_is_trainable_when_state_is_measured(tmp_path: Path) -> None:
    path = _physical_episode(tmp_path / "episodes")
    report = validate_episode(path)
    assert report["source"] == "physical"
    assert report["frame_count"] == 2
    assert report["trainable"] is True
    assert report["errors"] == []


def test_gateway_timestamp_domain_is_retained_without_cross_clock_comparison(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(tmp_path / "episodes")
    path = recorder.start("remote clock", _metadata())
    action = _action()
    observation = _observation(1_000_000_000)
    observation.update(
        {
            "received_timestamp_ns": 900_000_000,
            "timestamp_domains": {
                "source": "robot",
                "state": "robot",
                "camera": "robot",
                "sent": "robot",
                "received": "gateway",
            },
        }
    )
    recorder.append(
        observation=observation,
        action=action,
        images={role: _jpeg() for role in CAMERA_ROLES},
        feedback=_feedback(action, 1_005_000_000),
    )
    recorder.finish(success=True)

    frame = json.loads((path / "frames.jsonl").read_text(encoding="utf-8"))
    assert frame["received_timestamp_ns"] == 900_000_000
    assert frame["observation"]["timestamp_domains"]["received"] == "gateway"
    report = validate_episode(path)
    assert report["trainable"] is True
    assert "timestamp_freshness" not in report["error_codes"]
    assert "timestamp_alignment" not in report["error_codes"]
    assert "timestamp_domains" in report["warning_codes"]


@pytest.mark.parametrize("mismatched_domain", ("state", "camera", "sent"))
def test_robot_timestamp_domain_mismatch_rejects_training_and_export(tmp_path: Path, mismatched_domain: str) -> None:
    recorder = EpisodeRecorder(tmp_path / "episodes")
    path = recorder.start("mismatched robot clock", _metadata())
    action = _action()
    observation = _observation(1_000_000_000)
    observation["timestamp_domains"] = {
        "source": "robot",
        "state": "robot",
        "camera": "robot",
        "sent": "robot",
        "received": "gateway",
    }
    observation["timestamp_domains"][mismatched_domain] = "different-robot-clock"
    recorder.append(
        observation=observation,
        action=action,
        images={role: _jpeg() for role in CAMERA_ROLES},
        feedback=_feedback(action, 1_000_000_000),
    )
    recorder.finish(success=True)

    report = validate_episode(path)
    assert report["timestamp_ok"] is False
    assert report["trainable"] is False
    assert "timestamp_alignment" in report["error_codes"]
    with pytest.raises(LeRobotExportError):
        export_lerobot([path], repo_id="user/mismatched-clock", output=tmp_path / "out", allow_demo=True)


def test_feedback_requires_a_nonempty_canonical_applied_action(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(tmp_path / "episodes")
    path = recorder.start("missing receipt", _metadata())
    _append_frame(
        recorder,
        1_000_000_000,
        action=_action(),
        feedback={"applied_action": {}, "sent_timestamp_ns": 1_000_000_000, "physical_outcome": "unknown"},
    )
    recorder.finish(success=False)
    report = validate_episode(path)
    assert report["trainable"] is False
    assert "applied_feedback" in report["error_codes"]


def test_read_only_or_synthetic_recording_is_retained_but_not_trainable(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(tmp_path / "episodes")
    path = recorder.start(
        "observe only",
        _metadata(source="synthetic", collection_mode="observation_only"),
    )
    _append_frame(recorder, 1_000_000_000)
    recorder.finish(success=None)
    report = validate_episode(path)
    assert report["frame_count"] == 1
    assert report["trainable"] is False
    assert any("not trainable" in error for error in report["errors"])
    assert (path / "results.json").is_file()


def test_resume_skips_only_incomplete_final_line_and_keeps_previous_frames(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(tmp_path / "episodes")
    path = recorder.start("recover", _metadata())
    action = _action()
    _append_frame(recorder, 1_000_000_000, action=action, feedback=_feedback(action, 1_000_000_000))
    with (path / "frames.jsonl").open("ab") as stream:
        stream.write(b'{"frame_index": 99')
    # A readable validation reports the complete prefix and explicitly warns
    # about the discarded final partial line.
    report = validate_episode(path)
    assert report["frame_count"] == 1
    assert "incomplete_final_line" in report["warning_codes"]
    recorder = EpisodeRecorder.resume(path)
    assert recorder.active
    assert recorder.root == tmp_path / "episodes"
    assert (path / "frames.jsonl").read_bytes().endswith(b"\n")
    _append_frame(
        recorder,
        1_050_000_000,
        action=action,
        feedback=_feedback(action, 1_050_000_000),
    )
    result = recorder.finish(success=False, reason="operator")
    assert result["frame_count"] == 2


def test_earlier_jsonl_corruption_is_never_hidden_or_resumed(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(tmp_path / "episodes")
    path = recorder.start("corrupt", _metadata())
    action = _action()
    _append_frame(recorder, 1_000_000_000, action=action, feedback=_feedback(action, 1_000_000_000))
    with (path / "frames.jsonl").open("ab") as stream:
        stream.write(b"not-json\n")
        stream.write(b"{}\n")
    report = validate_episode(path)
    assert report["frame_count"] == 1
    assert "json_corruption" in report["error_codes"]
    with pytest.raises(EpisodeCorruptionError, match="line 2"):
        EpisodeRecorder.resume(path)


def test_complete_invalid_final_line_is_corruption_not_a_recoverable_partial(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(tmp_path / "episodes")
    path = recorder.start("bad final", _metadata())
    action = _action()
    _append_frame(recorder, 1_000_000_000, action=action, feedback=_feedback(action, 1_000_000_000))
    with (path / "frames.jsonl").open("ab") as stream:
        stream.write(b"not-json")
    report = validate_episode(path)
    assert report["frame_count"] == 1
    assert "json_corruption" in report["error_codes"]
    assert "incomplete_final_line" not in report["warning_codes"]
    with pytest.raises(EpisodeCorruptionError, match="line 2"):
        EpisodeRecorder.resume(path)


def test_export_adapter_uses_official_writer_shape_and_preserves_canonical_order(
    tmp_path: Path, monkeypatch, export_dependencies
):
    FakeDataset = _install_fake_lerobot(monkeypatch)
    episode = _physical_episode(tmp_path / "episodes")
    output = tmp_path / "lerobot-output"
    result = export_lerobot([episode], repo_id="user/quest", output=output)

    assert result["adapter"].startswith("official LeRobotDataset")
    assert result["episode_count"] == 1
    assert result["frame_count"] == 2
    dataset = FakeDataset.instances[-1]
    assert dataset.finalized is True
    assert dataset.kwargs["repo_id"] == "user/quest"
    assert dataset.kwargs["use_videos"] is False
    assert set(dataset.kwargs["features"]) == {
        "observation.state",
        "action",
        "observation.images.front",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    }
    frames = dataset.episodes[0]
    assert frames[0]["task"] == "pick the object"
    assert frames[0]["observation.state"].dtype.name == "float32"
    assert frames[0]["action"].tolist() == [float(index + 10) for index in range(12)]
    assert frames[0]["observation.images.front"].shape == (3, 4, 3)


def test_export_adapter_resamples_irregular_or_mismatched_source_with_configured_max_gap(
    tmp_path: Path, monkeypatch, export_dependencies
):
    FakeDataset = _install_fake_lerobot(monkeypatch)
    recorder = EpisodeRecorder(tmp_path / "episodes", fps=10)
    path = recorder.start("resample", _metadata(fps=10))
    action = _action()
    _append_frame(recorder, 1_000_000_000, action=action, feedback=_feedback(action, 1_000_000_000))
    _append_frame(recorder, 1_100_000_000, action=action, feedback=_feedback(action, 1_100_000_000))
    recorder.finish(success=True)

    result = export_lerobot([path], repo_id="user/resample", output=tmp_path / "out", fps=20)
    assert result["frame_count"] == 3
    assert result["resampling"][0]["mode"] == "nearest_previous_valid"
    assert result["resampling"][0]["max_observation_age_ms"] == 50.0
    assert len(FakeDataset.instances[-1].episodes[0]) == 3


def test_export_resampling_preserves_epoch_sized_integer_origin(tmp_path: Path, monkeypatch, export_dependencies):
    FakeDataset = _install_fake_lerobot(monkeypatch)
    recorder = EpisodeRecorder(tmp_path / "episodes", fps=15)
    path = recorder.start("epoch resample", _metadata(fps=15))
    action = _action()
    start_ns = 1_789_549_556_865_642_689
    _append_frame(recorder, start_ns, action=action, feedback=_feedback(action, start_ns))
    _append_frame(
        recorder,
        start_ns + 66_666_667,
        action=action,
        feedback=_feedback(action, start_ns + 66_666_667),
    )
    recorder.finish(success=True)

    result = export_lerobot([path], repo_id="user/epoch", output=tmp_path / "out", fps=15)
    assert result["frame_count"] == 2
    assert len(FakeDataset.instances[-1].episodes[0]) == 2


def test_export_rejects_nonzero_or_missing_enabled_mobile_base_evidence(
    tmp_path: Path, monkeypatch, export_dependencies
):
    _install_fake_lerobot(monkeypatch)
    recorder = EpisodeRecorder(tmp_path / "episodes")
    path = recorder.start("moving base", {**_metadata(), "enable_base": True})
    action = _action()
    _append_frame(
        recorder,
        1_000_000_000,
        action=action,
        feedback=_feedback(action, 1_000_000_000),
        observation=_observation(1_000_000_000),
    )
    recorder.finish(success=True)
    with pytest.raises(LeRobotExportError, match="base"):
        export_lerobot([path], repo_id="user/moving", output=tmp_path / "out")

    recorder = EpisodeRecorder(tmp_path / "episodes")
    path = recorder.start("moving base", _metadata())
    _append_frame(
        recorder,
        1_000_000_000,
        action=action,
        feedback=_feedback(action, 1_000_000_000),
        observation=_observation(
            1_000_000_000,
            base_velocity={"vx_m_s": 0.1, "vy_m_s": 0.0, "yaw_rate_rad_s": 0.0},
        ),
    )
    recorder.finish(success=True)
    with pytest.raises(LeRobotExportError, match="nonzero mobile-base"):
        export_lerobot([path], repo_id="user/moving-2", output=tmp_path / "out-2", allow_demo=True)


def test_export_refuses_existing_output_and_missing_real_lerobot_dependency(tmp_path: Path, monkeypatch):
    episode = _physical_episode(tmp_path / "episodes")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="append"):
        export_lerobot([episode], repo_id="user/existing", output=existing)

    import embodirun_xlerobot_owner.recording as recording_module

    original_import = recording_module.importlib.import_module

    def missing_lerobot(name: str):
        if name == "lerobot.datasets.lerobot_dataset":
            raise ModuleNotFoundError("lerobot is intentionally unavailable")
        return original_import(name)

    monkeypatch.setattr(recording_module.importlib, "import_module", missing_lerobot)
    with pytest.raises(LeRobotExportError, match="install.*lerobot"):
        export_lerobot([episode], repo_id="user/no-lerobot", output=tmp_path / "missing")
