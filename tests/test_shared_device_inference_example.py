from __future__ import annotations

import json

import pytest
from examples.run_shared_device_inference import (
    FRONT_NAME,
    MODEL_UNITS,
    POLICY_ACTION_SPACE,
    POLICY_FEATURE_NAMES,
    WRIST_NAME,
    ExperimentError,
    RecordedReplayCameraSource,
    ReplayInputError,
    build_experiment_config,
    load_replay_records,
    native_state_from_record,
    validate_capabilities,
    validate_model_units,
)


def _write_manifest(tmp_path):
    front = tmp_path / "main.jpg"
    wrist = tmp_path / "wrist.jpg"
    front.write_bytes(b"\xff\xd8\xfffront")
    wrist.write_bytes(b"\xff\xd8\xffwrist")
    manifest = tmp_path / "frames.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "index": 12,
                "observation.images.camera2": front.name,
                "observation.images.camera0": wrist.name,
                "captured_timestamp_ns": 123456,
                "clock_domain": "dataset_clock",
                "present": {"state_native": [1, 2, 3, 4, 5, 6]},
                "action": {"state_native": [6, 5, 4, 3, 2, 1]},
                "timestamp": "recorded-wall-clock",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest, front, wrist


def test_recorded_pair_preserves_provenance_without_claiming_live_capture(tmp_path):
    manifest, front, wrist = _write_manifest(tmp_path)

    records = load_replay_records(manifest)
    assert len(records) == 1
    record = records[0]
    assert record.sequence == 12
    assert record.front_path == front.resolve()
    assert record.wrist_path == wrist.resolve()
    assert record.captured_timestamp_ns == 123456
    assert record.original_clock_domain == "dataset_clock"
    assert record.measured_state == {"state_native": [1, 2, 3, 4, 5, 6]}
    assert record.commanded_action == {"state_native": [6, 5, 4, 3, 2, 1]}
    assert record.measured_state != record.commanded_action

    source = RecordedReplayCameraSource(records)
    frames = source.capture()
    assert [frame.name for frame in frames] == [FRONT_NAME, WRIST_NAME]
    assert [frame.data for frame in frames] == [front.read_bytes(), wrist.read_bytes()]
    for frame in frames:
        assert frame.captured_timestamp_ns == frame.received_timestamp_ns
        assert frame.received_timestamp_ns is not None
        assert frame.clock_domain == "host_monotonic_ns"
        assert frame.profile["replay"] is True
        assert frame.profile["hardware_access"] is False
        assert frame.profile["original_capture_timestamp_ns"] == 123456
        assert frame.profile["original_clock_domain"] == "dataset_clock"


def test_nested_frame_manifest_and_explicit_keys_are_supported(tmp_path):
    front = tmp_path / "front.png"
    wrist = tmp_path / "wrist.png"
    front.write_bytes(b"\x89PNG\r\n\x1a\nfront")
    wrist.write_bytes(b"\x89PNG\r\n\x1a\nwrist")
    manifest = tmp_path / "frames.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "frames": [
                    {"name": "front_stream", "path": front.name},
                    {"name": "wrist_stream", "path": wrist.name},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_replay_records(
        manifest,
        front_key="front_stream",
        wrist_key="wrist_stream",
    )
    assert records[0].front_path == front.resolve()
    assert records[0].wrist_path == wrist.resolve()


def test_record_line_selects_camera_index_map_and_measured_present(tmp_path):
    front = tmp_path / "main.jpg"
    wrist = tmp_path / "wrist.jpg"
    front.write_bytes(b"\xff\xd8\xfffront")
    wrist.write_bytes(b"\xff\xd8\xffwrist")
    manifest = tmp_path / "frames.jsonl"
    lines = [
        {"sequence": 99, "cameras": {"2": "missing.jpg", "0": "missing.jpg"}},
        {
            "sequence": 100,
            "cameras": {"2": front.name, "0": wrist.name},
            "present": {
                "shoulder_pan": 1.0,
                "shoulder_lift": 2.0,
                "elbow_flex": 3.0,
                "wrist_flex": 4.0,
                "wrist_roll": 5.0,
                "gripper": 6.0,
            },
            "action": {"shoulder_pan.pos": 10.0},
        },
    ]
    manifest.write_text("\n".join(json.dumps(item) for item in lines) + "\n", encoding="utf-8")

    records = load_replay_records(manifest, record_line=2)
    assert len(records) == 1
    assert records[0].sequence == 100
    assert native_state_from_record(records[0]) == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


def test_action_only_image_fields_are_not_accepted_as_observations(tmp_path):
    action_image = tmp_path / "action.jpg"
    action_image.write_bytes(b"\xff\xd8\xffaction")
    manifest = tmp_path / "frames.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "action": {
                    "observation.images.camera2": action_image.name,
                    "observation.images.camera0": action_image.name,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ReplayInputError, match="no front/wrist image pair"):
        load_replay_records(manifest)


def test_capability_check_is_exact_and_does_not_accept_ee_or_normalized_schema():
    capabilities = {
        "adapter": {
            "action_space": POLICY_ACTION_SPACE,
            "state_fields": ["state_native"],
            "action_feature_names": list(POLICY_FEATURE_NAMES),
            "return_steps": 50,
        }
    }
    assert validate_capabilities(capabilities)["units"] == MODEL_UNITS

    normalized = dict(capabilities)
    normalized["adapter"] = dict(capabilities["adapter"])
    normalized["adapter"]["state_fields"] = ["joint_positions_normalized"]
    with pytest.raises(ExperimentError, match="state_fields"):
        validate_capabilities(normalized)

    libero = dict(capabilities)
    libero["adapter"] = dict(capabilities["adapter"])
    libero["adapter"]["action_feature_names"] = ["x", "y", "z", "roll", "pitch", "yaw"]
    with pytest.raises(ExperimentError, match="action_feature_names"):
        validate_capabilities(libero)


def test_model_units_are_explicit_and_config_is_fake_only(tmp_path):
    assert validate_model_units(MODEL_UNITS) == MODEL_UNITS
    with pytest.raises(ExperimentError, match="no degree or normalized conversion"):
        validate_model_units("degrees")

    record = load_replay_records(_write_manifest(tmp_path)[0])[0]
    assert native_state_from_record(record) == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    config = build_experiment_config(
        "http://127.0.0.1:18885",
        port=18901,
        initial_state_native=native_state_from_record(record),
    )
    assert config.binding_kind == "simulated.policy_vector.pi05"
    assert config.robot_kind == "simulated.policy_vector"
    assert config.inputs[0].kind == "recorded_replay"
    assert config.robot_options["initial_state_native"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
