"""Unit tests for the wired SO-101 teleoperation integration.

Nothing here needs an arm, a camera or a socket: the layout logic, the framing,
the safety clamp and the episode bookkeeping are all separable from hardware.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from embodirun_so101_wired_teleop.config import (
    ConfigError,
    load_config,
    parse_config,
)
from embodirun_so101_wired_teleop.follower import clamp_target
from embodirun_so101_wired_teleop.protocol import JOINTS, MAGIC, PACKET, ProtocolError, decode, encode, is_newer
from embodirun_so101_wired_teleop.recorder import EpisodeRecorder, role_names

EXAMPLE = Path(__file__).parents[2] / "integrations" / "so101_wired_teleop" / "config.example.yaml"


def _leader(identifier: str, port: int = 55101) -> dict:
    return {
        "id": identifier,
        "serial": f"/dev/serial/by-id/{identifier}",
        "arm_id": identifier,
        "calibration_dir": "/calibration",
        "advertise": "10.0.0.1",
        "port": port,
    }


def _follower(identifier: str, leader: str, bind: str = "10.0.0.2") -> dict:
    return {
        "id": identifier,
        "leader": leader,
        "bind": bind,
        "serial": f"/dev/serial/by-id/{identifier}",
        "arm_id": identifier,
        "calibration_dir": "/calibration",
    }


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #


def test_one_leader_can_drive_every_follower() -> None:
    """Pointing every follower at one leader is all 'one leader, many arms' means."""

    config = parse_config(
        {
            "leaders": [_leader("only")],
            "followers": [
                _follower("a", "only", "10.0.0.11"),
                _follower("b", "only", "10.0.0.12"),
                _follower("c", "only", "10.0.0.13"),
            ],
        }
    )

    assert config.destinations("only") == ("10.0.0.11", "10.0.0.12", "10.0.0.13")
    assert len(config.followers_of("only")) == 3


def test_leaders_split_the_followers_between_them() -> None:
    config = parse_config(
        {
            "leaders": [_leader("left", 55101), _leader("right", 55102)],
            "followers": [
                _follower("a", "left", "10.0.0.11"),
                _follower("b", "left", "10.0.0.12"),
                _follower("c", "right", "10.0.0.13"),
            ],
        }
    )

    assert config.destinations("left") == ("10.0.0.11", "10.0.0.12")
    assert config.destinations("right") == ("10.0.0.13",)


def test_a_leader_without_followers_is_valid_and_drives_nothing() -> None:
    config = parse_config(
        {
            "leaders": [_leader("used"), _leader("idle", 55102)],
            "followers": [_follower("a", "used")],
        }
    )

    assert config.destinations("idle") == ()
    assert config.followers_of("idle") == ()


def test_moving_a_follower_between_leaders_is_only_a_configuration_change() -> None:
    shared = {
        "leaders": [_leader("left", 55101), _leader("right", 55102)],
        "followers": [
            _follower("a", "left", "10.0.0.11"),
            _follower("b", "right", "10.0.0.12"),
        ],
    }
    before = parse_config(shared)
    assert before.destinations("left") == ("10.0.0.11",)
    assert before.destinations("right") == ("10.0.0.12",)

    shared["followers"][1]["leader"] = "left"
    after = parse_config(shared)
    assert after.destinations("left") == ("10.0.0.11", "10.0.0.12")
    assert after.destinations("right") == ()


def test_follower_must_name_a_configured_leader() -> None:
    with pytest.raises(ConfigError, match="unknown leader"):
        parse_config({"leaders": [_leader("known")], "followers": [_follower("a", "missing")]})


def test_leaders_may_not_share_a_port() -> None:
    """The port is the group key, so two leaders on one port would interleave."""

    with pytest.raises(ConfigError, match="share port"):
        parse_config(
            {
                "leaders": [_leader("left", 55101), _leader("right", 55101)],
                "followers": [],
            }
        )


def test_duplicate_identifiers_are_rejected() -> None:
    with pytest.raises(ConfigError, match="duplicate id"):
        parse_config({"leaders": [_leader("same"), _leader("same", 55102)]})


def test_camera_roles_must_match_cameras_one_for_one() -> None:
    follower = _follower("a", "l")
    follower.update({"cameras": [0, 2], "camera_roles": ["main"]})
    with pytest.raises(ConfigError, match="camera_roles"):
        parse_config({"leaders": [_leader("l")], "followers": [follower]})


def test_safety_limits_must_be_positive() -> None:
    follower = _follower("a", "l")
    follower["max_step"] = 0
    with pytest.raises(ConfigError, match="max_step"):
        parse_config({"leaders": [_leader("l")], "followers": [follower]})


# --------------------------------------------------------------------------- #
# placeholders
# --------------------------------------------------------------------------- #


def test_example_layout_is_pure_placeholders_and_refuses_to_run() -> None:
    config = load_config(EXAMPLE)

    missing = config.unresolved()
    assert missing, "the shipped example must not contain real deployment values"
    with pytest.raises(ConfigError, match="placeholders"):
        config.require_resolved()


def test_a_resolved_layout_passes_the_placeholder_check() -> None:
    config = parse_config({"leaders": [_leader("l")], "followers": [_follower("a", "l")]})

    assert config.unresolved() == ()
    config.require_resolved()


def test_the_example_parses_into_the_two_leaders_two_each_layout() -> None:
    config = load_config(EXAMPLE)

    assert [leader.id for leader in config.leaders] == ["leader_a", "leader_b"]
    assert [f.leader for f in config.followers] == ["leader_a", "leader_a", "leader_b"]
    assert config.leaders[0].port != config.leaders[1].port


# --------------------------------------------------------------------------- #
# framing
# --------------------------------------------------------------------------- #


def test_packets_round_trip_every_joint() -> None:
    action = {name: float(index) for index, name in enumerate(JOINTS)}

    sequence, restored = decode(encode(7, action))

    assert sequence == 7
    assert restored == action


def test_packets_reject_a_short_or_foreign_datagram() -> None:
    with pytest.raises(ProtocolError, match="wrong packet size"):
        decode(b"short")
    with pytest.raises(ProtocolError, match="wrong packet magic"):
        decode(b"XXXX" + encode(0, dict.fromkeys(JOINTS, 0.0))[4:])


def test_encode_reports_a_missing_joint() -> None:
    with pytest.raises(ProtocolError, match="missing joint"):
        encode(0, {"shoulder_pan.pos": 0.0})


def test_network_probe_cannot_be_decoded_as_an_arm_command() -> None:
    packet = encode(0, dict.fromkeys(JOINTS, 0.0), probe=True)
    assert decode(packet, probe=True)[0] == 0
    with pytest.raises(ProtocolError, match="magic"):
        decode(packet)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_targets_and_safety_limits_are_rejected(value) -> None:
    with pytest.raises(ProtocolError, match="finite"):
        encode(0, dict.fromkeys(JOINTS, value))
    with pytest.raises(ProtocolError, match="finite"):
        decode(PACKET.pack(MAGIC, 0, *([value] * 6)))
    follower = _follower("a", "l")
    follower["max_step"] = value
    with pytest.raises(ConfigError, match="max_step"):
        parse_config({"leaders": [_leader("l")], "followers": [follower]})


def test_sequence_order_rejects_stale_packets_and_allows_wraparound() -> None:
    assert is_newer(0, -1)
    assert is_newer(0, 0xFFFFFFFF)
    assert is_newer(12, 10)
    assert not is_newer(10, 10)
    assert not is_newer(9, 10)
    assert not is_newer(0xFFFFFFFF, 0)


def test_console_only_selects_the_requested_follower(monkeypatch) -> None:
    from embodirun_so101_wired_teleop.cli import _Endpoint

    config = parse_config({"leaders": [_leader("l")], "followers": [_follower("a", "l")]})
    endpoint = _Endpoint(config.followers[0])
    monkeypatch.setattr(
        endpoint,
        "_shell",
        lambda _: (
            "11 /bin/python /bin/embodirun-so101-follower --follower a --config layout.yaml\n"
            "12 /bin/embodirun-so101-follower --follower ab --config layout.yaml\n"
            "13 /bin/embodirun-so101-follower --follower=b --config layout.yaml\n"
        ),
    )
    assert endpoint.processes() == [11]


# --------------------------------------------------------------------------- #
# safety clamp
# --------------------------------------------------------------------------- #


def test_the_target_advances_by_at_most_max_step_per_cycle() -> None:
    desired = {"elbow_flex.pos": 90.0}
    present = {"elbow_flex": 0.0}
    commanded = {"elbow_flex.pos": 0.0}

    step = clamp_target(desired, present, commanded, max_step=3.0, max_lead=100.0)

    assert step["elbow_flex.pos"] == pytest.approx(3.0)


def test_the_target_may_not_lead_the_measured_position() -> None:
    """A stale or jumped target must not pull the arm further than max_lead."""

    desired = {"elbow_flex.pos": 90.0}
    present = {"elbow_flex": 10.0}
    commanded = {"elbow_flex.pos": 80.0}

    step = clamp_target(desired, present, commanded, max_step=30.0, max_lead=15.0)

    assert step["elbow_flex.pos"] == pytest.approx(25.0)


def test_the_target_never_moves_backwards_faster_than_max_step() -> None:
    desired = {"elbow_flex.pos": -90.0}
    present = {"elbow_flex": 0.0}
    commanded = {"elbow_flex.pos": 0.0}

    step = clamp_target(desired, present, commanded, max_step=3.0, max_lead=100.0)

    assert step["elbow_flex.pos"] == pytest.approx(-3.0)


def test_a_target_within_both_limits_is_applied_unchanged() -> None:
    desired = {"elbow_flex.pos": 2.0}
    present = {"elbow_flex": 0.0}
    commanded = {"elbow_flex.pos": 0.0}

    step = clamp_target(desired, present, commanded, max_step=3.0, max_lead=15.0)

    assert step["elbow_flex.pos"] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# recording
# --------------------------------------------------------------------------- #


def test_roles_default_to_main_and_wrist_then_name_the_device() -> None:
    assert role_names([0, 2], ()) == ("main", "wrist")
    assert role_names([0, 2, 4], ()) == ("main", "wrist", "camera_4")
    assert role_names([0, 2], ["front", "back"]) == ("front", "back")


def test_recording_without_an_open_episode_is_a_no_op(tmp_path) -> None:
    recorder = EpisodeRecorder(tmp_path, [], 20.0)

    recorder.record(0, {}, {}, {})

    assert list(tmp_path.iterdir()) == []


def test_an_episode_opens_writes_a_control_row_and_closes(tmp_path) -> None:
    recorder = EpisodeRecorder(tmp_path, [], 20.0)

    episode = recorder.start()
    assert recorder.active
    recorder.record(3, {"a": 1.0}, {"a": 0.0}, {"a": 1.0})
    closed = recorder.stop()

    assert closed == episode
    assert not recorder.active
    rows = [json.loads(line) for line in (Path(episode) / "frames.jsonl").read_text().splitlines()]
    assert rows[0]["event"] == "start"
    assert rows[1]["sequence"] == 3
    assert rows[-1]["event"] == "stop"


def test_marking_an_episode_records_the_decision(tmp_path) -> None:
    recorder = EpisodeRecorder(tmp_path, [], 20.0)
    episode = recorder.start()
    recorder.stop()

    path = recorder.mark(episode, "keep", task="pick", follower="a")

    assert json.loads(path.read_text())["decision"] == "keep"


def test_starting_twice_reuses_the_open_episode(tmp_path) -> None:
    recorder = EpisodeRecorder(tmp_path, [], 20.0)

    first = recorder.start()
    second = recorder.start()
    recorder.stop()

    assert first == second


def test_a_second_episode_can_start_after_a_stop(tmp_path) -> None:
    """The guard against racing camera workers must not block the next episode."""

    recorder = EpisodeRecorder(tmp_path, [], 20.0)

    first = recorder.start()
    recorder.stop()
    second = recorder.start()
    recorder.stop()

    assert first != second
    assert len(list(tmp_path.iterdir())) == 2
