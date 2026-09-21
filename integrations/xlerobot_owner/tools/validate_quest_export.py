#!/usr/bin/env python3
"""Exercise the real installed LeRobot writer with explicitly synthetic input."""

import argparse
import json
import time
from pathlib import Path

from embodirun_xlerobot_owner.control import JOINT_NAMES
from embodirun_xlerobot_owner.recording import EpisodeRecorder, export_lerobot
from embodirun_xlerobot_owner.robot import DemoRobot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    robot = DemoRobot()
    metadata = {
        **robot.metadata,
        "mode": "demo",
        "collection_mode": "teleoperation",
        "fps": 20,
        "max_skew_ms": 100,
        "max_gap_ms": 150,
    }
    recorder = EpisodeRecorder(args.output / "raw", fps=20)
    episode = recorder.start("synthetic export validation - not robot training data", metadata)
    began = time.time_ns()
    for index in range(8):
        observation, images = robot.read()
        stamp = began + index * 50_000_000
        observation.update(
            {
                "source_timestamp_ns": stamp,
                "state_timestamp_ns": stamp,
                "camera_timestamps_ns": dict.fromkeys(images, stamp),
                "received_timestamp_ns": stamp + 1_000_000,
            }
        )
        action = {name: observation["state"][name] for name in JOINT_NAMES}
        recorder.append(
            observation=observation,
            action=action,
            images=images,
            feedback={
                "source": "synthetic",
                "applied_action": action,
                "accepted": True,
                "sent_timestamp_ns": stamp + 2_000_000,
                "physical_outcome": "synthetic",
            },
        )
    recorder.finish(success=None, reason="synthetic software test")
    result = export_lerobot(
        [episode],
        repo_id="local/quest-export-test",
        output=args.output / "lerobot",
        fps=20,
        allow_demo=True,
    )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("local/quest-export-test", root=args.output / "lerobot")
    frame = dataset[0]
    report = {
        "source": "synthetic",
        "physical_validation": False,
        "frame_count": len(dataset),
        "action_shape": list(frame["action"].shape),
        "state_shape": list(frame["observation.state"].shape),
        "camera_shapes": {
            name: list(frame[f"observation.images.{name}"].shape) for name in ("front", "left_wrist", "right_wrist")
        },
        "export_result": result,
    }
    assert len(dataset) == 8
    assert report["action_shape"] == report["state_shape"] == [12]
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "export_result"}))


if __name__ == "__main__":
    main()
