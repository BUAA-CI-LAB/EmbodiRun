"""Record real SO101 camera frames and calibrated state without motor writes."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from embodirun.robots.lerobot.so101 import SO101Adapter, SO101Config
from embodirun.robots.sensors import SensorInput
from embodirun.robots.sensors.cameras.v4l2 import create_source


def capture(config, output, *, prompt, count, interval_s):
    if config["robot"]["type"] != "lerobot.so101":
        raise ValueError("capture requires an SO101 control configuration")
    if not prompt.strip() or count <= 0 or not math.isfinite(interval_s) or interval_s < 0:
        raise ValueError("capture requires a prompt, positive count and non-negative interval")
    if any(item["type"] != "v4l2" for item in config["inputs"]):
        raise ValueError("this capture script requires V4L2 camera inputs")
    robot = SO101Adapter(SO101Config.from_mapping(config["robot"]["id"], config["robot"]["options"]), read_only=True)
    cameras = None
    output.mkdir(parents=True, exist_ok=False)
    try:
        robot.connect()
        cameras = create_source(
            tuple(
                SensorInput(sensor_id=item["sensor_id"], name=item["name"], kind=item["type"], options=item["options"])
                for item in config["inputs"]
            )
        )
        with (output / "observations.jsonl").open("x") as manifest:
            for index in range(count):
                start = time.monotonic()
                captured_at = time.time()
                frames = cameras.capture()
                observation = robot.observe()
                image_paths = {}
                for camera_index, frame in enumerate(frames):
                    filename = f"{index:04d}-camera-{camera_index}.jpg"
                    (output / filename).write_bytes(frame.data)
                    image_paths[frame.name] = filename
                row = {
                    "instruction": prompt,
                    "state": observation.values,
                    "images": image_paths,
                    "source": "recorded-so101:" + config["runtime_id"],
                    "capture_started_unix_s": captured_at,
                    "state_timestamp_unix_s": observation.timestamp_s,
                    "capture_ms": (time.monotonic() - start) * 1000,
                }
                manifest.write(json.dumps(row, allow_nan=False) + "\n")
                manifest.flush()
                if index + 1 < count:
                    time.sleep(max(0, interval_s - (time.monotonic() - start)))
    finally:
        try:
            if cameras is not None:
                cameras.close()
        finally:
            robot.close()
    return output / "observations.jsonl"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="generated control config JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()
    print(
        capture(
            json.loads(args.config.read_text()),
            args.output,
            prompt=args.prompt,
            count=args.count,
            interval_s=args.interval,
        )
    )


if __name__ == "__main__":
    main()
