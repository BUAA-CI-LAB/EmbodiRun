"""Read-only observations for transport experiments, with no hardware boundary."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


class ReplaySource:
    """Cycle through JSONL observations or return a deterministic synthetic frame.

    Records contain ``state`` (a numeric list or SO101 state dictionary),
    ``images`` (camera name to path relative to the manifest), and ``instruction``.
    This source has no execute method, SDK connection, or dynamics/reward model.
    """

    def __init__(self, manifest: str | None = None, *, state_dim: int = 7, image_size: int = 224):
        if state_dim <= 0 or image_size <= 0:
            raise ValueError("state_dim and image_size must be positive")
        self.state_dim = state_dim
        self.image_size = image_size
        self.frames = []
        if manifest:
            from PIL import Image

            path = Path(manifest)
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                state = record["state"]
                if isinstance(state, dict):
                    state = [*state["joint_positions_deg"], state["gripper_position"]]
                if len(state) != state_dim:
                    raise ValueError("replay state dimension does not match config")
                if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in state):
                    raise ValueError("replay state must contain finite numbers")
                images = {}
                for name, filename in record["images"].items():
                    with Image.open(path.parent / filename) as source:
                        images[name] = source.convert("RGB").resize((image_size, image_size)).tobytes()
                if not images:
                    raise ValueError("replay frame must contain an image")
                self.frames.append(
                    {
                        "state": state,
                        "images": images,
                        "instruction": record.get("instruction", "replay"),
                    }
                )
            if not self.frames:
                raise ValueError("empty replay manifest")
        else:
            self.frames = [
                {
                    "state": [0.0] * state_dim,
                    "images": {
                        "front": bytes((31, 97, 173)) * image_size**2,
                        "wrist": bytes((173, 97, 31)) * image_size**2,
                    },
                    "instruction": "synthetic communication replay",
                }
            ]

    def observation(self, index: int) -> dict:
        return self.frames[index % len(self.frames)]

    def fingerprint(self) -> str:
        """Identify decoded replay content, including images, across trial nodes."""
        digest = hashlib.sha256()
        digest.update(json.dumps([self.state_dim, self.image_size]).encode())
        for frame in self.frames:
            record = {
                "state": frame["state"],
                "instruction": frame["instruction"],
                "images": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(frame["images"].items())},
            }
            digest.update(json.dumps(record, sort_keys=True).encode())
        return digest.hexdigest()
