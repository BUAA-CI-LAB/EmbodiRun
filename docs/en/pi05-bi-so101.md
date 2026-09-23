# Pi0.5 with two SO-101 followers

This configuration treats two followers as one dual-arm robot. It differs
from the [multi-robot demo](demos/multi-robot-serving.md), where separate
clients control individual arms.

## Prerequisites

Prepare two calibrated followers with distinct ports and calibration IDs,
front and wrist cameras, and a CUDA host. The checkpoint must match the
dual-arm state/action layout and the three camera inputs. Use a checkpoint
trained for this setup.

## Configure and validate

```bash
cp configs/pi05/bi-so101-embodiinfer.yaml my-deployment.yaml
# Edit the deployment revision, devices, calibration, and checkpoint paths.
uv run embodirun --config my-deployment.yaml validate
uv run embodirun --config my-deployment.yaml probe
```

The example runtime is `bi-so101-pi05`. Follow [Quick start](quickstart.md)
for preparation, startup, bounded execution, and shutdown. Read
[Safety](safety.md) before connecting or moving the arms.

## Action layout and failure behavior

`lerobot.bi_so101` composes two existing calibrated SO-101 adapters. Each arm
has its own serial port and calibration identity; state and action order is
left six values followed by right six values, with grippers in the native
`range_0_100` convention. The binding matches action features by name and
rejects malformed or over-sized commands before the first bus write.

Use [configs/pi05/bi-so101-embodiinfer.yaml](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/pi05/bi-so101-embodiinfer.yaml) as a
validation template. Replace serial paths, camera paths, calibration IDs, and
the inference checkpoint. `validate` and `build_plan` do not contact devices.
The dual adapter belongs to the same Deploy Control resource owner as the
single-arm adapter. Give it exclusive access to both serial ports; stop any
other teleoperation or control service using those arms first.

The implementation keeps each bus sequential. If the second command fails,
both arms receive a measured hold attempt and the original failure remains
visible. Disconnect torque handling, step limits, passive connection, and
explicit preparation are inherited from the single-arm adapter. The dual-arm
path has automated software tests; physical-robot validation is still pending.
