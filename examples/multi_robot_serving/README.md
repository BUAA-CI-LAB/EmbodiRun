# Three robots, one inference service

Launch three independent SO-101 runtimes against one π0.5 service. Each arm has
its own robot computer, calibration, front/wrist cameras, and session. The
three rollout clients start concurrently.

## Prepare the deployment

1. Install the Host environment with `uv sync --frozen`.
2. Provide SSH access to the GPU host and three robot computers. Verify their
   host keys first; the template does not automatically trust them.
3. Place an SO-101-trained π0.5 checkpoint on the GPU host. Its state/action
   features and normalization must match the six-axis SO-101 binding.
4. Calibrate each arm and gripper, confirm the camera roles, and test the
   emergency stop. Keep all three workspaces clear and supervised.

Copy both files in place:

```bash
cp examples/multi_robot_serving/example.yaml examples/multi_robot_serving/example.local.yaml
cp examples/multi_robot_serving/deployment.yaml examples/multi_robot_serving/deployment.local.yaml
```

Set `parameters.deployment: deployment.local.yaml` in the local manifest.
Replace every `REPLACE_*` value in the deployment: SSH addresses/user, serial
and camera paths, calibration directory/IDs, and checkpoint directory. These
paths belong to their respective nodes, not the Host machine.

The template follows `metadata.deploy-commit: main`. For a repeatable campaign,
set this to a published EmbodiRun commit whose EmbodiInfer pin includes batch
serving before running `setup`; keep that revision across all repetitions.

The template requests 640 × 480 MJPEG at 20 fps and runs up to 15 chunks of
50 actions at 20 Hz. Actions exceeding the configured step limits are rejected.
Choose task limits for your calibrated robot before enabling motion.

The shared service collects requests into batches of up to three, with a 5 ms
collection window. Each robot retains its own session and action sequence.
Requests arriving outside the same window run in smaller batches.

## Run

From the repository root:

```bash
CONFIG=examples/multi_robot_serving/example.local.yaml
bash examples/run.sh "$CONFIG" validate
bash examples/run.sh "$CONFIG" plan
bash examples/run.sh "$CONFIG" setup
bash examples/run.sh "$CONFIG" up --allow-hardware
bash examples/run.sh "$CONFIG" run --allow-hardware
bash examples/run.sh "$CONFIG" down
```

`setup` prepares remote environments and syncs this checkout. `up` loads the
checkpoint and connects the devices. `run` starts one bounded rollout per arm;
it does not reset the scene. Reset the cubes and bowls before another run.
Use `down` before changing deployments or leaving the robots.

Host state is under `artifacts/examples/multi-robot-serving/host-state`.
Each run directory contains `run.json` and three command logs in the order of
`parameters.runtimes`. These report execution; the launcher does not compose
a demo video or produce a benchmark latency table.

[Manifest reference](../README.md) ·
[Demo video](https://embodirun.readthedocs.io/en/latest/demos/multi-robot-serving/)
