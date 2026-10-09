# VLA Recipe: SO-101 grasping with π0.5

Run one SO-101 follower arm through EmbodiRun, with π0.5 served by EmbodiInfer
over HTTP. The task is “Pick up the cube and place it in the bowl.” This Recipe
reuses the existing `engine_comparison` manifest and code.

## Requirements

- A calibrated SO-101 follower, front and wrist cameras, and an operator.
- A robot host with the optional SO-101 dependencies and a GPU inference host.
- SSH access from the Host machine to both hosts; they may be the same machine.
- A π0.5 checkpoint trained for the single-arm SO-101 feature names and units.
  Checkpoints and calibration files are supplied by the deployment owner.

Read [installation](../docs/en/installation.md) and [safety](../docs/en/safety.md)
before preparing hardware. The templates use two 640 × 480 cameras and 50-step
action chunks at 20 Hz; hardware limits still apply.

## Inspect and configure

From the repository root:

```bash
uv sync --frozen
bash examples/run.sh examples/engine_comparison/embodiinfer-http.yaml validate
bash examples/run.sh examples/engine_comparison/embodiinfer-http.yaml plan
cp examples/engine_comparison/embodiinfer-http.yaml examples/engine_comparison/so101.local.yaml
cp examples/engine_comparison/embodiinfer-http.deployment.yaml examples/engine_comparison/so101.deployment.local.yaml
```

`validate` and `plan` are offline. In `so101.local.yaml`, change
`parameters.deployment` to `so101.deployment.local.yaml`, retain
`runtimes: [arm-1]`, and choose the task limits and output directory.
In the deployment, replace every `REPLACE_*` value: SSH hosts/users, arm and
camera paths, calibration directory/id, and checkpoint path. Device paths belong
to the configured robot node; the checkpoint path belongs to the GPU node.

The shipped model configuration uses BF16, 10 denoising steps, HTTP, and
`--max-batch 1`. The public inference service also supports opt-in π0.5
cross-session batching for the [multi-robot Recipe](multi_robot_serving/README.md).

## Prepare, launch, execute, and stop

After completing calibration and hardware preparation:

```bash
CONFIG=examples/engine_comparison/so101.local.yaml
bash examples/run.sh "$CONFIG" validate
bash examples/run.sh "$CONFIG" plan
bash examples/run.sh "$CONFIG" setup
bash examples/run.sh "$CONFIG" up --allow-hardware
bash examples/run.sh "$CONFIG" run --allow-hardware
bash examples/run.sh "$CONFIG" down
```

`setup` prepares the configured deployment environments; `up` starts the
model and Control services. `run` executes a bounded task. Reset the scene
between tasks, and use `down` when finished. Hardware startup and motion require
`--allow-hardware` and operator preparation described in the shared launcher.

## Outputs

Each invocation writes a timestamped directory under `output_dir` containing
`run.json`, preserved configuration inputs, and command logs. Inspect the
rollout log for task execution details. If a Control recorder is configured,
retrieve its recording through the recording API.

## Variants

- [WirelessComm and SGLang profiles](engine_comparison/README.md).
- [Three arms sharing one service](multi_robot_serving/README.md).
- [Manipulation Recipe index](README.md#vla--manipulation).
- [中文场景与教程入口](../docs/zh/demos/so101-grasping.md).
