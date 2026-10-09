# Reproduce the demos

The four demos use one entrypoint, `embodirun example`, and versioned YAML
manifests. Deployment files describe devices and services; the example manifest
selects the task, execution limits, and output directory.

| Demo | Example directory | Execution |
|---|---|---|
| [Three robots, one service](demos/multi-robot-serving.md) | [multi_robot_serving](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/examples/multi_robot_serving) | Three concurrent SO-101 loops, shared π0.5 batches |
| [Engine comparison](demos/engine-e2e-contrast.md) | [engine_comparison](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/examples/engine_comparison) | EmbodiInfer HTTP/WirelessComm or SGLang HTTP |
| [Snack delivery](demos/xlerobot-snack-delivery.md) | [xlerobot_snack_delivery](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/examples/xlerobot_snack_delivery) | Routes, VLA grasping, supervised handover |
| [MicroDuck navigation](demos/microduck-vln.md) | [microduck_vln](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/examples/microduck_vln) | MuJoCo and ActiveVLN, local GPU or Slurm |

## Inspect a configuration

From the repository root:

```bash
uv sync --frozen
uv run --frozen embodirun example init so101
CONFIG=examples/local/so101/example.local.yaml
uv run --frozen embodirun example "$CONFIG" validate
uv run --frozen embodirun example "$CONFIG" plan
```

`validate` checks fields and deployment references locally; `plan` prints the
commands that a run would execute.

`init` creates a new ignored directory containing a manifest and its linked
deployment files. It does not overwrite existing configuration. Choose
`xlerobot`, `microduck`, `embodiinfer-http`, `embodiinfer-wireless`, or
`sglang-http` instead of `so101` for those recipes. Fill in device paths, SSH
hosts, calibration, checkpoint paths, and task settings before deployment.
For SO-101 and XLeRobot, `embodirun example "$CONFIG" check` lists local
missing prerequisites without opening devices. It does not establish live
hardware or model readiness. MicroDuck's `check` additionally runs a GPU/EGL
scene preflight on the target Linux host.

Manifest paths resolve relative to the YAML file. Deployment device and model
paths belong to the node where they are used. The complete field and command
reference is in [examples/README.md](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/README.md).

## Run a hardware example

After following the selected example's calibration and environment instructions:

```bash
CONFIG=examples/local/so101/example.local.yaml
uv run --frozen embodirun example "$CONFIG" setup
uv run --frozen embodirun example "$CONFIG" check
uv run --frozen embodirun example "$CONFIG" up --allow-hardware
uv run --frozen embodirun example "$CONFIG" run --allow-hardware
uv run --frozen embodirun example "$CONFIG" down
```

SO-101 services remain running between tasks. Reset the scene manually and use
`down` when finished. XLeRobot keeps `up` in the foreground; run the task in
a second terminal, then use `down` or Ctrl-C to stop its stack.
Keep an operator present and verify the emergency stop before enabling motion.

For a hardware-free task rehearsal:

```bash
uv run --frozen embodirun example examples/local/xlerobot/example.local.yaml dry-run
```

## Run the simulator

Follow [MicroDuck setup](microduck-vln.md) to prepare the external scene and
checkpoint. On the Linux GPU host, run:

```bash
uv run --frozen embodirun example init microduck --assets /absolute/asset/root
CONFIG=examples/local/microduck/example.local.yaml
uv run --frozen embodirun example "$CONFIG" setup
uv run --frozen embodirun example "$CONFIG" check
uv run --frozen embodirun example "$CONFIG" run
```

`check` verifies assets and the GPU/rendering environment. The YAML selects
local or Slurm execution, episodes, seed, action limit, and video frame rate.
The inference child stops when the run exits.

## Inspect the outputs

Each invocation gets a fresh directory under its configured `output_dir`, with
`run.json` recording the configuration, revision, input hashes, and status.
SO-101 has one command log per runtime. MicroDuck and XLeRobot write their
scene-specific results under `result/`, including episode videos/metrics or
delivery events and proposals.

Use the [inference transport benchmark](inference-transport.md) for repeated
latency measurements.
