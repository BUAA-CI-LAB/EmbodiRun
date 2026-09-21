# Reproduce the demos

The four demos use one entrypoint, `examples/run.sh`, and versioned YAML
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
bash examples/run.sh examples/multi_robot_serving/example.yaml validate
bash examples/run.sh examples/multi_robot_serving/example.yaml plan
```

`validate` checks fields and deployment references locally; `plan` prints the
commands that a run would execute.

Copy the selected YAML and its referenced deployment to `*.local.yaml` in the
same directory. Update the reference in `parameters.deployment`, then fill in
device paths, SSH hosts, calibration, checkpoint paths, and task settings.
Local YAML/JSON files are ignored by Git.

Manifest paths resolve relative to the YAML file. Deployment device and model
paths belong to the node where they are used. The complete field and command
reference is in [examples/README.md](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/README.md).

## Run a hardware example

After following the selected example's calibration and environment instructions:

```bash
CONFIG=examples/multi_robot_serving/example.local.yaml
bash examples/run.sh "$CONFIG" setup
bash examples/run.sh "$CONFIG" up --allow-hardware
bash examples/run.sh "$CONFIG" run --allow-hardware
bash examples/run.sh "$CONFIG" down
```

SO-101 services remain running between tasks. Reset the scene manually and use
`down` when finished. XLeRobot keeps `up` in the foreground; run the task in
a second terminal, then Ctrl-C in the service terminal to stop its stack.
Keep an operator present and verify the emergency stop before enabling motion.

For a hardware-free task rehearsal:

```bash
bash examples/run.sh examples/xlerobot_snack_delivery/example.yaml dry-run
```

## Run the simulator

Follow [MicroDuck setup](microduck-vln.md) to install its optional environment
and prepare the scene and checkpoint. Fill in `example.local.yaml`, then run:

```bash
CONFIG=examples/microduck_vln/example.local.yaml
bash examples/run.sh "$CONFIG" check
bash examples/run.sh "$CONFIG" run
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
