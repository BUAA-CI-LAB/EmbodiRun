# Scenarios and Recipes

Run manipulation, navigation, or mobile-manipulation tasks using the shared
`examples/run.sh` launcher. Each Recipe covers environment setup, local
configuration, task execution, output files, and shutdown.

| Scenario | Demo | Recipe |
|---|---|---|
| **VLA** | [SO-101 grasping](demos/so101-grasping.md) | [One arm + π0.5 / HTTP](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/so101_grasping.md) |
| **VLN** | [MicroDuck navigation](demos/microduck-vln.md) | [MuJoCo + ActiveVLN](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/microduck_vln/README.md) |
| **Agent** | [XLeRobot snack delivery](demos/xlerobot-snack-delivery.md) | [Recorded routes, RPent/Astra, and VLA grasping](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/xlerobot_snack_delivery/README.md) |

The [Recipe index](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/README.md)
also includes [multi-arm serving](demos/multi-robot-serving.md),
[engine and transport variants](demos/engine-e2e-contrast.md), and software-only
examples. Use the [support matrix](support-matrix.md) to choose other devices
or simulators.

Deployment YAML defines devices and services; the example manifest selects the
task, execution limits, and output directory.

## Inspect a configuration

From the repository root:

```bash
uv sync --frozen
bash examples/run.sh examples/engine_comparison/embodiinfer-http.yaml validate
bash examples/run.sh examples/engine_comparison/embodiinfer-http.yaml plan
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
CONFIG=examples/engine_comparison/so101.local.yaml
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
