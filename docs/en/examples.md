# Scenarios and Recipes

Run manipulation, navigation, or mobile-manipulation tasks using the shared
`embodirun example` CLI (`examples/run.sh` remains compatible). Each Recipe
covers environment setup, local configuration, task execution, output files,
and shutdown. When testing an unmerged PR, install from its head branch.
For an actual software trial by a new user or coding Agent, follow
[independent first use](first-use.md).

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
hardware or model readiness. MicroDuck's software `check` reads configuration
and installed metadata; its default simulation `check` without `--json`
additionally runs a GPU/EGL scene preflight on the target Linux host.

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

Start with the software path from the repository root:

```bash
uv sync --frozen --python 3.12
uv run --frozen embodirun example init microduck
CONFIG="$PWD/examples/local/microduck/example.local.yaml"
uv run --frozen embodirun example "$CONFIG" validate
uv run --frozen embodirun example "$CONFIG" plan
uv run --frozen embodirun example "$CONFIG" setup --mode software
RECIPE_ENV="$PWD/.venv-microduck"
"$RECIPE_ENV/bin/embodirun" example "$CONFIG" check --mode software --json
```

Software `check` works with or without `--json` and leaves assets, CUDA/EGL,
dependency imports and model execution unverified. No scene or inference process
starts. `plan` only previews a simulation command; MicroDuck has no fixture
`dry-run`. Both setup modes use the Recipe lock, as do its Docker targets.
Follow [MicroDuck setup](microduck-vln.md) to provide the external scene,
compatible checkpoint, episodes, inventory and pinned inference source, then
use `setup --mode simulation` on Linux. Simulation `check --json` reads installed
metadata and local paths; simulation `check` without `--json` continues into
asset/GPU/rendering preflight. The default setup/check mode is `simulation`.
Preflight does not run learned-model inference. The YAML selects local or Slurm
execution, episodes, seed, action limit and video frame rate.
The inference child stops when the run exits; `down` from another terminal
with the same checkout/configuration requests this foreground launcher's shutdown.

## Inspect the outputs

Each invocation gets a fresh directory under its configured `output_dir`, with
`run.json` recording the configuration, revision, input hashes, and status.
SO-101 has one command log per runtime. MicroDuck and XLeRobot write their
scene-specific results under `result/`, including episode videos/metrics or
delivery events and proposals.

Use the [inference transport benchmark](inference-transport.md) for repeated
latency measurements.
The separate [Runtime cost procedure](runtime-value.md) links an Agent's report
of three prepared-software startup/stop pairs and a separate occupied-port
fault/recovery per route, alongside empty collection templates. Full same-model
native LeRobot comparison, the six deployment-cost procedures and human
diagnosis time remain unmeasured.
