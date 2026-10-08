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

Start with the locked configuration-only profile from the repository root:

```bash
RECIPE_ENV="$PWD/.venv-microduck"
UV_PROJECT_ENVIRONMENT="$RECIPE_ENV" uv sync \
  --project examples/microduck_vln --frozen --python 3.12 --no-default-groups
"$RECIPE_ENV/bin/embodirun" example init microduck --assets /absolute/asset/root
CONFIG=examples/local/microduck/example.local.yaml
"$RECIPE_ENV/bin/embodirun" example "$CONFIG" validate
"$RECIPE_ENV/bin/embodirun" example "$CONFIG" plan
"$RECIPE_ENV/bin/embodirun" example "$CONFIG" check --json
```

`check --json` reports local path prerequisites only. `check` without `--json`
continues into asset/GPU/rendering preflight after the local checks pass; it
does not run learned-model inference. Follow [MicroDuck setup](microduck-vln.md)
to provide the scene/checkpoint, install the locked `simulation` extra on Linux
and execute `check` and `run`. The CLI has no MicroDuck `--mode software` option;
the existing `setup` and Docker installer have not yet adopted the new lock.
The YAML selects
local or Slurm execution, episodes, seed, action limit, and video frame rate.
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
The separate [Runtime cost procedure](runtime-value.md) supplies unexecuted
deployment comparison instructions and empty collection templates.
