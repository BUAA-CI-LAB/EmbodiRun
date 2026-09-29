# Runnable examples

Use the installed `embodirun example` command for all recipes. `examples/run.sh`
uses the same Python runner for source checkouts. From the repository root:

```bash
uv sync --frozen
uv run --frozen embodirun example init xlerobot
uv run --frozen embodirun example examples/local/xlerobot/example.local.yaml validate
uv run --frozen embodirun example examples/local/xlerobot/example.local.yaml plan
```

`init` creates one new, ignored directory with all referenced files linked:
manifest, deployment, hardware configuration, task, and routes for XLeRobot.
It refuses to overwrite an existing directory. Other templates are `so101`,
`embodiinfer-http`, `embodiinfer-wireless`, `sglang-http`, and `microduck`.
Use `--output NEW_DIRECTORY` to choose a location; MicroDuck accepts
`--assets ASSET_ROOT` to fill its project, checkpoint, and episode paths.
`validate` and `plan` are offline. `plan` prints the command and output path
without launching it.

| Demo | Configuration | Setup and execution |
|---|---|---|
| Three SO-101 arms, one service | [multi_robot_serving/example.yaml](multi_robot_serving/example.yaml) | [Instructions](multi_robot_serving/README.md) |
| SO-101 engine comparison | [engine_comparison/embodiinfer-http.yaml](engine_comparison/embodiinfer-http.yaml), [embodiinfer-wireless.yaml](engine_comparison/embodiinfer-wireless.yaml), [sglang-http.yaml](engine_comparison/sglang-http.yaml) | [Instructions](engine_comparison/README.md) |
| XLeRobot snack delivery | [xlerobot_snack_delivery/example.yaml](xlerobot_snack_delivery/example.yaml) | [Instructions](xlerobot_snack_delivery/README.md) |
| MicroDuck navigation | [microduck_vln/example.yaml](microduck_vln/example.yaml) | [Instructions](microduck_vln/README.md) |

## Configuration contract

```yaml
version: 1
name: multi-robot-serving
kind: rollout
output_dir: ../../artifacts/examples/multi-robot-serving
parameters:
  deployment: deployment.local.yaml
  runtimes: [arm-1, arm-2, arm-3]
  prompt: Pick up the cube and place it in the bowl.
  chunks: 15
  chunk_steps: 50
  control_hz: 20
  request_timeout: 120
```

- `version`: manifest schema version, currently `1`.
- `name`: a readable identifier for the example.
- `kind`: `rollout`, `microduck`, or `snack`; selects an existing execution path.
- `output_dir`: output root. Each invocation gets a fresh timestamped directory.
- `python` (optional): interpreter for MicroDuck or XLeRobot; the launcher
  otherwise selects the dedicated recipe environment after `setup`. Host
  commands use the Host environment.
- `parameters`: settings for that execution path, shown in the shipped manifests.
  Unknown fields, duplicate YAML keys, and invalid numeric limits are rejected.

Manifest paths resolve **relative to the YAML file**, independent of the shell's
working directory. Paths inside a deployment remain paths on the corresponding
node. There is no shell interpolation in YAML; write concrete values. Use
`init` to create and link editable local files; `examples/local/` is ignored by
Git.

Deployment YAML owns nodes, devices, cameras, models, and bindings. The example
manifest owns the task and execution limits. XLeRobot references its existing
task JSON and owner deployment; MicroDuck references its model, scene assets,
and episode manifest.

## Commands

| Command | SO-101 rollout | XLeRobot | MicroDuck |
|---|---|---|---|
| `validate` / `plan` | Offline configuration / command preview | Same | Same; assets need not be installed |
| `setup` | Host `init`, then sync this checkout | Install Host from `uv.lock` and owner integration | Install Host from `uv.lock` and MicroDuck integration |
| `up --allow-hardware` | Start model and robot services | Start owner and Control in the foreground | Inference starts with `run` |
| `run` | Concurrent bounded rollouts; requires `--allow-hardware` | Supervised delivery; requires `--allow-hardware` | Launch simulation and inference |
| `dry-run` | — | Fixture-only task rehearsal | — |
| `check` | Report local placeholders and node-owned prerequisites | Report local SDK, calibration, camera role, route, and fixture gaps | Check paths and run the scene's CUDA/EGL/asset preflight |
| `down` | Stop this deployment | Ask this recipe's foreground launcher to stop | Ask this recipe's foreground launcher to stop |

`check` is a prerequisite check, not proof that a robot, model, camera, or stop
feedback is ready. The MicroDuck scene preflight uses GPU resources; run it on
the target Linux GPU host. Hardware examples require calibrated devices, a working emergency stop, and an
operator. `--allow-hardware` permits startup or motion; it does not skip action
limits or XLeRobot's confirmation gates. Stop one deployment before starting
another that uses the same robot or ports.

## Outputs and shutdown

Every executing command writes `run.json` with the manifest, argv, source
revision, input hashes, and completion status. `inputs/` preserves the referenced
configuration files so later edits do not erase the run's settings. Noninteractive child output goes
to `command-0.log`, `command-1.log`, etc. XLeRobot's hardware prompts and service
startup stay in the terminal. MicroDuck and XLeRobot write detailed results
under `result/`; their guides describe those files.

SO-101 services remain running after a successful rollout so you can reset the
scene and repeat it. Use `down` when finished. On a rollout failure or interrupt,
the launcher stops the deployment, not just the local HTTP clients. Inspect the
robot and use the physical emergency stop if shutdown cannot be confirmed.
Do not run unrelated work in the example's deployment.

Set `EMBODIRUN_EXAMPLE_PYTHON` to an absolute Host Python path if you are not
using the repository's `.venv`.

For a first Linux deployment, run `uv run --frozen embodirun example
examples/local/xlerobot/example.local.yaml setup` before `check`. Host packages
come from `uv.lock`; the separately installed owner/MicroDuck integration
packages use their declared version ranges and are not fully locked by that
file. Do not copy an existing machine's virtual environment. Container targets
`host`, `xlerobot`, and `microduck` are in the root `Dockerfile`; see the
installation guide for their Linux prerequisites and the Compose opt-in
profiles. Building an image does not enable robot motion.

## Smaller software examples

- [Simulated device and public Control API](shared-device-fake.md)
- [Recorded observations with real inference and simulated execution](shared-device-inference.md)

For latency measurements and report comparison, use
[benchmarks](../benchmarks/README.md).

## First-use feedback

For a first software-only trial, start from a fresh checkout, run
`uv sync --frozen`, then `uv run --frozen embodirun example init xlerobot`,
`validate`, `plan`, and `dry-run` against
`examples/local/xlerobot/example.local.yaml` through the same `uv run --frozen
embodirun example` entrypoint. On the
target Linux host, try `setup` and `check` separately; stop before `up` or
`run --allow-hardware` unless an authorized operator has completed calibration
and physical stop checks. Record the OS/architecture, Python and uv versions,
exact commands and exit codes, the first blocking error, and the generated
`run.json`/command log path with secrets and private routes removed. This
documents what a new user could reproduce; it is not a substitute for actual
independent-user feedback.
