# Runnable examples

Use one YAML manifest and one launcher for each demo. Run these commands from
the repository root after `uv sync --frozen`:

```bash
bash examples/run.sh examples/multi_robot_serving/example.yaml validate
bash examples/run.sh examples/multi_robot_serving/example.yaml plan
```

Both commands are offline: they neither connect to devices nor start inference.
`validate` checks the configuration; `plan` prints the command, arguments, and
output directory that `run` would use.

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
- `python` (optional): interpreter for MicroDuck or XLeRobot; defaults to the
  launcher's interpreter. Host commands use the Host environment.
- `parameters`: settings for that execution path, shown in the shipped manifests.
  Unknown fields, duplicate YAML keys, and invalid numeric limits are rejected.

Manifest paths resolve **relative to the YAML file**, independent of the shell's
working directory. Paths inside a deployment remain paths on the corresponding
node. There is no shell interpolation in YAML; write concrete values. Copy a
manifest and deployment to `*.local.yaml` in the same directory before editing;
local YAML/JSON files are ignored by Git.

Deployment YAML owns nodes, devices, cameras, models, and bindings. The example
manifest owns the task and execution limits. XLeRobot references its existing
task JSON and owner deployment; MicroDuck references its model, scene assets,
and episode manifest.

## Commands

| Command | SO-101 rollout | XLeRobot | MicroDuck |
|---|---|---|---|
| `validate` / `plan` | Offline configuration / command preview | Same | Same; assets need not be installed |
| `setup` | Host `init`, then sync this checkout | Install the example and owner environment | Follow the dedicated environment instructions |
| `up --allow-hardware` | Start model and robot services | Start owner and Control in the foreground | Inference starts with `run` |
| `run` | Concurrent bounded rollouts; requires `--allow-hardware` | Supervised delivery; requires `--allow-hardware` | Launch simulation and inference |
| `dry-run` | — | Fixture-only task rehearsal | — |
| `check` | — | — | Check assets, CUDA, EGL, controller, and encoder |
| `down` | Stop this deployment | Ctrl-C in the `up` terminal | Child services stop when `run` exits |

Hardware examples require calibrated devices, a working emergency stop, and an
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

## Smaller software examples

- [Simulated device and public Control API](shared-device-fake.md)
- [Recorded observations with real inference and simulated execution](shared-device-inference.md)

For latency measurements and report comparison, use
[benchmarks](../benchmarks/README.md).
