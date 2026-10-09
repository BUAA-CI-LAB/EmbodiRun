# Runnable examples

Use the installed `embodirun example` command for all recipes. `examples/run.sh`
uses the same Python runner for source checkouts. From the repository root:

```bash
uv sync --frozen --python 3.12
uv run --frozen embodirun example init xlerobot
uv run --frozen embodirun example examples/local/xlerobot/example.local.yaml validate
uv run --frozen embodirun example examples/local/xlerobot/example.local.yaml plan
uv run --frozen embodirun example examples/local/xlerobot/example.local.yaml setup --mode software
uv run --frozen embodirun example examples/local/xlerobot/example.local.yaml check --mode software --json
uv run --frozen embodirun example examples/local/xlerobot/example.local.yaml dry-run
```

`init` creates one new, ignored directory with all referenced files linked:
manifest, deployment, hardware configuration, task, and routes for XLeRobot.
It refuses to overwrite an existing directory and shows how to reuse an existing
manifest. Other templates are `so101`,
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

An explicit manifest `python` takes priority over `EMBODIRUN_SCENE_PYTHON`,
including the interpreter configured by a Docker image. Before moving a native
manifest into Docker, back it up and remove only the Docker copy's top-level
`python` override to use the image's `/opt/venv/bin/python`; preserve other
settings and calibration. Alternatively, initialize a separate Docker directory
and migrate your local configuration with its path references.

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
| `setup` | Host `init`, then sync this checkout | `--mode software` for the small runtime; `--mode hardware` for full owner, both from the Recipe lock | `--mode software` for CLI/integration checks; default `simulation` adds its locked simulator/inference extra |
| `up --allow-hardware` | Start model and robot services | Start owner and Control in the foreground | Inference starts with `run` |
| `run` | Concurrent bounded rollouts; requires `--allow-hardware` | Supervised delivery; requires `--allow-hardware` | Launch simulation and inference |
| `dry-run` | — | Fixture-only task rehearsal | — |
| `check` | Report local placeholders and node-owned prerequisites | Read-only diagnostics for `--mode software` or `--mode hardware`; `--json` includes issues and next actions | Software mode reads configuration/metadata; default simulation checks resources and runs scene preflight without `--json` |
| `down` | Stop this deployment | Ask this recipe's foreground launcher to stop | Ask this recipe's foreground launcher to stop |

XLeRobot `setup` / `check` default to hardware mode when `--mode` is omitted.
Use `--mode software` explicitly for the first rehearsal. Its `check --json`
exits 0 with `status: "passed"` or 2 with `needs_attention`; resolve the listed
`location` / `next_action` and repeat the same check. The report separates
`verified` from `unverified` so its scope remains visible.

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

If bootstrap `uv sync` or `docker compose build` fails before this launcher
starts, there is no Recipe `Outputs:` / `run.json` yet. Save that command's
terminal stdout/stderr to a chosen log file and record its command and exit code.

SO-101 services remain running after a successful rollout so you can reset the
scene and repeat it. Use `down` when finished. On a rollout failure or interrupt,
the launcher stops the deployment, not just the local HTTP clients. Inspect the
robot and use the physical emergency stop if shutdown cannot be confirmed.
Do not run unrelated work in the example's deployment.

Docker services share a Compose-project `/tmp` named volume for private launcher
sockets. Container startup and `down` must use the same Compose project,
host UID/GID, and configuration so a second container can reach the launcher.
The volume shares launcher IPC, not dependency caches; retain the private
permissions and wait for cleanup confirmation. Hardware stop still needs
physical feedback.

Set `EMBODIRUN_EXAMPLE_PYTHON` to an absolute Host Python path if you are not
using the repository's `.venv`.

For XLeRobot, the separate environment uses
`examples/xlerobot_snack_delivery/pyproject.toml` and `uv.lock` for both software
and full hardware setup. The full owner installs CPU PyTorch; the GPU model
service is prepared separately. The root checkout Host environment includes
the default development group. MicroDuck uses its own Recipe project and lock
under `examples/microduck_vln/` for software and simulation modes.
Do not copy an existing machine's virtual environment.

Container targets `host`, `xlerobot-software`, `xlerobot`, `microduck-software`, and `microduck` are in
the root `Dockerfile`. XLeRobot native and container profiles use the same
Recipe lock; the software container already has its environment. See the
[XLeRobot README](xlerobot_snack_delivery/README.md) for copyable Linux software
commands, and the [installation guide](../docs/en/installation.md) for opt-in
hardware/GPU profiles. Building an image does not enable robot motion.

## Smaller software examples

- [Simulated device and public Control API](shared-device-fake.md)
- [Recorded observations with real inference and simulated execution](shared-device-inference.md)

For latency measurements and report comparison, use
[benchmarks](../benchmarks/README.md).

## First-use feedback

Use the [classroom trial and feedback template](xlerobot_snack_delivery/FIRST_USE_FEEDBACK.md)
for a first software-only trial. The [deployment Agent guide](xlerobot_snack_delivery/AGENT_GUIDE.md)
provides a starting prompt for your existing coding Agent. Record where the
entrypoint was unclear, which local file you needed to edit, what a successful
software run meant, and which missing paths or real information required help.
Actual student feedback has not been collected. Independent Agent rehearsal,
classroom feedback, model validation, and physical delivery are separate results.
