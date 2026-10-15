# Independent first use

Use this procedure to check whether a new user or coding Agent can execute the
documented software path, not just understand it. Choose a fresh Linux target
for the recorded trial; the smaller fake-device example also supports laptops.
This page supplies instructions, not a completed independent trial.

## Give the operator these inputs

- Repository and the branch/PR head to test. Check out an unmerged PR's head
  before installation; its commands may not exist on default `main`.
- Authorized Linux target and a new task directory. Start with no copied
  virtual environment or previously initialized deployment state.
- Python 3.12, uv 0.12.x, network/package access, and whether caches are fresh
  or reused. Save installation failures as well as successes.
  A fresh Host `init` also needs Git access to the deployment repository and
  its configured revision, with working TLS trust and access to any uncached
  dependencies. An installed local checkout does not replace this source preparation.
- Scope: the fake-device Host/Control workflow and the XLeRobot fixture
  rehearsal. Neither requires a checkpoint, camera, calibration or robot.
- Optional MicroDuck software scope: configuration, locked base installation
  and environment metadata checks need no external assets or GPU. Full scene
  preflight and episodes additionally require supplied, authorized external
  assets/checkpoint, Linux CUDA and EGL.

An initial prompt for your existing coding Agent:

```text
From the requested repository branch in a new Linux task directory, follow
README and the first-use guide without a pre-existing virtual environment.
Install the documented dependencies; execute the fake-device Host lifecycle,
describe/observe calls, and the XLeRobot software fixture rehearsal. Save the
actual commands, exit codes, outputs and help requests. Do not configure or
operate physical devices. For MicroDuck, execute init, validate, plan,
setup --mode software, and check --mode software --json. Defer simulation
preflight and episodes until the operator supplies and authorizes complete
GPU/asset prerequisites. Report software,
real-model and physical validation separately. A missing input is a result to
record, not permission to replace it with a mock and call the task complete.
```

## Execute the existing paths

1. Install from the selected checkout with `uv sync --frozen --python 3.12`.
   Preserve stdout/stderr if installation fails before any Recipe output exists.
   `uv sync` creates `.venv` without adding its executables to your shell's
   `PATH`. Recipe hints use the caller environment's absolute `embodirun` path
   when available, along with the checkout and configuration paths. Copy the
   complete printed command to preserve that environment. If no console script
   is available, the hint falls back to `uv run --frozen --project CHECKOUT`.
2. Follow [Quick start, no robot required](quickstart.md#1-no-robot-required):
   `validate` → `init` → `sync --source .` → `up` → `describe` → `observe`.
   This starts the actual Control service with in-memory joints and a fake camera.
   Keep its task-owned `--root` and `--state-dir`, check port 8100 is free before
   `init`/`up`, and carry the same `STATE` into every Agent command.
   `init` prepares managed sources using `metadata.deploy-commit` before
   `sync --source .` overlays the local checkout. If Git or package access
   fails, preserve the command, exit code and stdout/stderr before retrying;
   continue to `up` only after initialization succeeds.
   For this local `simulated.joints` trial, pass `UV_PYTHON=3.12` to `init`,
   or use a verified absolute Python 3.12 interpreter path. The CLI's Python
   3.12 bootstrap does not select the managed Host interpreter automatically.
   Check and record the managed environment's actual `bin/python --version`,
   as shown in Quick start.
   Defer Quick start's final `down` until the optional Agent calls below finish.
3. Follow [Agent workflow](agent-workflow.md) if inspecting or executing one
   bounded simulated action. Preserve caller/session/request IDs and inspect
   an uncertain submission. Keep the runtime `fake-device` for this trial.
   Then run `uv run --frozen embodirun --state-dir "$STATE" --config "$CONFIG" down`
   and confirm this service has exited and its endpoint is closed. If skipping
   Agent actions, stop immediately after the observation; avoid an extra restart.
4. Follow [Recipes](examples.md) or the
   [XLeRobot software rehearsal](xlerobot-snack-delivery.md): initialize its
   linked local directory, `validate`, `plan`, `setup --mode software`,
   `check --mode software --json`, then `dry-run`. Use a new output directory
   if `init` reports that the default directory already exists.
   XLeRobot's `plan` previews the manifest's hardware `run` command. The
   software rehearsal uses the explicit software setup/check and `dry-run`
   commands above.
5. Follow [MicroDuck](microduck-vln.md): `init` → `validate` → `plan` →
   `setup --mode software` → `check --mode software --json` (plain software
   `check` is also supported). Record configuration and installed metadata
   separately from the unverified CUDA/EGL, assets and model execution.
   `plan` previews simulation; MicroDuck has no fixture `dry-run`.
   The default setup/check mode is `simulation`. Its JSON check reads installed
   metadata and local resource/source paths; plain simulation `check` performs
   scene preflight. Use those only after supplying the simulation environment,
   compatible checkpoint, episodes, inventory and scene assets. A passed
   preflight does not establish model inference or navigation success.
6. Record the printed output directories, `run.json`, command/service logs and
   shutdown outcome. Recipe [output guidance](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/README.md#outputs-and-shutdown)
   explains which outputs exist before or after launcher startup.

## Record the first-use result

Keep this record with the actual terminal transcript:

| Field | Fill after execution |
|---|---|
| Starting conditions | Checkout revision/local changes, target OS/architecture, caller and managed Python/uv, new environment/state and cache state |
| Documents followed | Pages/sections actually used, in order |
| Commands and outcomes | Exact command, working directory, exit code and log/output location |
| Friction | Unclear entrypoint, missing input, file edited, dependency failure and the help required |
| Software result | Host observation and stopped-service evidence; XLeRobot fixture result; MicroDuck configuration/metadata result |
| Model/physical result | `not_run` and reason when assets/devices were not supplied; never infer from software output |
| Independence | Who executed the commands and what prior environment or author assistance they used |

A documentation-only review is useful feedback but is not independent first
use. A cached install is not a fresh download, a fixture rehearsal is not a
learned-model trial, and a successful software stop is not physical stop evidence.
If a maintainer supplies a source mirror, predownloaded packages or configuration
repairs to complete a retry, retain the original failure and record the source
revision, cache and assistance. Report it as an assisted trial rather than
independent first-use acceptance or an uncached end-to-end installation.
