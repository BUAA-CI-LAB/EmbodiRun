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
- Scope: the fake-device Host/Control workflow and the XLeRobot fixture
  rehearsal. Neither requires a checkpoint, camera, calibration or robot.
- Optional MicroDuck scope, only if its external assets/checkpoint, Linux CUDA
  and EGL are separately supplied and authorized. Otherwise stop at offline
  configuration/command preview and report the missing inputs.

An initial prompt for your existing coding Agent:

```text
From the requested repository branch in a new Linux task directory, follow
README and the first-use guide without a pre-existing virtual environment.
Install the documented dependencies; execute the fake-device Host lifecycle,
describe/observe calls, and the XLeRobot software fixture rehearsal. Save the
actual commands, exit codes, outputs and help requests. Do not configure or
operate physical devices. For MicroDuck, only validate/plan until the operator
supplies and authorizes the complete GPU/asset prerequisites. Report software,
real-model and physical validation separately. A missing input is a result to
record, not permission to replace it with a mock and call the task complete.
```

## Execute the existing paths

1. Install from the selected checkout with `uv sync --frozen --python 3.12`.
   Preserve stdout/stderr if installation fails before any Recipe output exists.
2. Follow [Quick start, no robot required](quickstart.md#1-no-robot-required):
   `validate` → `init` → `sync --source .` → `up` → `describe` → `observe`.
   This starts the actual Control service with in-memory joints and a fake camera.
   Keep its task-owned `--root` and `--state-dir`, check port 8100 is free before
   `init`/`up`, and carry the same `STATE` into every Agent command.
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
5. Follow [MicroDuck](microduck-vln.md) only within the supplied scope.
   Its current `check` without `--json` is a full CUDA/EGL/assets scene preflight.
   `check --json` only checks local paths and cannot establish GPU readiness.
   A passed scene preflight still does not establish model
   inference or navigation success. Do not add `--mode software` to MicroDuck:
   that option is currently available only to the XLeRobot Recipe.
6. Record the printed output directories, `run.json`, command/service logs and
   shutdown outcome. Recipe [output guidance](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/README.md#outputs-and-shutdown)
   explains which outputs exist before or after launcher startup.

## Record the first-use result

Keep this record with the actual terminal transcript:

| Field | Fill after execution |
|---|---|
| Starting conditions | Checkout revision/local changes, target OS/architecture, Python/uv, new environment/state and cache state |
| Documents followed | Pages/sections actually used, in order |
| Commands and outcomes | Exact command, working directory, exit code and log/output location |
| Friction | Unclear entrypoint, missing input, file edited, dependency failure and the help required |
| Software result | Host observation and stopped-service evidence; fixture rehearsal result |
| Model/physical result | `not_run` and reason when assets/devices were not supplied; never infer from software output |
| Independence | Who executed the commands and what prior environment or author assistance they used |

A documentation-only review is useful feedback but is not independent first
use. A cached install is not a fresh download, a fixture rehearsal is not a
learned-model trial, and a successful software stop is not physical stop evidence.
