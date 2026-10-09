# XLeRobot snack delivery

Rehearse a delivery without devices, then prepare your own XLeRobot for a
supervised run. The task combines recorded base routes, RPent/Astra scene
review, π0.5 grasp proposals, and a calibrated handover. EmbodiRun coordinates
the task through Control; the robot owner holds the motor and camera connections.

| Your goal | Start here | What you need |
|---|---|---|
| Learn the task or try the deployment instructions | [Software rehearsal](#software-rehearsal) | A checkout and either Python/uv or Docker; no robot or GPU |
| Prepare your own cart | [Prepare your robot](#prepare-your-robot) | Its Linux host, SDK, calibration, cameras, routes, and a compatible model service |
| Let a coding Agent help with deployment | [Agent guide](AGENT_GUIDE.md) | The target host and which of the two goals you want |

This recipe remains experimental. A rehearsal tests software sequencing;
model quality and physical delivery need separate evidence on your setup.

## Software rehearsal

Run these commands in the checkout on the target Linux host. Native setup uses
Python 3.12 and uv 0.12.x. Network access is needed for the first installation.

```bash
uv sync --frozen --python 3.12
uv run --frozen embodirun example init xlerobot
CONFIG=examples/local/xlerobot/example.local.yaml
uv run --frozen embodirun example "$CONFIG" validate
uv run --frozen embodirun example "$CONFIG" plan
uv run --frozen embodirun example "$CONFIG" setup --mode software
uv run --frozen embodirun example "$CONFIG" check --mode software --json
uv run --frozen embodirun example "$CONFIG" dry-run
```

`init` creates linked local files under `examples/local/xlerobot/`. If that
directory already exists, it exits with code 2 and shows how to reuse its
`example.local.yaml`; use that path or choose a new `--output` directory.
Your edited files are not overwritten.

`validate` checks configuration and `plan` prints the proposed command and
output path. `setup --mode software` prepares a small environment without the
owner or PyTorch. `check --mode software --json` should exit 0 with
`status: "passed"`. The dry-run uses route and feedback fixtures and starts no
owner, inference service, or RPent.

The checkout's initial `uv sync` also installs its default development group;
the separate recipe software environment contains the smaller runtime dependencies.

For a dry-run, the printed `Outputs:` directory contains `run.json`, `command-0.log`, and
`result/status.json` / `result/events.jsonl`. A completed fixture run keeps
`task_success: "unverified"` and `physical_success: null`.

CI installs the software Recipe from its frozen lock, builds the production
software image in a fresh Docker builder and runs both public entrypoints as
the caller UID. It checks printed commands outside the checkout with space-containing
paths, failure diagnostics, fixture results, launcher cleanup and native/container
package parity. To repeat the installed-environment checks:

```bash
"$PWD/.venv-xlerobot-snack/bin/python" scripts/check_recipe_software.py --recipe xlerobot
```

Independent human first use, model compatibility and physical delivery remain
separate follow-up acceptance work.

### Docker alternative

Use Docker Engine and Compose on the target Linux host. This path needs no
device mapping, NVIDIA runtime, SDK, checkpoint, or host Python environment:

```bash
mkdir -p examples/local
docker compose build xlerobot-software
docker compose run --rm --user "$(id -u):$(id -g)" xlerobot-software init xlerobot --output /workspace/xlerobot
docker compose run --rm --user "$(id -u):$(id -g)" xlerobot-software /workspace/xlerobot/example.local.yaml validate
docker compose run --rm --user "$(id -u):$(id -g)" xlerobot-software /workspace/xlerobot/example.local.yaml plan
docker compose run --rm --user "$(id -u):$(id -g)" xlerobot-software /workspace/xlerobot/example.local.yaml check --mode software --json
docker compose run --rm --user "$(id -u):$(id -g)" xlerobot-software /workspace/xlerobot/example.local.yaml dry-run
```

The image already contains the software environment, so skip native `setup`.
`/workspace/xlerobot` maps to `examples/local/xlerobot` in your checkout;
generated files belong to your host user. Both this image and native XLeRobot
setup use `examples/xlerobot_snack_delivery/uv.lock`; the hardware profile adds
the owner dependencies from that same project. See the
[installation guide](../../docs/en/installation.md) for dependency sources and
the hardware and GPU profiles. Linux ARM64 compatibility depends on the selected
integration and must be checked on the target; this software path does not test
a GPU or devices.

If you reuse a native configuration in Docker, check the manifest's top-level
`python` field first: it takes priority over the image's interpreter setting.
A host virtual-environment path is not available inside the container.
Back up `example.local.yaml`, then remove only its `python` override from the
Docker copy so the image selects `/opt/venv/bin/python`; keep all other settings.
Keep that copy in the original configuration directory, for example as
`example.docker.yaml`, so relative deployment/task paths still resolve. In the
Docker commands above, replace the manifest argument with
`/workspace/xlerobot/example.docker.yaml`.
Use the native backup when returning to native execution. Alternatively,
`init` into a new directory such as `/workspace/xlerobot-docker` and migrate
your deployment, hardware, task, and route settings with their path references.
Keep the original configuration and calibration files.

For container commands, use the same Compose project and host UID/GID each time.
Run from the same checkout, or pass the same explicit `-p NAME` for each command.
The services share a project-local `/tmp` volume for the launcher's private
control socket, enabling a later container's `down` to reach the active launcher.
This volume is launcher IPC storage; it does not cache dependencies. Wait for
cleanup confirmation and, for hardware, verify physical stop feedback.

### If a step fails

For `check`, exit code 2 means there is work to do. Read each issue's `location`,
`message`, and `next_action`, fix that input, then repeat the same check.
`verified` and `unverified` describe the scope of a successful check.

If `uv sync` or `docker compose build` fails before the Recipe CLI starts,
there is no Recipe `Outputs:` directory or `run.json` yet. Save that command's
terminal stdout/stderr to a log file of your choice, with the exact command and
exit code. For Recipe `setup` failures that printed `Outputs:`, retain its
`run.json` and `command-0.log` there. Check the first error before retrying the
same setup mode. If it reports a uv HTTP read timeout, a native retry can use
`UV_HTTP_TIMEOUT=300` before the command. Record whether downloads were cached;
successful retries alone do not establish cold-install time.

## Prepare your robot

Continue with the same generated directory and
[setup and calibration](guide.md). Prepare real values before running the
hardware check:

| File | Fill in for your setup |
|---|---|
| `deployment.local.yaml` | Model endpoint, Control ports, and camera feature-to-owner-role mapping |
| `hardware.local.json` | SDK/device paths, verified calibration, wheel geometry/direction, cameras, and motion limits |
| `config.local.json` and `routes/` | Your recorded directed routes, grasp instruction, planner factory, and calibrated handover pose |

The supplied zero-motion routes are fixtures and are rejected for hardware.
Managed recipe runs derive Control endpoints from `deployment.local.yaml`;
route paths remain relative to the task JSON. Start inference separately with
a π0.5 checkpoint adapted to `lerobot.xlerobot.pi05`, its two-arm feature names,
units, and camera inputs. Installing the owner uses CPU PyTorch; it does not
install or validate the GPU model service.

```bash
uv run --frozen embodirun example "$CONFIG" setup --mode hardware
uv run --frozen embodirun example "$CONFIG" check --mode hardware --json
```

Both commands default to hardware mode when `--mode` is omitted. Hardware
`check` reads local files and declarations. Before enabling motion, the
operator must verify actual calibration, camera images, model compatibility,
and physical stop feedback. Keep `allow_motion: false` during preparation;
the guide explains the operator's next steps.

## Prepare routes and recordings

Record outbound and return routes through the owner's teleoperation interface,
then export bounded chunks at the configured playback rate. Follow
[route input choices](guide.md#route-input-choices) for formats and export commands.

To save camera and state observations, configure a Control recorder and use the
[recording API](../../docs/en/agent-workflow.md). Supply a compatible π0.5
checkpoint through the separately running model service; use your existing
tools for dataset preparation and training.

The delivery run also saves events, review decisions, and VLA proposals.
See the [workflow and service diagram](../../docs/en/demos/xlerobot-snack-delivery.md)
for how route preparation, recording, deployment, and execution fit together.

## Run a supervised delivery

On the robot's Linux host, after those preparations and operator authorization:

```bash
CONFIG=examples/local/xlerobot/example.local.yaml
uv run --frozen embodirun example "$CONFIG" up --allow-hardware
```

Keep `up` running in this terminal. It starts one owner and two scoped Control
services. Confirm the physical stop through the owner UI, then in a second
terminal run:

```bash
CONFIG=examples/local/xlerobot/example.local.yaml
uv run --frozen embodirun example "$CONFIG" run --allow-hardware
```

Follow the prompts for arrival, grasp, and handover. Hardware confirmation is
never automated. Use `uv run --frozen embodirun example "$CONFIG" down` from another terminal
or Ctrl-C in the `up` terminal to stop its service stack; check
the physical stop state before leaving the robot.

The generated example's output root is `examples/local/xlerobot/runs/`.
`services/` holds private owner/Control configs and service logs; timestamped
task directories contain `run.json` and `result/` with events, status, agent
decisions, and proposals. Keep generated credentials local.

## Reference

- [Setup and calibration](guide.md)
- [Hardware and software](hardware.md)
- [Agent and runtime interfaces](architecture.md)
- [Deployment Agent instructions](AGENT_GUIDE.md)
- [Classroom first-use trial and feedback](FIRST_USE_FEEDBACK.md)
- [Shared manifest and launcher](../README.md)
