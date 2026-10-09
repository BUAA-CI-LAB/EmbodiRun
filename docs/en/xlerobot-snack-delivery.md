# XLeRobot snack delivery

In this mobile manipulation example,
an XLeRobot follows a route to a pickup table, asks an upper-layer RPent/Astra
agent to review the current camera view, obtains a bounded π0.5/VLA proposal,
returns on a second route, and presents the item with a calibrated arm pose.

[Watch the snack-delivery demo](demos/xlerobot-snack-delivery.md) to see the
approach, grasp, and return in action.

The task uses four components:

- **EmbodiRun** handles deployment, shared observations, action validation,
  freshness checks, bounded execution, stop requests, and feedback.
- **The XLeRobot owner** manages motor and camera connections, and reports
  observations, control status, and stop confirmation over HTTP.
- **RPent/Astra** optionally reviews observations and selects task stages.
  You can supply another planner through the same adapter interface.
- **The VLA service** generates grasping actions from camera images and state.

## Start from the example directory

The configuration, launcher, and task code are under
[`examples/xlerobot_snack_delivery`](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/examples/xlerobot_snack_delivery):

| Need | Document |
|---|---|
| Scene, components, and agent/model split | [`README.md`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/xlerobot_snack_delivery/README.md) |
| Hardware bill of materials and planning prices | [`hardware.md`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/xlerobot_snack_delivery/hardware.md) |
| Setup, calibration, routes, and supervised execution | [`guide.md`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/xlerobot_snack_delivery/guide.md) |
| Interfaces and execution feedback | [`architecture.md`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/xlerobot_snack_delivery/architecture.md) |
| Deployment fields | [`deployment.example.yaml`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/xlerobot_snack_delivery/deployment.example.yaml), [`config.example.json`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/xlerobot_snack_delivery/config.example.json) |
| Unified setup and launch | [`example.yaml`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/xlerobot_snack_delivery/example.yaml), [`examples/run.sh`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/run.sh) |
| Deployment help from your coding Agent | [`AGENT_GUIDE.md`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/xlerobot_snack_delivery/AGENT_GUIDE.md) |
| Classroom trial and first-use feedback | [`FIRST_USE_FEEDBACK.md`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/xlerobot_snack_delivery/FIRST_USE_FEEDBACK.md) |

The default route files are marked as fixtures and are accepted only by
`dry-run`. For a real scene, record directed route chunks in the target space
or ask RPent to select from locally recorded segments using a route diagram.
Route diagrams help select recorded segments; motion comes from the calibrated
route files.

## Rehearse without hardware

Choose native setup or the [Docker software path](installation.md#managed-deployments).
Both need a checkout on the target Linux host and network access for the first
installation. Native XLeRobot setup uses Python 3.12 and uv 0.12.x:

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

No real SDK, calibration, cameras, route recording, checkpoint, model endpoint,
RPent, or GPU is required. If `init` finds an existing directory, reuse the
configuration path it prints or choose a new `--output`; it does not overwrite
local work. Docker uses `xlerobot-software` and already contains the software
environment, so it skips native `setup`. Its dependency source is the same
recipe project and lock as native XLeRobot setup.

`check` exits 0 with `status: "passed"` or 2 with issues to resolve. Read each
issue's `location` and `next_action`, then repeat the same check. The printed
output directory contains `run.json`, `command-0.log`, and
`result/status.json` / `result/events.jsonl`. A completed dry-run records
fixture sequencing, with `task_success: "unverified"` and
`physical_success: null`; it starts no owner, inference service, or task Agent.

## Prepare your own cart

Continue in the generated local directory:

| File | Your inputs |
|---|---|
| `deployment.local.yaml` | Model endpoint, Control ports, camera feature-to-owner-role mapping |
| `hardware.local.json` | Installed SDK path, devices, verified calibration, wheel parameters, limits |
| `config.local.json` and `routes/` | Recorded directed routes, grasp instruction, planner, calibrated handover |

Use `setup --mode hardware` for the complete owner environment, then
`check --mode hardware --json` for local prerequisites. These commands default
to hardware mode when `--mode` is omitted. See the
[support matrix](support-matrix.md#xlerobot-recipe-prerequisites) for the
two-arm `lerobot.xlerobot.pi05` binding, units, cameras, and required inputs.

Static checks do not establish live readiness. Follow the setup guide and keep
an operator present to verify calibration, camera images, model compatibility,
and fresh stop feedback before authorizing motion. Inference runs
separately; installing the owner does not supply a checkpoint. Use the local
README for supervised `up` / `run` and shutdown. Classroom feedback materials
are provided, but actual student feedback still needs to be collected.
