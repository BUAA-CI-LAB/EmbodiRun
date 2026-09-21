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

The default route files are marked as fixtures and are accepted only by
`dry-run`. For a real scene, record directed route chunks in the target space
or ask RPent to select from locally recorded segments using a route diagram.
Route diagrams help select recorded segments; motion comes from the calibrated
route files.

## Rehearse without hardware

After `uv sync --frozen`, rehearse the task through the shared entrypoint:

```bash
bash examples/run.sh examples/xlerobot_snack_delivery/example.yaml validate
bash examples/run.sh examples/xlerobot_snack_delivery/example.yaml dry-run
```

The package README covers local YAML setup, service startup, supervised
execution, and output files. See [Reproduce the demos](examples.md) for common
commands.

For hardware, follow the owner and safety instructions in the setup guide,
keep an operator present, and validate the robot-specific calibration and
emergency stop before allowing motion. Record routes and calibrate the arm and
cameras for your own workspace.
