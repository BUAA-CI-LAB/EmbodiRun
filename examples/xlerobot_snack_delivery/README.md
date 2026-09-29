# XLeRobot snack delivery

Connect base routes, an RPent/Astra scene review, VLA grasp proposals, and a
supervised handover through EmbodiRun. The robot owner holds the serial and
camera connections; the example coordinates tasks through Control.

## Start without hardware

From the repository root, after `uv sync --frozen`:

```bash
uv run --frozen embodirun example init xlerobot
CONFIG=examples/local/xlerobot/example.local.yaml
uv run --frozen embodirun example "$CONFIG" validate
uv run --frozen embodirun example "$CONFIG" plan
uv run --frozen embodirun example "$CONFIG" dry-run
```

The dry-run exercises task sequencing with fixture routes and feedback. It
does not start the owner, inference, or RPent. Inspect `result/status.json`
and `result/events.jsonl` in the printed output directory.

## Prepare your robot

Follow [setup and calibration](guide.md). The `init` command above already
created the manifest, task, deployment, hardware file and both route files in
one directory, with references connected. It refuses to overwrite existing
local work. Edit the generated files:

- In the deployment, set the running model service's endpoint and camera roles.
- In the hardware config, set SDK/device paths, calibration, and motion limits.
  Confirm camera roles and stop feedback before enabling motion.
- In the task JSON, set your recorded routes, grasp instruction, agent factory,
  and calibrated handover pose. Route paths resolve relative to that JSON.
  The supplied zero-motion routes are fixtures and are rejected in hardware mode.

The GPU inference service is started separately; use a π0.5 checkpoint adapted
to XLeRobot's two-arm feature names and units. The [guide](guide.md) describes
route export, owner setup, and agent configuration.

## Run a supervised delivery

```bash
CONFIG=examples/local/xlerobot/example.local.yaml
uv run --frozen embodirun example "$CONFIG" setup
uv run --frozen embodirun example "$CONFIG" check
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
never automated. Use `embodirun example "$CONFIG" down` from another terminal
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
- [Shared manifest and launcher](../README.md)
