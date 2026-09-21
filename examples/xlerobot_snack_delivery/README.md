# XLeRobot snack delivery

Connect base routes, an RPent/Astra scene review, VLA grasp proposals, and a
supervised handover through EmbodiRun. The robot owner holds the serial and
camera connections; the example coordinates tasks through Control.

## Start without hardware

From the repository root, after `uv sync --frozen`:

```bash
CONFIG=examples/xlerobot_snack_delivery/example.yaml
bash examples/run.sh "$CONFIG" validate
bash examples/run.sh "$CONFIG" plan
bash examples/run.sh "$CONFIG" dry-run
```

The dry-run exercises task sequencing with fixture routes and feedback. It
does not start the owner, inference, or RPent. Inspect `result/status.json`
and `result/events.jsonl` in the printed output directory.

## Prepare your robot

Follow [setup and calibration](guide.md), then copy the local configuration:

```bash
cp examples/xlerobot_snack_delivery/example.yaml examples/xlerobot_snack_delivery/example.local.yaml
cp examples/xlerobot_snack_delivery/config.example.json examples/xlerobot_snack_delivery/config.local.json
cp examples/xlerobot_snack_delivery/deployment.example.yaml examples/xlerobot_snack_delivery/deployment.local.yaml
cp examples/xlerobot_snack_delivery/hardware.example.json examples/xlerobot_snack_delivery/hardware.local.json
```

Edit the local files:

- In the manifest, point `parameters.config` to `config.local.json` and
  `parameters.deployment` to `deployment.local.yaml`. Set
  `python: ../../.venv-xlerobot-snack/bin/python` for the hardware environment.
- In the deployment, set `owner.hardware_config: hardware.local.json` and the
  running model service's endpoint and camera roles.
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
CONFIG=examples/xlerobot_snack_delivery/example.local.yaml
bash examples/run.sh "$CONFIG" setup
bash examples/run.sh "$CONFIG" validate
bash examples/run.sh "$CONFIG" up --allow-hardware
```

Keep `up` running in this terminal. It starts one owner and two scoped Control
services. Confirm the physical stop through the owner UI, then in a second
terminal run:

```bash
bash examples/run.sh examples/xlerobot_snack_delivery/example.local.yaml run --allow-hardware
```

Follow the prompts for arrival, grasp, and handover. Hardware confirmation is
never automated. Ctrl-C in the `up` terminal stops its service stack; check
the physical stop state before leaving the robot.

The default output root is `artifacts/examples/xlerobot-snack-delivery/`.
`services/` holds private owner/Control configs and service logs; timestamped
task directories contain `run.json` and `result/` with events, status, agent
decisions, and proposals. Keep generated credentials local.

## Reference

- [Setup and calibration](guide.md)
- [Hardware and software](hardware.md)
- [Agent and runtime interfaces](architecture.md)
- [Shared manifest and launcher](../README.md)
