# Safety

EmbodiRun moves physical robots. This document describes the software
protections and the operator's responsibilities. It is not a substitute for a
hardware emergency stop, a risk assessment, or the robot vendor's safety
manual.

## Scope

EmbodiRun provides:

- per-row joint and gripper step limits (`step_limit_mode: clip` or `reject`),
- bounded action chunks (the model's chunk bound is the stopping condition),
- control arbitration between model actions and manual input,
- manual takeover and a latching software emergency stop,
- fail-closed execution: inference failure does not submit actions.

EmbodiRun does **not** provide collision avoidance, workspace monitoring,
force limiting, or a certified functional-safety path. Action clipping is a
rate limit, not obstacle avoidance.

## Software stop versus hardware stop

The software emergency stop latches outside the motion queue, cancels active
work, and clears pending actions. It runs in the Control process. An ARX5 or
SO-101 adapter serializes stop with the current SDK call, so a blocked SDK call
can delay the physical stop. The FR3 adapter permits concurrent stop.

Always keep the hardware emergency stop within reach. The software stop is an
additional layer, not a replacement.

## Authority and lifecycle

- Each Control service owns one robot and one arbiter.
- Model actions use a bounded FIFO; manual actions have one pending slot
  (latest wins).
- Acquiring manual control cancels model work, including inference results that
  arrive after takeover.
- Releasing manual control allows a new model task; it never resumes a
  cancelled one.
- Task completion holds the robot through its adapter. It does not disconnect
  it or plan a return-to-home trajectory.
- Service shutdown releases the hardware.

## Operator checklist

Before running a task:

1. Read this file and [`control.md`](control.md).
2. Confirm the hardware emergency stop and keep it reachable.
3. Confirm the workspace is clear and no person is inside the motion range.
4. Verify calibration, camera roles, and `step_limit_mode` in the YAML.
5. Confirm the checkpoint's state/action dimensions match the binding.
6. Run `validate`, then `probe`, then `init`, then `up`, with an operator
   present.
7. Start with `--max-steps 1` and a small `--chunk-steps`.

During a task:

- Watch the robot, not only the terminal.
- Be ready to use manual takeover or the software stop.
- Treat an accepted request as "in progress", never as task success.

## Failure semantics

- Inference failure → no actions are submitted.
- Unknown execution result → the request is reported unknown and is not
  re-sent automatically.
- Unconfirmed stop → never reported as success.
- Cancel → late inference results are discarded, not executed.
- Emergency stop → pending model and manual actions are cleared.

## Reporting a safety issue

Open an issue with the configuration shape (redact addresses and secrets), the
command, the observed behavior, and whether a physical stop was used. Do not
include checkpoint paths, tokens, or personal data.
