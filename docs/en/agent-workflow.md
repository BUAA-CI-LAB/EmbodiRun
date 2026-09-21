# Agent execution workflow

An agent drives a running deployment through the Host JSON commands
(`describe`, `observe`, `media`, `execute`, `inspect`, `cancel`, `stop`, and the
recording commands). This page describes the contract those commands implement.
The Control API is documented in the
[client reference](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/agents/CLIENT.md);
hardware operation is covered in [Control](control.md) and
[Safety](safety.md).

## Connect through the Host CLI

The Host CLI resolves the endpoint from the deployment YAML and the initialized
Host state, then reaches the Control service over the configured SSH and
loopback channel. Control manages the device connections, shared observations,
and action queue, so the agent can work through one interface.

Control and recorder commands write one JSON value to stdout. Keep `--json` on
and parse it. A self-contained software walkthrough that starts the real
`ControlHttpServer` against simulated adapters is in
[`examples/shared-device-fake.md`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/shared-device-fake.md).
You can run the walkthrough without a robot or model checkpoint.

## Carry a stable identity

Pass a stable `--caller-id` and `--session-id` on every invocation. Ownership,
observation scope and cancellation all resolve against them, so changing either
one mid-workflow detaches the workflow from its own state.

Choose a new `--request-id` for a new action and reuse that exact ID for every
later `inspect`, `cancel` or `stop`. The request ID is the idempotency key: the
service commits a session step once per request ID, so a retry that keeps the ID
cannot execute the action twice.

## Read before acting

`describe` reports the runtime's capabilities and current authority. `observe`
returns one shared observation and an `observation_id`. `media` reads the
encoded frames referenced by that ID, but frame data is opt-in: it is only
returned when `--include-data` is passed. A stale or missing reference returns
an error; request a new observation before continuing.

## Submit one bounded action

`execute` submits one action JSON object, read from a path or from stdin with
`--action -`, and applies the binding's limit checks before anything reaches the
robot:

```sh
printf '%s\n' '{"timestamp_s":0,"values":{"type":"joint_position","joint_positions_deg":[0,0,0,0,0],"gripper_position":25},"metadata":{"action_space":"simulated.so101.position.v1"}}' \
  | embodirun --config "$CONFIG" execute --runtime "$RUNTIME" \
      --caller-id "$CALLER" --session-id "$SESSION" \
      --request-id action-1 --action - --steps 1 --json
```

`--observation-id`, `--max-age-ns` and `--max-skew-ns` bound the observation an
action may be based on. When those are omitted the run uses the latest
shared observation.

## Track execution

An accepted response and a completed response are different states, and
`execute` returns before the action has necessarily finished. If the request
times out, treat the outcome as unknown: keep the original request ID, run
`inspect`, and decide from the reported state. Do not resend the action after a
timeout — resending with a new ID can execute it twice, and resending with the
same ID only re-reads the committed result.

`inspect` is the only way to learn the outcome of an action whose submission was
uncertain. Preserve the returned `status`, `physical_status`, `error` and any
job or media detail when reporting the outcome.

## Cancel or stop a request

`cancel` and `stop` act on one caller/session/request ID. Check the returned
state for confirmation. If the connection is lost, inspect the job after
reconnecting; until then, the stop outcome is unknown.

## Recording is service-owned

Recorder state is queryable without changing ownership. `recording-status` is a
read-only check; use `recording-start`, `recording-stop` and `recording-get`
only when the service description says recording is available. A service without
a configured recorder returns an explicit `unsupported` result instead of
creating a local one, so no client should maintain a second recorder.

## Handle service results

The Control service is authoritative for action limits, stale-input checks,
unsupported capabilities, busy ownership, stop confirmation and recording
state. Preserve results such as `unknown`, `uncertain`, `stop_unconfirmed`,
`stale`, `busy`, and `unsupported` in your application's response. Resolve
them through the Control API or operator intervention, while leaving the
service in control of its device connections.
