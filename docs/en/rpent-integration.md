# RPent integration

RPent plans tasks and selects skills; EmbodiRun deploys services and executes
actions. The integration connects them through the Control HTTP API.

## Set up the adapter

RPent discovers robots through its top-level `robots/<name>/` packages.
Register `RobotSpec` and `Toolkit` there, using the reference client in
[`agents/rpent`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/agents/rpent/README.md).

The maintained RPent-side adapter lives in the **`BUAA-CI-LAB/RPent` fork** on
the `embodirun-integration` branch: `robots/embodirun/` implements
`get_robot_spec`/`get_toolkit`, connects to the Control HTTP API, and returns
`([], runtime_kwargs)` from
`init_runtime`.

## What the adapter maps

| RPent surface | EmbodiRun Control API |
|---|---|
| robot identity / capabilities | `GET /v1/describe` (including `binding.kind`, `binding.maximum_chunk_steps`, `binding.action_feature_names`) |
| `reset()` | `observe()` only — zero motion |
| `step` / `chunk_step` | one bounded `POST /v1/execute`, then `observe()` |
| `propose` | `POST /v1/propose` against a retained `observation_id` |
| `inspect` / `cancel` / `stop` | `GET /v1/jobs/{id}`, `POST /v1/jobs/{id}/cancel`, `POST /v1/jobs/{id}/stop` |
| task success | observation or task metadata |

Execution semantics are the ones documented in
[`agents/CLIENT.md`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/agents/CLIENT.md):
`observe`/`propose` never move the device; only a bounded `execute` moves;
`unknown` outcomes are surfaced as unknown and never auto-resent; `cancel`
discards late results; an unconfirmed stop persists across a client restart.

## Example: π0.5 with a simulated device

This run connects RPent to Control and a π0.5 inference service. Actions execute
on an in-memory `simulated.policy_vector` device with two synthetic cameras
(`physical_robot=false`).

Conditions:

| Item | Value |
|---|---|
| Inference host | 2×A100 80GB, service on GPU0, `embodiinfer-http` (`π0.5`) |
| Model adapter | `state_native`; cameras `observation.images.front` + `wrist`; `return_steps=50`; 6 action features |
| Control | local `ControlService` from a deployment YAML; external inference endpoint via an SSH tunnel |
| Robot | `simulated.policy_vector` binding `simulated.policy_vector.pi05` |
| RPent | `BUAA-CI-LAB/RPent` `embodirun-integration`, robot `--robot embodirun` |

Observed result (abridged):

```text
describe.binding: {"action_feature_names": ["shoulder_pan.pos", "shoulder_lift.pos",
  "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"],
  "kind": "simulated.policy_vector.pi05", "maximum_chunk_steps": 50}
reset.observation_id: 4090:8:...:o2
reset.task_success: unverified
proposal.status: proposed mapped_actions: 50
fresh observation_id: 4090:8:...:o7
chunk_step 5-tuple -> obs 0.0 False False
info: {"cancelled": false, "discarded_late_result": false,
  "execution_evidence_unknown": false,
  "request_id": "rpent-embodirun:execute:...", "status": "completed",
  "task_success": "unverified"}
inspect.status: completed
```

The adapter refreshes observations before executing a proposal to avoid
`observation_stale` errors. This simulated device reports `task_success: unverified`.

## Configure your application

- **Scene reset.** EmbodiRun has no scene-restoring reset route, so the adapter
  sets `supports_exploration=False` and `reset()` never restores a scene.
- **Real-robot guard.** `RobotSpec.is_real_robot` is frozen per package. The
  fork adapter is simulator-oriented; a real-robot package or an explicit
  safeguard decision is required before driving hardware.
- **Planner.** Select `--robot embodirun`, configure the `--control-*` flags,
  and provide the LLM credential required by your planner.
- **RPent base install.** RPent's base dependency set (for example
  `openai-codex`) must resolve from the index you use; installing from the
  public PyPI works.
