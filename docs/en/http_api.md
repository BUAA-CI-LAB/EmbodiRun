# Inference API v1

This is the contract between EmbodiRun and the EmbodiInfer policy service.
Requests carry encoded images, robot state, a task instruction, and session
identifiers. Responses contain model-native actions; the runtime's binding
maps them into device commands.

The Control API used by agents is a separate interface. For observation,
proposal, execution, and job inspection, see [Agent execution workflow](agent-workflow.md)
and the [public client reference](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/agents/CLIENT.md).

## Control and inference sessions

An agent calls Control's `/v1/propose` to obtain a non-executing proposal from
a retained observation. Control invokes the inference service and applies the
runtime binding; the agent later submits an execution request to Control.
The request includes the task instruction as text. The current
proposal path uses a temporary inference session. Stateful navigation requires
a path that retains the model session across steps.

## HTTP endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz` | Check service health. |
| GET | `/v1/capabilities` | Inspect adapter capabilities and action space. |
| POST | `/v1/sessions` | Open a policy session. |
| POST | `/v1/sessions/{session_id}/steps` | Infer one ordered step. |
| POST | `/v1/sessions/{session_id}/reset` | Clear session state and restart step numbering. |
| DELETE | `/v1/sessions/{session_id}` | Close the session. |

When authentication is configured, send `Authorization: Bearer <token>` on
requests other than the health check. Server setup and error semantics are
documented in [EmbodiInfer Serving](https://embodiinfer.readthedocs.io/en/latest/serving/).

Open a session with JSON containing `schema: embodiinfer.policy.session.v1`, a
`robot_id`, the adapter's `action_space`, and optional `metadata`. Use the
returned `session_id` for subsequent operations.

## Session ordering

Each step carries a stable `session_id`, monotonic `step_id`, and unique
`request_id`. The server must make repeated request IDs idempotent and commit
session state only once.

## HTTP step request

`POST /v1/sessions/{session_id}/steps` uses multipart form data.

- `metadata`: `application/json`, schema `embodiinfer.policy.step.v1`
- `image_0..N`: encoded JPEG/PNG bytes

Send `Idempotency-Key` equal to the metadata's `request_id`. The metadata
contains `session_id`, `request_id`, a non-negative `step_id`, a non-empty
`instruction`, a `state` object, an `images` list, and optional `metadata`.
Each image declaration has `name` and `mime_type`; its position corresponds
to `image_0`, `image_1`, and so on. The body and URL session IDs must match.

Start with `step_id: 0`. After a successful step, increment it by one. On a
transport failure, retry the same request with the same ID and content;
do not advance the counter on an unconfirmed result. Cached responses are
bounded, so an old retry can be rejected after its response has been evicted.

## Step response

```json
{
  "request_id": "step-0-...",
  "session_id": "fr3-episode-1",
  "step_id": 0,
  "session_revision": 1,
  "action_space": "pi05.action_chunk.v1",
  "actions": [
    {
      "type": "action_chunk",
      "values": {
        "data": [
          [0.12, -0.03, 0.45, -2.21, 0.08, 1.12, -0.34, 0.04]
        ],
        "feature_names": [
          "joint_1",
          "joint_2",
          "joint_3",
          "joint_4",
          "joint_5",
          "joint_6",
          "joint_7",
          "gripper_width_m"
        ]
      }
    }
  ],
  "timing": {
    "policy_ms": 61.0
  }
}
```

EmbodiRun maps and validates the rows in `pi05.action_chunk.v1` through the
selected binding, executes the `run --chunk-steps` prefix, and plays those robot
commands at the runtime control rate. The binding defines robot state fields
and action dimensions.

## WirelessComm mapping

WirelessComm uses the same `embodiinfer.policy.session.v1`, `embodiinfer.policy.step.v1` and
result schemas. RPC metadata uses schema `embodiinfer.policy.rpc.v1`, a per-attempt
`rpc_id`, and one of the methods `health`, `capabilities`, `open_session`,
`step`, `reset` or `close`.

For step calls, metadata fields and each encoded image are sent as one structured
payload. Image bytes are native WirelessComm byte segments; they are not encoded
as base64 or assembled into HTTP multipart data. The policy `request_id` remains
the semantic idempotency key and is distinct from the per-attempt `rpc_id`.
