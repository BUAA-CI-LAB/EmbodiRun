# RPent + ActiveVLN navigation

RPent observes, tracks route progress, chooses bounded ActiveVLN subgoals and
decides when to replan or finish. ActiveVLN runs through the existing Go2 HTTP
policy binding. The Go2 backend reuses the camera and motion-lease clients;
Habitat uses the same policy and skill lifecycle.

| Tool | Behavior |
| --- | --- |
| `observe` | Fresh RGB and consumed execution budget |
| `navigate` | Run the local visual loop until an event or bounded checkpoint; optionally save the plan in the same call |
| `finish` | Optionally save the plan and end the task; the success claim remains unverified |

The default `event` profile returns on local STOP, repeated collisions, visually
unchanged or repeating views, exhausted budget, or a checkpoint after at most
32 chunks. ActiveVLN still receives a fresh full-resolution image before every
chunk. Visual events are watchdog signals, not verified localization or semantic
completion. The Go2 backend has no collision telemetry, so it uses visual events.
Planner feedback alone is encoded as JPEG quality 80 at the original resolution;
the remote executor compresses it before sending it over SSH. Plan updates are
optional `plan={completed,remaining,observations}` arguments to `navigate`/`finish`.
The first frame is attached directly to the task, avoiding an initial `observe`
tool round. Its acquisition is timed separately and included in episode time.
`--profile legacy` retains the four-tool, 1–8 chunk interface for comparison.

Repeating the same subgoal preserves model history. Changing it, explicitly
restarting, or continuing after a local STOP opens a new ActiveVLN session.
RPent retains task progress across those sessions. A local STOP does not finish
the overall task. Go2 releases its own motion lease before waiting for RPent.
This entrypoint uses synchronous chunks even when the deployment configuration
defaults to overlap.

From the repository root, install the optional planner environment:

```bash
uv venv --python 3.11 .venv-rpent
uv pip install --python .venv-rpent/bin/python -e . \
  -r agents/rpent/navigation/pyproject.toml
```

RPent is pinned to `7f403130`; the Codex Python SDK is pinned to `0.163.0`.
The Go2 entrypoint defaults to `shared-model`: the existing navigation endpoint
must expose `shared_text` with the same model-instance identity. This mode uses
RPent's tool and memory infrastructure without a separate planner model or Codex
login. A missing or mismatched capability fails before opening hardware clients;
there is no provider fallback. Shared-model serving and model capability remain
experimental; see the endpoint requirements below.

Read-only Go2 planning against **already deployed** services:

```bash
PYTHONPATH=. \
.venv-rpent/bin/python -m agents.rpent.navigation.go2 \
  --config /path/to/go2-navigation.json \
  --instruction 'Leave the room, turn left into the hallway, and stop inside the study.' \
  --planner-backend shared-model --output /path/to/new-run
```

Motion additionally requires `--execute` and an already armed owner supporting
scoped lease release. Endpoint/token environment variables and timing bounds
come from `NavigationConfig`. Without `--execute`, the shared planner receives
only `observe` and `finish`; success claims are rejected as unexecuted. With
execution enabled it receives `navigate_task`, `navigate`, `advance_goal` and
`finish`, including fixed tool-format examples. `--max-new-tokens` bounds text
output (default 512). Use the event profile and omit `--planner-model` in this mode.
Planner images must fit the shared service's 307200-pixel budget; this entrypoint
does not resize them. This new planner layer has not been tested on a
physical Go2. Its simulation comparison is documented in
[`benchmarks/rpent-navigation`](../../../benchmarks/rpent-navigation/README.md).
The optimized profile has a separate [comparison](../../../benchmarks/rpent-navigation/optimization.md).

The earlier cloud planner remains available only with explicit
`--planner-backend codex --planner-model MODEL`. That backend requires an
authenticated Codex CLI (`codex login status`; validated CLI version `0.164.0`)
and uses RPent's `CodexPlanner`/MCP transport. Shell, web search and generic file
tools are disabled; credentials remain on the planner host.

中文说明见 [RPent 导航集成](../../../docs/zh/rpent-navigation.md)。

### Experimental shared-model planner

`SharedModelPlanner` implements RPent's `Planner.solve` contract using the same
resident model as navigation. It reuses `NavigationToolkit`, episode memory and
tool dispatch. Its client requires the navigation endpoint to advertise
`vvla.shared_text.v1` with a matching model-instance identity. In an EmbodiInfer
build that includes `ActiveVLNSharedTextAdapter`, enable the adapter setting
`"shared_text": true`. Older builds reject this mode rather than using another
model. The optional transport is HTTP and is used by the Go2 CLI's default
`shared-model` backend as well as the simulation evaluator.

The simulation evaluator accepts `--planner-backend shared-model --endpoint URL`
with `--profile event`. `--shared-tools navigate finish` restricts the exposed
tool set when every decision already receives a fresh image. The optional
`navigate_task` tool runs the complete current task verbatim when the model
selects it; expose it with `--shared-tools navigate_task navigate finish`.
It avoids copying a long instruction into each response. Explicit `navigate`
calls still carry model-written subgoals or recovery instructions. User steering
updates the task reference before the next decision; stale responses are discarded.
For ordered plans, `--shared-plan-tools` registers `advance_goal`; also include it
in `--shared-tools` to expose it to the model. The model supplies evidence that
the first remaining goal is complete. The harness moves exactly that entry into
completed history, preserving order and repeat visits, then runs the next goal
or requests final STOP. An optional `instruction` specifies the next local route.
`navigate` with a replacement plan remains available for explicit replanning.
This bookkeeping does not verify visual completion. A `finish(success)` call
with remaining goals in the supplied plan or current memory is rejected before
STOP. The model must explicitly update its plan or advance the completed goal;
omitting the plan does not discard saved obligations.
`--shared-source-tools` optionally registers `plan_task` and `continue_goal`;
include them in `--shared-tools` to expose them. The context numbers the literal
words of the current task. The model selects ordered inclusive word ranges;
`plan_task` copies those exact spans into the remaining plan, preserves completed
history and starts the first goal. `continue_goal` reuses that goal verbatim and
keeps its navigation history. The harness does not choose boundaries, infer
completion or silently repair invalid ranges. Free-form `navigate` remains
available for new recovery instructions. These tools require separate capability
and closed-loop validation; reference copying alone does not establish planning.
The optional
`--shared-tool-examples` flag adds two fixed format demonstrations, plus two for
`advance_goal` when that tool is exposed, and requires `navigate` and
`finish`. Demonstrations do not update the episode's memory or execute tools.
There is no `--planner-model` for this mode. Malformed calls are returned to the same model
once, then recorded as an agent failure. They are never repaired into motion.
Raw responses, observation images, token usage and separate client/server wall
times are recorded outside the source tree.
Task success still comes from the independent simulator evaluator.

This integration is under development. It does not yet establish improved
navigation accuracy, on-device real-time behavior, or the 50% overhead target.
