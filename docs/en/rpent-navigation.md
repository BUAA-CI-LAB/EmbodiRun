# RPent + ActiveVLN navigation

RPent tracks visual progress, chooses local navigation goals and decides when to
replan or finish. ActiveVLN executes the local skill; EmbodiRun owns observations,
HTTP sessions and bounded execution.

| Tool | Behavior |
| --- | --- |
| `observe` | Fresh RGB and consumed execution budget |
| `navigate` | Local visual control until an event or bounded checkpoint; optional plan update in the same call |
| `navigate_task` | Optional event-profile tool that runs the initialized current task verbatim when selected by the model |
| `finish` | Save an optional plan and end the task; planner success is not evaluator success |

The default `event` profile yields on local STOP, repeated collisions, visual
stall/repetition, budget exhaustion, or at most 32 chunks. ActiveVLN observes a
fresh image before every chunk. Visual events flag possible stalls or loops;
they do not certify location or semantic progress. Go2 uses visual feedback
because this backend does not expose collision telemetry. `navigate` and
`finish` accept `plan={completed,remaining,observations}` without another tool
round. Only planner feedback uses JPEG quality 80, without resizing; policy
images are unchanged. The remote executor compresses before SSH transfer.
The initial frame is attached to the task without an `observe` tool round.
`--profile legacy` retains the four-tool, 1–8 chunk interface.

Repeating a subgoal preserves model history. A changed goal, explicit restart or
continuation after local STOP resets ActiveVLN while RPent retains task progress.
A local STOP does not end the overall task. The Go2 backend uses the existing
camera, control clients and synchronous chunk executor, releasing its motion
lease before waiting for RPent.

The local skill checks cancellation and its episode deadline before and after
each model call. A completed inference still counts toward the budget, but a
late action chunk is not passed to the backend. The shared-model planner retains
actual tool feedback while rejecting a finish returned after cancellation or
its deadline.
For shared-model tools running on the caller's thread, a signal records episode
cancellation and lets execution unwind at its next checked boundary before
waiting for cleanup. This avoids waiting on the same thread's active tool;
background MCP tools still finish their cancellation boundary before cleanup.

See the [navigation entrypoint](https://github.com/yufoo1/EmbodiRun/tree/feat/activevln-go2-adaptation/agents/rpent/navigation)
for installation and commands. Without `--execute`, it only observes and plans.
Motion requires an already armed owner supporting scoped lease release. The new
planner layer has not been tested on a physical Go2.

Simulation uses identical ActiveVLN weights, precision, initial states and a
500-step budget. RPent sees instructions, RGB, action records, collision counts
and consumed steps. Goal coordinates, reference paths and evaluation distances
remain evaluator-only. Memory is isolated per episode. Success requires final
STOP within a geodesic distance strictly below 3 metres; SPL and added planning
cost are reported separately. Initial memory contains the complete task as one
unfulfilled goal; only subsequent model decisions change its semantic progress.
R2R unseen-scene navigation does not establish
general long-task capability on a physical robot.

See the [simulation comparison](https://github.com/yufoo1/EmbodiRun/tree/feat/activevln-go2-adaptation/benchmarks/rpent-navigation)
for results and reproduction.

An experimental `shared-model` evaluator backend uses the same resident model
for navigation and RPent tool decisions. It verifies model-instance identity,
keeps the original task and episode memory in bounded context, and validates
model-generated arguments against the actual RPent schemas. It has no second
planner-model configuration. The Go2 CLI now defaults to `--planner-backend shared-model`;
it checks the navigation model identity before opening hardware clients and has
no provider fallback. Use the event profile and omit `--planner-model`. Without
`--execute`, only `observe`/`finish` are exposed and success claims are rejected
as unexecuted. Execution exposes `navigate_task`, `navigate`, `advance_goal` and
`finish`. The earlier cloud backend requires explicit `--planner-backend codex
--planner-model MODEL`. Shared-model serving, its accuracy/latency contract and
physical Go2 behavior remain under development; the entrypoint has CPU fake-client
checks, not a robot validation. Planner images must fit the service's 307200-pixel
budget and are not resized by this entrypoint.
Planner statistics separate client request wall time (`request_elapsed_s`) from
server handling time (`service_elapsed_s`, with `service_timing_samples`). Server
time includes scheduling and preprocessing; neither field measures GPU time alone.
`--shared-plan-tools` registers `advance_goal`; include it in `--shared-tools` to
expose it. After the model judges the current goal complete and supplies evidence,
the tool moves the first remaining goal into completed history, preserving order
and repeat visits, then runs the next goal or requests final STOP. An optional
`instruction` specifies the next route. Explicit replanning still uses `navigate`
with a replacement plan. This bookkeeping does not verify visual completion.
`finish(success)` rejects remaining goals in the supplied plan or current memory
before requesting STOP. The model must explicitly update its plan or advance
the completed goal; omitting the plan does not discard saved obligations.
`--shared-source-tools` optionally registers `plan_task` and `continue_goal`;
also select them in `--shared-tools`. The context numbers literal task words.
The model chooses ordered word ranges; the harness copies them exactly, retains
completed history and starts the first goal. Continuing a saved goal preserves
navigation history. These tools neither choose semantic boundaries nor judge
completion, and require separate capability and closed-loop validation.
See the [navigation integration](../../agents/rpent/navigation/README.md#experimental-shared-model-planner)
for the required experimental endpoint and evaluator flags.
