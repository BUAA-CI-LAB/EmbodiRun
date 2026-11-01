# RPent + ActiveVLN navigation

RPent tracks visual progress, chooses local navigation goals and decides when to
replan or finish. ActiveVLN executes the local skill; EmbodiRun owns observations,
HTTP sessions and bounded execution.

| Tool | Behavior |
| --- | --- |
| `observe` | Fresh RGB and consumed execution budget |
| `navigate` | Local visual control until an event or bounded checkpoint; optional plan update in the same call |
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

See the [navigation entrypoint](https://github.com/yufoo1/EmbodiRun/tree/feat/activevln-go2-adaptation/agents/rpent/navigation)
for installation and commands. Without `--execute`, it only observes and plans.
Motion requires an already armed owner supporting scoped lease release. The new
planner layer has not been tested on a physical Go2.

Simulation uses identical ActiveVLN weights, precision, initial states and a
500-step budget. RPent sees instructions, RGB, action records, collision counts
and consumed steps. Goal coordinates, reference paths and evaluation distances
remain evaluator-only. Memory is isolated per episode. Success requires final
STOP within a geodesic distance strictly below 3 metres; SPL and added planning
cost are reported separately. R2R unseen-scene navigation does not establish
general long-task capability on a physical robot.

See the [simulation comparison](https://github.com/yufoo1/EmbodiRun/tree/feat/activevln-go2-adaptation/benchmarks/rpent-navigation)
for results and reproduction.
