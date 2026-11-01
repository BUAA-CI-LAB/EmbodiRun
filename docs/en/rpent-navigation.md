# RPent + ActiveVLN navigation

RPent tracks visual progress, chooses local navigation goals and decides when to
replan or finish. ActiveVLN executes the local skill; EmbodiRun owns observations,
HTTP sessions and bounded execution.

| Tool | Behavior |
| --- | --- |
| `observe` | Fresh RGB and consumed execution budget |
| `navigate` | 1–8 complete ActiveVLN chunks, followed by fresh RGB and feedback |
| `update_plan` | Episode-local completed goals, remaining goals and observations |
| `finish` | End the task; planner success is not evaluator success |

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
