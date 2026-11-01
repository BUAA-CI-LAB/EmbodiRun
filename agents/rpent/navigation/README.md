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
codex login status
```

RPent is pinned to `7f403130`; the Codex Python SDK is pinned to `0.163.0`.
The integration uses RPent's `CodexPlanner`, `Toolkit`, multimodal MCP transport
and episode-local memory. It exposes only the selected navigation tools, with shell,
web search and generic file tools disabled. Existing Codex credentials stay on
the planner host. The validated CLI version is `0.164.0`.

Read-only Go2 planning against **already deployed** services:

```bash
PYTHONPATH=. \
.venv-rpent/bin/python -m agents.rpent.navigation.go2 \
  --config /path/to/go2-navigation.json \
  --instruction 'Leave the room, turn left into the hallway, and stop inside the study.' \
  --planner-model gpt-6-astra --output /path/to/new-run
```

Motion additionally requires `--execute` and an already armed owner supporting
scoped lease release. Endpoint/token environment variables and timing bounds
come from `NavigationConfig`. This new planner layer has not been tested on a
physical Go2. Its simulation comparison is documented in
[`benchmarks/rpent-navigation`](../../../benchmarks/rpent-navigation/README.md).
The optimized profile has a separate [comparison](../../../benchmarks/rpent-navigation/optimization.md).

中文说明见 [RPent 导航集成](../../../docs/zh/rpent-navigation.md)。
