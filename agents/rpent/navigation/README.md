# RPent + ActiveVLN navigation

RPent observes, tracks route progress, chooses bounded ActiveVLN subgoals and
decides when to replan or finish. ActiveVLN runs through the existing Go2 HTTP
policy binding. The Go2 backend reuses the camera and motion-lease clients;
Habitat uses the same policy and skill lifecycle.

| Tool | Behavior |
| --- | --- |
| `observe` | Fresh RGB and consumed execution budget |
| `navigate` | 1–8 complete ActiveVLN chunks; returns RGB and execution feedback |
| `update_plan` | Episode-local completed goals, remaining goals and observations |
| `finish` | End the task; the planner's success claim remains unverified |

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
and episode-local memory. It exposes the four navigation tools, with shell,
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

中文说明见 [RPent 导航集成](../../../docs/zh/rpent-navigation.md)。
