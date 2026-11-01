# RPent navigation optimization

The `event` profile keeps ActiveVLN's per-chunk visual loop on the execution host.
RPent receives control at local STOP, collision/visual watchdog events, a
32-chunk checkpoint, or budget exhaustion. Plan updates are combined with
navigation/finish. Only planner images are JPEG-compressed, at quality 80 and
the original 640×480 resolution, before crossing SSH.
The first frame is attached directly to the task, avoiding an initial `observe`
tool round. Its collection time is still included in episode time.

Development episode 1711 (manifest offset 128, outside the formal first 128):

| RPent + ActiveVLN profile | Success | SPL | Episode time (s) | Planner tool calls | ActiveVLN calls |
| --- | --- | --- | --- | --- | --- |
| Legacy, contemporaneous control | Yes | 1.0000 | 66.58 | 10 | 10 |
| Event, initial image attached | Yes | 0.8932 | 45.89 | 2 | 12 |

This single episode is 1.45× faster, with lower SPL. It does not establish an
aggregate success rate or stable speedup. Baseline action sequences also varied
between runs despite the same initial pose and frozen weights. Original
development attempts are retained outside Git.

<!-- formal-event-results -->

Use the current source on both the planner and execution hosts, with the model
service and executor setup described in the [paired evaluation](README.md):

```bash
PYTHONPATH=.:src NO_PROXY=localhost,127.0.0.1 \
.venv-rpent/bin/python benchmarks/rpent-navigation/evaluate.py \
  --manifest /path/to/selected-r2r-navigation-256.json \
  --scenes /path/on/gpu-host/to/scene_datasets \
  --simulator-command /path/to/executor-command.json --remote-executor \
  --count 128 --planner-model gpt-6-astra --profile event \
  --event-max-chunks 32 --output /path/to/new-event-comparison
```

`protocol.json` records source
hashes, event thresholds and image settings; mixed-source resumes are rejected.
The frozen first implementation is commit `9497686` and retains its own results.

`transport-timings.json` records executor duration, image encode time and bytes,
and client round-trip time for each operation. `agent/tools.jsonl` records tool
handler duration and image bytes. These intervals overlap; do not add ActiveVLN
inference or executor time to the enclosing tool/episode duration. The client
minus executor interval includes transport and serialization, not network alone.
Raw timing, image, trajectory and JSON artifacts stay outside Git.
Initial-frame acquisition is recorded in `agent/initial-observation.json` and
does not count as an LLM tool call.

Functional development checks use episodes outside the formal first 128.
Formal speed and success comparisons are reported separately after completion.
