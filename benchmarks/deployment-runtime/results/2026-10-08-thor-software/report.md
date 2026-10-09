# Prepared Control startup and one port fault on Thor

On 2026-10-08, an **Agent** executed three alternating Direct/Host pairs against
the same real Control process implementation, using `simulated.joints` and a
fake camera. All six startups returned a fresh observation. All six stops
confirmed that the owned Control process exited and port 8100 closed. This
demonstrates the prepared software lifecycle; it does not establish deployment
savings against native LeRobot.

## Source, workload and preparation

The measured Host and Control code was
`3fe4df9d6667a02226302c7f0766e3d1ec8b7c27`, without Runtime source edits.
The external recorder was outside the repository. Later MicroDuck installer
and test changes were not part of this measured source.

Both routes used one Linux/aarch64 host, Python 3.12.3, uv 0.12.17, the **same
managed environment with 13 unique package name/version pairs**, the same active
source directory and generated Control JSON, and trusted loopback port 8100.
[Environment](environment.json) and [managed packages](managed-packages.json)
record the actual versions. The raw editable `embodirun` metadata was enumerated
twice; that is one package/version, not a fourteenth dependency. The caller's
root development environment and MicroDuck's base environment are separate.

The input was one 64×48 fake PNG and five in-memory joint positions plus a
gripper value. No execute request, learned policy, checkpoint, scene, GPU model
or physical robot was used. Readiness required `/healthz` status `ok`, then a
fresh `/v1/observe?include_robot=true` result containing the front image and
`robot.metadata.simulated: true`, `hardware_access: false`.

Preparation was **assisted** and excluded equally from both startup routes:

- Target public Git cloning failed strict TLS verification; a process-local
  Git resolution workaround allowed a revision query, but the subsequent
  bounded clone timed out. An original-revision source archive and fresh
  one-commit shallow Git metadata were relayed from another host. The official
  origin and actual revision were retained; this was not a successful target
  public-network clone.
- Only the task copy of the deployment manifest changed `deploy-commit: main`
  to the recorded revision. The public example manifest was preserved.
- The task environments, state and download cache started new. Target locked
  downloads timed out. Matching ARM64 wheels were freshly downloaded on another
  host and transferred; original frozen requirements were exported, imported
  into new environments and audited with normal frozen sync. No old venv,
  deployment state or package cache was copied.
- The first assisted managed environment selected Python 3.11 despite the
  caller using 3.12. That failed preparation was preserved. A new trial set
  `UV_PYTHON=/usr/bin/python3` and verified managed Python 3.12.3. Core supports
  3.11; this was an explicit choice of the comparison's 3.12 reference.
- System Python and an already installed uv executable were reused. A common
  `init → sync → up → describe → observe → down` pass prepared the generated
  config. The measured startups reused this prepared task environment/cache.

[Operator records](operator-actions.csv) describe the assistance. Per-action
timestamps and active human time were not collected. These records are not a
paired manual-work total, a classroom trial, an independent first-use acceptance,
or evidence of fast uncached target installation.

## Three prepared startup pairs

The orders were Direct/Host, Host/Direct, Direct/Host. The endpoint was closed
before each attempt. An external monotonic recorder launched the command and
began HTTP probing immediately, then every one second, with a 60-second limit.
It recorded Host command exit in a separate waiter and waited for `up` to finish
before calling `down`. Both routes used the same readiness condition.

| Pair | Order | Direct first valid observation (s) | Host first valid observation (s) | Host `up` command exit (s) |
|---|---|---:|---:|---:|
| 1 | Direct → Host | 1.020183 | 1.021737 | 0.758030 |
| 2 | Host → Direct | 1.019798 | 1.021831 | 0.765642 |
| 3 | Direct → Host | 1.020023 | 1.022098 | 0.766176 |

| Boundary | Direct: median, range (s) | Host: median, range (s) |
|---|---|---|
| Request start → valid observation | 1.020, 1.019798–1.020183 | 1.022, 1.021737–1.022098 |
| Stop request → owned Control exited and endpoint closed | 1.001, 1.001111–1.001385 | 1.003, 1.002812–1.003069 |
| Host command exit | Foreground process exits after stop, not startup | `up`: 0.766, 0.758030–0.766176; `down`: 0.516, 0.515288–0.518016 |

**All observations and stop confirmations landed on the second one-second
probe. The roughly 0.002-second difference cannot support a speed or overhead
benefit claim.** These are three coarse paired samples, not a latency distribution
or a cold first deployment. Large package installation/build work was paused
during this window; network transfer was permitted and not resource-sampled.

Host `up` and `down` exited 0 in every successful trial. Direct was intentionally
stopped using SIGINT and returned **-2**. Its foreground lifetime is recorded
separately and must not be compared to Host's `up` exit time. The first external
recorder labeled those three foreground-exit rows `failed`; the published copy
classifies that requested interrupt as `operator_interrupt`, retaining the exact
exit codes and timings. The original raw records remain in the private audit.

## Separate occupied-port failure and recovery

After the quiet window, the Agent started an owned plain HTTP server on loopback
8100 with an empty task directory, attempted the normal Control startup, retained
the failure, stopped only that occupier, and restarted the same route. Package
installation/build work could run concurrently in this later window. These are
one fault example per route, separate from the three startup pairs.

| Route | Failed startup command | Actual evidence | Post-cleanup restart → valid observation (s) |
|---|---|---|---:|
| Direct | Exit 2, 0.183533 s | `[Errno 98] Address already in use` | 1.021511 |
| Host | Exit 1, 0.696307 s | Service exited before ready; the referenced Control log contains the same bind error | 1.025351 |

Both recovery services subsequently stopped and port 8100 closed. Host failed
state was cleared with task `down` after its failed `up` command exited. These
restart times begin **after** occupier removal/failed-state cleanup; they are not
total outage time or human diagnosis time. The error and log reference show
where to inspect a real software failure; one sample and different background
load do not support a fault-diagnosis speed comparison.

Two intervening SSH attempts failed before remote execution. They did not start
an occupier and are network failures, not Control startup samples. The later
attempts above actually ran. Final port inspection found no listener, and the
task process listing found no remaining recorder, occupier or Control process.
A later PID audit found all 11 recorded owned preparation/Control/occupier PIDs
absent and confirmed that the final generated Control JSON matched preparation.
Another worker's owned build/installation processes were preserved.

## Evidence and remaining work

[Measurements](measurements.csv) retain the collection template's columns,
UTC endpoints, phase boundaries and exit codes. [Commands](commands.jsonl),
[command logs](command-logs.jsonl), [probe/process/stop evidence](trial-evidence.jsonl),
and the [Host service log](host-control-service.log) are sanitized published
copies. `<TASK>`, `<USER_HOME>` and `measurement-host` replace private paths or
host identity. JSONL records retain their original filename as `record_id`; CSV
notes identify the prepared/fault group. Exact private logs and the external
recorder remain outside the repository. The [software procedure](../../README.md#software-procedure-using-existing-commands)
explains the actual command surface; the task's [deployment](deployment.yaml)
and [generated Control config](control-config.json) retain the measured inputs.

Workload CPU/RAM/GPU/VRAM fields stay blank/null because no reliable aggregate
process sampler was run. Machine memory capacity is not service peak usage.
Full same-model native LeRobot comparison and its six deployment-cost procedures
remain pending: first deployment, paired human/manual work, backend change,
device extension, actual source update, and equivalent full-policy diagnosis.
Model inference, independent human first use and physical validation remain
`not_run`. This software report supplies no inference-optimization claim.
