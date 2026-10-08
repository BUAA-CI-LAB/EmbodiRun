# Deployment and Runtime work

Measure the work required to obtain and maintain a ready deployment. This
directory supplies a procedure and **empty collection/report templates**, not
measured results. The model-engine chart elsewhere in this repository measures
a different question. Read the [responsibility comparison](../../docs/en/runtime-value.md)
([中文](../../docs/zh/runtime-value.md)) before selecting a workload.

## Two comparison scopes

| Scope | Paired routes | Available evidence |
|---|---|---|
| Software service management | Direct process startup versus Host management of the same Control service, `simulated.joints`, and fake camera | Existing executable service and Host commands; measurement not yet run. No inference, CUDA, learned-policy, or physical-robot conclusion. |
| Full policy deployment | Upstream LeRobot deployment versus EmbodiRun, using the same model/backend/hardware and observations | Pending assets, matched configuration, and protocol adapter. No full comparison commands or results are claimed here. |

LeRobot v0.6.1's remote policy server speaks gRPC. EmbodiRun's current model
providers speak its HTTP/WirelessComm interfaces. The repository does not
contain a ready native LeRobot compatibility adapter. To make the full scope
executable:

1. Select a policy/checkpoint and device combination supported on both routes;
   obtain the authorized model and recorded inputs, and record their versions.
2. Implement a provider/adapter for the required protocol, feature names,
   preprocessing, session behavior, action mapping, and health/readiness.
   Keep model code outside the generic Runtime domains.
3. Verify the same inputs give equivalent policy outputs, mapped actions and
   bounds, with the same dtype, decode steps, compile/graph settings and horizon.
4. Write both actual startup/update/recovery command lists and check equivalent
   readiness. Include adapter setup and maintenance in the operator record.
5. Run the six procedures below. Until then, mark unsupported cases
   `not_run`/`unavailable`; do not substitute a fixture or an optimized engine.

## Native backend preparation

The initial candidate is **Pi0.5/SO-101 named recorded-input replay**, with no
robot connection or action writes. Direct gRPC and a future optional HTTP bridge
would call separately started instances of the same LeRobot v0.6.1 backend.
This checks compatibility before a full deployment comparison; it does not
measure native async rollout costs or replace the software procedure below.
The [interface mapping](../../docs/en/runtime-value.md#prepare-a-comparison-with-the-native-lerobot-backend)
([中文](../../docs/zh/runtime-value.md#准备同原生-lerobot-后端的对照)) explains the existing support and gaps.

After preparing the same isolated v0.6.1 backend environment for both routes,
the upstream server command is:

```bash
"$BACKEND_ENV/bin/python" -m lerobot.async_inference.policy_server \
  --host=127.0.0.1 --port=8080 --fps=5 \
  --inference_latency=0 --obs_queue_timeout=1
```

These options are verified against the upstream [configuration](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/async_inference/configs.py#L46);
the command has not been run for this task. Use a fresh owned process for each
route/trial, with port 8080 free and only one client. `Ready` resets global queue
state. `SendPolicyInstructions(RemotePolicyConfig)` supplies `policy_type=pi05`,
the authorized checkpoint path, the same LeRobot feature descriptions,
`actions_per_chunk` and device; model loading begins there, not at process
startup. Keep pickle/gRPC on trusted loopback or authenticated SSH forwarding
to a trusted peer, and retain HTTP authentication and payload limits.

Remaining work for the bridge and replay client is bounded:

1. Provide the existing HTTP health/capability/session/step/reset/close surface
   from an optional integration process. Health probing must not call native
   `Ready`. Explicitly reject an additional session and any unsupported reset;
   preserve request identity, bounded waits and idempotency. A fresh backend
   process is the initial trial reset procedure.
2. Turn the same lossless images and named state into the same native
   `TimedObservation`, task string and feature order. In this sequential replay,
   submit one observation at a time with the same `must_go=True` on both routes
   so native similarity filtering does not silently skip a recorded input.
   Read `GetActions` to completion before the next input; retain its source
   timesteps/timestamps and finite named numeric chunk. This is deliberately a
   replay schedule, not the native robot client's async queue behavior.
3. Check input/preprocessed/output equivalence with the same checkpoint,
   processors, dtype, horizon and random-state treatment. Compare unknown,
   missing and reordered features, empty/malformed chunks, a delayed response,
   repeated request and reset/close behavior using ordinary tests. Record
   mismatch and failure cases, including successful checkpoint-loading evidence;
   do not treat a connection or numeric chunk as proof that weights loaded.
4. Supply actual bridge/replay commands and a matched execution configuration
   only after target validation. Complete native async queue/clock/stop-feedback
   mapping before reporting an async robot deployment comparison. Include this
   adapter's setup and maintenance in the final manual-work record.

No bridge, replay command or authorized checkpoint/input set is provided yet.
No learned-policy run is authorized or reported by this preparation section.

## Timing and collection

Use a new task-owned directory on the same Linux measurement host. Record Git
revisions and local diffs, OS/CPU/RAM/GPU/driver/CUDA, node placement/network,
package versions, checkpoint/input versions, policy settings, and the exact
commands. Start from [environment-template.json](environment-template.json).
Keep credentials and private asset paths out of published copies.

Copy [measurements.csv](measurements.csv),
[operator-actions.csv](operator-actions.csv), and
[report-template.md](report-template.md) into that directory. They contain no
sample measurements. Missing values stay blank/null with a reason; they are
not zero. Record failures and retain stdout/stderr, exit status, service logs,
and the readiness response for each attempt.

| Phase | Start → end | Record separately |
|---|---|---|
| `download` | First retrieval → required files present | Source, model/assets, and Docker base-image retrieval; state what was already present. |
| `environment` | Environment preparation begins → required imports available | Fresh task environment and uv download cache versus reused cache; native installation and Docker build in separate rows. |
| `model_load` | Backend starts loading weights → weights usable | Model/framework versions and log events; unavailable for the no-model workload. |
| `warmup` | First warmup/compile request → agreed warm readiness | Compile/graph capture and warmup count; never merge into model loading without saying so. |
| `control_ready` | Control start requested → successful health check and first valid observation | Device initialization, input readiness, polling interval, and timeout. |
| `stop` | Stop requested → owned service exits and endpoint closes | Model/Control order, confirmed outcome, failures; physical stopped-state evidence is separate. |

If a command spans several phases and logs cannot separate them, record one
`combined` row naming the included phases. Do not invent a breakdown. A GNU
time capture for an actual foreground command is, for example:

```bash
/usr/bin/time -v -o "$RESULTS/init.time.txt" \
  uv run --frozen embodirun --state-dir "$STATE" --config "$CONFIG" init --root "$RESULTS/managed" \
  > "$RESULTS/init.stdout.txt" 2> "$RESULTS/init.stderr.txt"
INIT_EXIT=$?
```

`up` starts background services: GNU time measures its command, not aggregate
service CPU/RAM/GPU. Record the sampling tool, interval, process set and window
when collecting service resources. Record UTC start/end and elapsed seconds;
for automated timing use a monotonic clock. No measurement helper is provided
in this version.

An operator action is one deliberate step, such as editing a node address,
choosing an environment, finding a log, or retrying a failed installation.
Record it when it occurs, with its category, related file/measurement, and help
or retry reason. A script invocation can perform many operations; command count
does not define manual effort. Count edits and help separately, and apply the
same counting rule to both routes. Avoid double-counting operator actions when
summing phase rows.

## Software procedure using existing commands

This procedure opens only in-memory simulated devices. Run it on the selected
Linux host in a fresh task checkout/environment. A successful run proves this
software workflow only. Do not use the self-contained
`run_shared_device_fake.py` walkthrough to report first deployment: it creates
a temporary initialized-state fixture rather than preparing a deployment.

### Prepare the common service

These commands have not been executed on the measurement target for this
documentation change. They describe the existing command surface, with runtime
results still pending.

From the repository root:

```bash
CONFIG=examples/shared-device-fake.yaml
RESULTS="$PWD/artifacts/deployment-runtime/software-01"
STATE="$RESULTS/host-state"
mkdir -p "$PWD/artifacts/deployment-runtime"
mkdir "$RESULTS"
ss -ltn 'sport = :8100'
```

Choose a new `RESULTS` directory; if it exists, stop and select another.
If `ss` lists a listener, record `needs_attention` and stop this trial before
`init` or `up`. Preserve the existing listener. The separate `--root` below
isolates managed sources/environments; `--state-dir` alone does not do that.

```bash
uv sync --frozen
uv run --frozen embodirun --state-dir "$STATE" --config "$CONFIG" validate
uv run --frozen embodirun --state-dir "$STATE" --config "$CONFIG" init --root "$RESULTS/managed"
uv run --frozen embodirun --state-dir "$STATE" --config "$CONFIG" sync --source .
uv run --frozen embodirun --state-dir "$STATE" --config "$CONFIG" up
uv run --frozen embodirun --state-dir "$STATE" --config "$CONFIG" describe \
  --runtime fake-device --caller-id runtime-comparison --session-id preparation --json
uv run --frozen embodirun --state-dir "$STATE" --config "$CONFIG" observe \
  --runtime fake-device --caller-id runtime-comparison --session-id preparation --include-robot --json
uv run --frozen embodirun --state-dir "$STATE" --config "$CONFIG" down
```

This first pass creates the generated service configuration. Archive its real
preparation time and operator actions as **Host preparation**, including any
source download and cache reuse. It is shared setup for the following startup
comparison and is excluded equally from its two startup measurements. It is
not a paired direct-versus-Host first-deployment result.

From `$STATE/shared-device-fake.json`, record the node root, the deploy
environment path, and the Control service ID. The active source directory is
`<node-root>/overlays/deploy/current`; generated Control JSON is
`<node-root>/generated/<service-id>.control.json`. Inspect only the task's state
and generated files. Fill the paths below from that real state, not guessed
environment names:

```bash
NODE_ROOT=/absolute/node-root/from-the-task-state
CONTROL_ENV=/absolute/deploy-environment/from-the-task-state
SERVICE_ID=control-service-id-from-the-task-state
CONTROL_SOURCE="$NODE_ROOT/overlays/deploy/current"
CONTROL_CONFIG="$NODE_ROOT/generated/$SERVICE_ID.control.json"
```

### Compare direct and Host startup

Use two terminals on the same measurement host, initially in the same repository
root. Set the same absolute `RESULTS` and `STATE` and the values of `NODE_ROOT`,
`CONTROL_ENV`, `SERVICE_ID`, `CONTROL_SOURCE` and `CONTROL_CONFIG` in both
terminals using the preparation record above; keep `CONFIG` relative to that
repository root. For the direct route, keep the process in the foreground in
terminal A:

```bash
cd "$CONTROL_SOURCE"
VIRTUAL_ENV="$CONTROL_ENV" PYTHONPATH="$CONTROL_SOURCE/src" \
  "$CONTROL_ENV/bin/embodirun-control-serve" --config "$CONTROL_CONFIG" \
  > "$RESULTS/direct.stdout.txt" 2> "$RESULTS/direct.stderr.txt"
```

`RESULTS` is an **absolute path** so it remains valid after changing directories. Start timing
immediately before launching the process. In terminal B, use the same ready
condition for both routes: `/healthz` reports `status: ok`, then
`/v1/observe?include_robot=true` returns a valid observation with
`robot.metadata.simulated: true` and `robot.metadata.hardware_access: false`.

```bash
curl --fail --silent --show-error http://127.0.0.1:8100/healthz
curl --fail --silent --show-error \
  -H 'X-EmbodiRun-Caller-ID: runtime-comparison' \
  -H 'X-EmbodiRun-Session-ID: trial-1' \
  'http://127.0.0.1:8100/v1/observe?include_robot=true'
```

Use the same one-second polling interval and 60-second timeout on both routes;
record the actual responses and elapsed time to the first valid observation.
Stop the foreground direct process with Ctrl-C and confirm its exit and closed
endpoint before the next trial.

For the Host route, from the repository root:

```bash
uv run --frozen embodirun --state-dir "$STATE" --config "$CONFIG" up --wait-timeout 60
```

Use the same HTTP probes and record readiness separately from command exit.
`up` prints the per-service log path; preserve that log. Finish with:

```bash
uv run --frozen embodirun --state-dir "$STATE" --config "$CONFIG" down
```

Keep the generated JSON, environment, source, port, sampling and workload
identical. Alternate direct/Host order across at least three paired trials and
report each failure and the variation. Both routes must start with the endpoint
closed. This experiment measures added lifecycle work and overhead around a
common prepared service. It cannot establish savings versus native LeRobot.

## Six paired procedures

Each case ends at the same ready condition. Record active operator time as well
as wall time; preserve the original failure sample before recovering.

| Case | Procedure and measurements | Software scope / full policy scope |
|---|---|---|
| `first_deployment` | Begin with new task environments, empty per-task uv download caches and no running service. Account for source/model/assets download and Docker base retrieval independently. Install, configure, load, warm up and obtain the first observation. Include setup assistance and retries. | Host preparation can be recorded now; the shared-preparation direct startup is not its pair. Full paired deployment awaits the adapter and assets. |
| `manual_work` | During every case, record deliberate actions, files edited, edit operations, help requests and retries in `operator-actions.csv`. Report totals per case and route. | Applicable to actual software trials; full deployment totals require full trials. |
| `backend_change` | Start with backend A ready. Stop, apply the same A→B change, install required packages, restart, warm up and obtain the same observation/action readiness. Record changed files, downtime and rejected/failed attempts. | Unavailable with no model. Full scope needs both backend implementations and equivalent policy settings. |
| `device_extension` | Start with one ready device; add a second equivalent device on the same specified node placement. Keep per-device input and inference workload fixed. Record config/source edits, setup, readiness of both, resources and failures. | A software extension needs an explicitly reviewed second simulated-device configuration; none is supplied as a measured result. Physical expansion needs authorized devices. |
| `update_restart` | Start with version A ready; apply the same actual source change A→B, stop services, update, prepare only what is needed and restart. Record stopped time, ready time, state retention and any recovery. | Executable after selecting a real revision pair. A no-op restart does not measure an update. |
| `failure_diagnosis` | With both routes stopped, start a task-owned HTTP process on port 8100, then attempt each route's normal startup. Preserve stderr/service logs and failed startup time. Record when the operator identifies the port collision, stops the task-owned process and obtains the same ready observation. | A concrete no-model software fault; diagnosis and recovery await execution. For full scope, select an equivalent model/device fault and preserve its safety checks. |

For the port fault, first confirm both task services are stopped and port 8100
is free. In a separate terminal on the measurement host, start the task-owned
process and retain its command/PID:

```bash
uv run --frozen python -m http.server 8100 --bind 127.0.0.1
```

Attempt each route with the same timeout and input configuration. Its normal
Control listener cannot bind that port. Preserve the failed sample; identify
the occupying process from the logs and task record, stop this foreground
process with Ctrl-C, then restart the selected route and repeat its ready
probes. Recreate the fault for the other route and alternate trial order.
Do not stop an unrelated listener. Corrupting generated Control JSON is not
equivalent: Host `up` would regenerate that file before startup.

## Report without mixing evidence

Fill the [report template](report-template.md) only after executing the commands.
State separately: command/software success, model inference success,
independent first-use success, and physical-device validation. Report medians,
range, sample counts and failures for paired trials. A faster setup command,
one warm run, model optimization chart, or documentation review does not answer
all six costs. Publish the collected samples and commands with any cost claim.
