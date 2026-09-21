# Inference transport

EmbodiRun carries one versioned, model-neutral inference contract over either
HTTP or WirelessComm ([Inference API v1](http_api.md)). Both are supported
transports, unlike the opt-in candidates in
[Transport candidates](transport-experiments.md). This note asks whether the
choice changes end-to-end latency as seen from the robot nodes, and where a
request's time actually goes.

Both protocols ran on the same wired LAN segment with no bandwidth limit configured.

## Setup

One compute node hosts a PI0.5 inference service with an SO101 adapter. Two
robot nodes — an AGX Orin and an Orin NX — each run one client, and each keeps
at most one request outstanding. The clients replay one shared input manifest
in file order, and a barrier releases both ends after warm-up. Warm-up is 20
requests per end; the formal measurement is 200 per end, 400 in total. The
service processes requests at B=1, so the two-client runs include contention
and queueing rather than parallel model execution.

The two deployment configurations differ only in transport. Checkpoint, BF16,
10 denoising steps, the full-loop CUDA graph and the 50×6 action output are
identical. The input is recorded: camera JPEGs, joint state and instruction come
from the arm, and only the link is under test. Each protocol replays the same
manifest. SHA-256 covers the instruction, state, camera names, encoded image
bytes and sample order, so `compare` rejects a pair whose inputs differ.
Alternating protocol order across rounds is recommended; the 2026-09-08 run
below used HTTP → WirelessComm → WirelessComm → HTTP.

The benchmark measures inference requests and action validation using replayed
observations; it does not execute robot motion.

## Timing definitions

| Metric | Definition |
|---|---|
| E2E mean / P50 / P95 / P99 (ms) | Request construction on the robot node through response reception and completion of SO101 action mapping and 50-row validation. |
| RPC (ms) | `client.step`: request serialization, transport, service queueing, preprocessing, model execution, post-processing and response parsing. |
| calls/s per end | Successful requests on that end divided by that end's formal measurement time. |
| aggregate calls/s | Successful requests on both ends divided by the interval between the host's start signal and both completion signals. |
| Failure | Failed attempts and their durations are retained; the end stops at the first failure without retrying and the round is marked failed. |

E2E excludes file reads, SSH input distribution, warm-up, connection and session
setup, sensor acquisition and arm motion. Each end uses its own monotonic clock,
so a latency never depends on cross-device clock synchronization; the aggregate
denominator includes a small SSH start/finish signalling cost. Per-request
`server_timing_ms` records time spent inside the service. Every
request carries a distinct ID, so warm-up and formal requests cannot hit an
idempotent result cache.

After both ends finish measuring and close their own server session, the hosts
release the transport connections together. A session-close failure keeps the
measurements and marks the round failed. The benchmark uses the inference client
and the value mapper only; it opens no serial port or camera and executes no
action.

## Link result, 2026-09-07

Two SO101 arms recorded 24 observations each — 48 groups, 96 JPEGs — and every
motor's calibration registers matched its file. All four camera views were
almost entirely black: more than 99.5% of pixels were below 16/255, and only a
few indicator LEDs were visible in the front view. The two JPEGs in a group
averaged 11,106 bytes on the AGX Orin and 12,164 bytes on the Orin NX. The
measurements below use these small, dark-image payloads.

Both protocols completed 400 formal requests with 50×6 actions and clean client
session teardown. Each row is a single round after 20 warm-up requests:

| Protocol | Client | Calls | E2E mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) |
|---|---|---:|---:|---:|---:|---:|
| HTTP | AGX Orin | 200 | 434.81 | 435.05 | 443.28 | 444.04 |
| HTTP | Orin NX | 200 | 433.74 | 435.88 | 442.60 | 444.54 |
| WirelessComm | AGX Orin | 200 | 419.31 | 421.62 | 425.42 | 426.87 |
| WirelessComm | Orin NX | 200 | 420.34 | 421.63 | 425.32 | 426.92 |

Aggregate throughput was 4.595 calls/s over HTTP and 4.755 calls/s over
WirelessComm.

The first observation from each arm was also checked against the real PI0.5
output: both returned 50×6 finite values. The first predicted target differed
from the current pose by up to about 113.0° and 56.2° in the joints and by 21.5
and 15.5 in the gripper, all above the configured per-step limit. No model action
was sent to the motors.

## Inference staging, 2026-09-08

A second run loaded the model once and served both transports from that process,
so the protocol arms reused the same processor and CUDA graph. Both robot nodes
reached the compute node over a wired LAN. Only previously recorded observations
were replayed; no serial port was opened and no action was executed. Single-client
HTTP and WirelessComm took 50 formal requests each, then the dual-client runs
took 100 per end in the order HTTP → WirelessComm → WirelessComm → HTTP. All 900
formal requests succeeded with 50×6 actions.

| Scenario | HTTP E2E mean | WirelessComm E2E mean | E2E reduction | Aggregate calls/s | Throughput gain |
|---|---:|---:|---:|---:|---:|
| AGX Orin, single client | 217.25 ms | 207.13 ms | 4.65% | 4.586 / 4.811 | 4.91% |
| AGX Orin + Orin NX, two rounds pooled | 437.03 ms | 418.12 ms | 4.33% | 4.560 / 4.765 | 4.49% |

The dual-client rounds individually averaged 437.82 / 436.23 ms for HTTP and
416.47 / 419.76 ms for WirelessComm. Pooled P95 was 442.37 / 421.92 ms and pooled
P99 was 443.85 / 423.12 ms. The single-client scenario has one round; each
dual-client protocol has two.

The same run recorded per-request stage times. The table is the dual-client mean
per request; exclusive stages sum to E2E, and a positive difference means
WirelessComm was shorter in that stage.

| Stage | HTTP (ms) | WirelessComm (ms) | HTTP − WirelessComm (ms) |
|---|---:|---:|---:|
| Client request and action mapping | 0.63 | 0.64 | −0.01 |
| RPC surroundings: network, protocol codec, service routing | 7.89 | 6.17 | 1.72 |
| Waiting for the shared inference lock | 208.41 | 200.92 | 7.49 |
| Input preparation: state, images, processor | 13.32 | 10.11 | 3.21 |
| Engine call, including synchronization and action copy to CPU | 203.43 | 197.58 | 5.85 |
| Action post-processing, adapter, remaining service timing | 3.34 | 2.69 | 0.65 |
| **E2E** | **437.03** | **418.12** | **18.91** |

Inside input preparation, image decode and tensor construction took 10.41 / 7.53
ms. The engine's existing CUDA events split into 92.77 / 89.91 ms of prefill and
109.91 / 107.40 ms for the 10 denoising steps. Those stream times are already
inside the engine call and are not added again.

| Added probe | Definition |
|---|---|
| `profile_lock_wait_ms` | Wall time waiting for the shared PI0.5 inference lock. |
| `profile_state_ms` / `profile_images_ms` | State conversion, and JPEG decode plus image tensor construction. |
| `profile_prepare_ms` | The model input processor. |
| `profile_engine_wall_ms` | The whole `EngineCore.execute` call, including its synchronization and the action copy back to CPU. |
| `profile_core_prefill_ms` / `profile_core_decode_ms` | The engine's existing CUDA events for the prefix stage and the denoising stage. Stream time, not the sum of individual kernels. |
| `profile_core_e2e_ms` | The engine's existing wall clock, ending before the action is packed. |
| `profile_restore_ms` | Action denormalization and similar post-processing. |
| `profile_adapter_other_ms` | Remaining adapter work: action checks, list conversion and probe overhead. |

The engine's CUDA segments are diagnostic values inside engine wall time and
must not be added to the end-to-end decomposition; the exclusive wall-clock
stages can be summed. Absolute clocks from different processes are not used to
compute one-way network latency. The diagnostic response carries these extra
fields, and profile output is stored separately from the original benchmark
report.

## Findings

- Under this load, WirelessComm gave about a **4.5% total throughput gain**. An
  earlier run that started the model service independently measured 3.47%.
- Of the **18.91 ms** E2E reduction, **1.72 ms** came from communication and RPC
  handling; the rest came from queueing, preprocessing, and model execution.
- With two clients, the engine call plus lock wait was **94–95%** of E2E, while
  single-client lock wait was below 0.001 ms. Doubling the client count left
  aggregate throughput near 4.6–4.8 calls/s and roughly doubled each end's
  latency, so the limit is model processing and its queueing under a serialized
  B=1 service.
- Preprocessing and CUDA stream times differed between the two entry points even
  with one loaded model, which means independent model loading was not the only
  source of variance.

## Interpreting the transport difference

The HTTP client used `urllib.request.urlopen`, which sends `Connection: close`
on every request, against a server that serves each connection from its own
thread. WirelessComm reused the connection through a thread pool. Thread reuse,
scheduling and device clocks can all affect service-internal timings, and no
thread-reuse ablation was run and no per-request clocks were logged, so the
contribution of thread reuse and device-clock variation remains unresolved.

To compare the protocols themselves, add an HTTP connection-reuse arm so the
thread-reuse conditions are closer, then repeat the staging measurement.

For your deployment, repeat the comparison with representative images, payload
sizes, and network conditions. These runs used roughly 11.6 kB of JPEG data per
request on a wired LAN.
