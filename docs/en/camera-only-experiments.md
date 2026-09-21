# Camera-only experiments

Install the `camera` extra in the camera/observation environment. Other EmbodiRun
roles do not need OpenCV or Pillow.

`embodirun.services.rollout.camera_capture` continuously captures V4L2 cameras,
records a bounded image sequence and exposes read-only observations for communication
experiments. It provides camera access only, with no robot connection or motion
endpoint. Its six state values are fixed test inputs.

```bash
python -m embodirun.services.rollout.camera_capture \
  --camera front /dev/v4l/by-id/FRONT-video-index0 \
  --camera wrist /dev/v4l/by-id/WRIST-video-index0 \
  --output /tmp/so101-camera-run \
  --width 640 --height 480 --fps 30 \
  --record-count 120 --record-hz 2 --duration-s 600
```

Use actual camera aliases. Every alias must resolve to a registered `/dev/videoN`
device. The output directory must be new. Stop the process with SIGTERM/SIGINT;
the configured duration also bounds its lifetime. Devices are released on exit.

The default listener is localhost port 29480. `/health` reports capture rate,
timing, process CPU time and recording progress. `/latest` returns one complete
camera generation with JPEG data and capture timestamps; it fails before capture
starts or after capture stops. `/frame/INDEX` cycles through the recorded sequence.
Wait until `recorded` equals `record_count_target` before paired measurements so
the indexed sequence stays fixed. Consumers of live frames must also check freshness.

To serve a specific experiment node, select a LAN `--bind` address and explicitly
add its IP with `--allow-host`. There is no POST or action endpoint. Disable proxy
environment handling in local observation clients to keep camera traffic local.

`observations.jsonl` and JPEGs can be consumed by `ReplaySource`. `capture.jsonl`
records ongoing acquisition load, and `process.json`/`final.json` record the process
and shutdown. Pair identical recorded inputs while acquisition continues to include
capture load without changing the model's paired input data. Live-input runs are
separate from comparisons requiring identical input fingerprints.

`CameraObservationSource` reads the publisher over HTTP with proxies disabled.
Recorded mode verifies every fetched frame against the fixed recording; it includes
HTTP read and JPEG decode work on each observation instead of serving a memory cache.
Live mode checks freshness when the publisher shares the consumer's host clock and
labels its input fingerprint as non-identical: separate live runs see different
frames. Cross-host monotonic timestamps are
retained as source metadata and are not subtracted to calculate latency.

## Camera-load measurements, 2026-09-11

The camera-load run used the four physical SO101 cameras on Orin NX and AGX Orin,
with Pi05 inference on Thor. Actions were validated only in memory, with fixture
joint states and zero rewards. Camera acquisition continued during paired tests;
both variants read the same recorded JPEGs over HTTP and decoded/resized the two
views to 224 × 224 RGB. Separate live-input runs measured software capture freshness.

Five normal C/D pairs compared the RLinf channel transport with the EmbodiRun
wireless endpoint, both on the same wired LAN. Each body offered 100 requests per trial at 2 Hz with a one-second
deadline. Both variants accepted 1000/1000 responses. Wireless reduced paired RTT p95
by 8.01% on average, with paired bootstrap 95% CI [-10.14%, -5.42%]; mean trial p95
was 253.88 ms for Channel and 233.56 ms for Wireless. This is a latency result under
the measured load; useful throughput was equal. RTT includes inference and queueing.
A separate five-pair A100–Thor test ran actual PPO updates and native CPU/Gloo patch
weight synchronization, pinned to specific RLinf and EmbodiRun revisions. Adding EmbodiRun
changed steady weight-ready p95 by +12.28% on average, with 95% CI [-8.03%, +32.16%].
The wide interval includes both faster and slower weight updates. Mean
updates were 1.110 GB, with 56.13% in indices. Lossless 16-bit column-index packing
would save approximately 249 MB/update (22.45% of payload), but no optimized hardware
ready-time improvement has been measured. Weight updates still use RLinf's syncer;
switching the observation transport does not replace that path.

Three two-second NX-process pause pairs accepted 282/300 affected responses with
Channel and 284/300 with Wireless. Both accepted 300/300 healthy AGX responses, and
all six trials continued through new post-restore slots to the last request. First
valid responses after restore took C/D 1579/1525, 1578/1457 and 1906/1089 ms. Actual
injection offsets were 5.36–5.53 seconds into the affected body's window, so request
phase contributes to these timings. Both recovered from the process pause.

### TCP disconnect recovery

Three original-receive TCP fault pairs explicitly shut down the affected worker's
existing data connections to Thor, protecting Ray control and camera ports. Both
variants used the same EmbodiRun envelope and action validation. Channel closed eight
native data/store sockets and Wireless closed one socket per trial under the same
peer/port selection rule. Subsequent reconnection was allowed; this is not sustained
network loss or a device restart.

Both original variants failed to restore the NX stream: each accepted 30/300 planned
NX responses, all before the fault, while healthy AGX accepted 300/300. Each affected
trial started 11 exchanges, with ten successes and one timeout; the remaining 89 slots
were marked aborted and retained in the denominator. None of the 267 slots scheduled
after disconnection per variant succeeded. Hardware logs showed Gloo connection errors
for Channel and termination of the per-body receive loop with `ConnectionClosedError`
for Wireless. Transport reconnection alone did not restart that receive task.

Three additional pairs enabled `retry_disconnected_receive`. Channel remained at
30/300 affected NX responses, with no recovery. Wireless accepted 300/300 affected
NX responses and 300/300 healthy AGX responses, with no errors or late/missed slots.
All 267 slots scheduled after shutdown succeeded, and every trial reached its last
request. First valid responses after shutdown took 433, 404 and 356 ms, including
inference and request cadence. The first strictly post-shutdown scheduled slots
completed 937, 910 and 834 ms later; the schedule is spaced at 500 ms.

The improvement requires the new optional receive-rearming behavior, disabled by
default. It preserves the original absolute receive timeout and strict stream
session/sequence checks, and never resends a payload. These approximately-five-second
faults occurred with no outstanding request, testing idle connection recovery.

Three further pairs closed connections at 5.1514–5.1519 seconds, with request 10
outstanding in every trial. Rearmed Wireless accepted 300/300 responses on each body,
including all 267 newly scheduled post-shutdown slots, and reached every final request.
Channel remained at 30/300 affected responses, with no recovery; its healthy body
accepted 300/300. Wireless's first valid response arrived after 178, 238 and 245 ms;
the first strictly post-shutdown slots completed after 684, 726 and 740 ms.
The outstanding request may already have reached the GPU. Testing partial
message loss or an interrupted PPO trajectory requires a different fault injection.

### Live capture

The separate live-input trials accepted 200/200 responses each. Channel/Wireless RTT
p95 was 254.36/251.36 ms, and local software capture-start-to-response p95 was
478.23/463.08 ms. This is not sensor-exposure timing or an identical-input paired comparison.
Cameras captured for approximately 3.2 hours at NX/AGX 10.26/10.23 paired frames/s;
average CPU was 33.77%/34.13% of one core. The 480 saved JPEGs passed checksum checks.
Thor completed 42,817 background camera reads without errors. Experiment processes
were stopped; a resident HTTP inference service from another Thor workspace was
retained, so GPU use was not guaranteed exclusive or free of other requests.
