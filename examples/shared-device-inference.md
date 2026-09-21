# Shared-device EmbodiInfer replay experiment

`run_shared_device_inference.py` is a bounded software-only experiment. It
uses the public loopback `ControlHttpServer` and the real `EmbodiInferHttpClient`,
while the robot is the explicit `simulated.policy_vector` adapter and the
camera source reads recorded front/wrist files. It does not open a serial,
USB, CAN, or vendor SDK resource.

The policy-vector path preserves six dataset-native values under the exact
SO-style feature names. `dataset_native_unverified` is an experiment label;
it does not claim degrees, normalized coordinates, joint limits, or physical
gripper semantics. The helper performs no unit conversion and refuses any
other `--model-units` value. The path therefore cannot be used to turn a
LIBERO/EE response into SO101 joint commands.

The input is either a `frames.jsonl` file or its containing directory. Each
line must identify a front and wrist image. The current recorded export uses
`cameras.2` for the main/front image and `cameras.0` for the wrist image;
`--front-key` and `--wrist-key` select other field names. Use
`--record-line 13` to select a particular JSONL line when the first lines do
not contain the desired pair. Image paths are resolved relative to
`frames.jsonl`. The loader keeps explicit `captured_timestamp_ns` values and
generic `timestamp` values separately. It reads the six-value measured vector
from `present` in the SO feature order and never uses `action`, `sent`, or
`goal` as the measured/present state.

Replay frames use a synthetic virtual-sensor delivery/capture timestamp in
`host_monotonic_ns` so the existing shared-observation freshness path remains
intact. Original capture timestamps and clock domains remain in frame profile
metadata with `replay=true`; the report declares `scene_live=false`. The
recorded scene does not change in response to simulated actions.

Preflight only checks the real endpoint and input manifest:

```sh
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python \
  examples/run_shared_device_inference.py \
  --endpoint http://127.0.0.1:18885 \
  --input /path/to/episode/frames.jsonl \
  --output-dir experiment-results/pi05-policy-vector \
  --state-dir experiment-results/pi05-policy-vector/state \
  --model-units dataset_native_unverified \
  --record-line 13 \
  --preflight-only
```

After the matching simulated binding and the endpoint capability response have
been checked, omit `--preflight-only` to run two requests through the same
fake robot/camera owner. Each request uses the configured task goal “pick up
the blue cube and place it in the bowl” and asks for one policy step, while
the policy may return a 50-step proposal and Deploy executes only a three-step
prefix. The helper starts recording before the requests, keeps a bounded
observation subscription active as a second consumer, writes the raw EmbodiInfer
results and timing to `raw-model-results.json`, and writes a concise
`report.json`.

Use a fresh output/state directory for each complete run. The job store keeps
request history and the recorder retains prior artifacts; rerunning with the
same directories is not a new experiment. Preflight does not execute a task.

The report distinguishes endpoint/model facts, replay provenance, shared
observation IDs, recorder status, and simulated execution. It sets
`physical_success` to `null`; a completed software task is not evidence of a
real robot action, scene change, grasp, or task success.
