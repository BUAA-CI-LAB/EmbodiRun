# Inference transport benchmark

Measures one PI0.5 inference service reached from the SO101 control nodes over
the two supported transports, so that HTTP and WirelessComm can be compared on
the same checkpoint, input and hardware. The same directory also carries an
inference-side profiler that breaks a single request into stages.

The method, the timing definitions and the measured results are in
[Inference transport](../../docs/en/inference-transport.md). This file only
describes how to run the scripts.

## Scripts

| Script | Purpose |
|---|---|
| `benchmark.py` | Drives the measurement. `run` distributes the worker and input over SSH to the configured control nodes, releases both ends from one barrier, and writes a JSON report. `compare` prints a table for an HTTP and a WirelessComm report. `fixture` writes synthetic PNG input for a link smoke test. |
| `profile_inference.py` | Inference-side staging. `serve` loads the checkpoint once and switches transport on stdin, so HTTP and WirelessComm reuse the same model, processor and CUDA graph. `analyze` aggregates per-stage times from the reports. |
| `capture_observations.py` | Records real SO101 front/wrist frames and calibrated joint state as replay input. Opens the arm `read_only=True`: it never writes a motor register and refuses action and hold commands. |

None of the scripts constructs a binding that moves hardware. `benchmark.py`
uses the inference client and the value mapper only; it never opens a serial
port or a camera.

## Configuration

Both transports run one at a time on the same GPU:

- `configs/http-wireless-inference/http.yaml`
- `configs/http-wireless-inference/wireless.yaml`

The two files must agree on checkpoint, dtype, denoising steps, CUDA graph
setting and action horizon; only the transport differs. `compare` refuses a
pair of reports whose recorded inputs, clients, model settings or revisions
differ.

## Input

Observations are JSONL, one SO101 observation per line, with image paths
resolved relative to the manifest. Every image must be an encoded JPEG or PNG:

```json
{"instruction":"Pick up the cube.","source":"recorded-so101","state":{"joint_positions_deg":[0,0,0,0,0],"gripper_position":0},"images":{"observation.images.front":"front.jpg","observation.images.wrist":"wrist.jpg"}}
```

To generate input without hardware:

```bash
uv run python benchmarks/inference-transport/benchmark.py fixture \
  --output artifacts/so101-input
```

Synthetic PNG rows exercise the link only. Their encoded size is not
representative of a real camera, so they must not appear in a reported result.

To record real input, start from a generated control configuration on the robot
node and run `capture_observations.py` there with the robot environment's
Python. The output directory must not already exist:

```bash
python capture_observations.py \
  --config <generated>/control-<runtime-id>.control.json \
  --output ~/so101-observations \
  --prompt 'Pick up the cube and place it in the bowl.' \
  --count 24 --interval 0.5
```

Capture needs the serial port and the cameras, so stop the Control service
first. Check the saved frames by eye before using them: a dark or blurred
recording is valid input for a link test but not for a performance claim.

## Run

The deploys are driven from a Host checkout with the deployment already
initialized. Stop any earlier deployment first so no service holds the ports.

```bash
uv sync --frozen

uv run embodirun --config configs/http-wireless-inference/http.yaml validate
uv run embodirun --config configs/http-wireless-inference/http.yaml init
uv run embodirun --config configs/http-wireless-inference/http.yaml up --wait-timeout 600
# The benchmark replaces the control client but keeps the model service.
uv run embodirun --config configs/http-wireless-inference/http.yaml down --target control
uv run python benchmarks/inference-transport/benchmark.py run \
  --config configs/http-wireless-inference/http.yaml \
  --observations artifacts/so101-input/observations.jsonl \
  --output artifacts/so101-http.json
uv run embodirun --config configs/http-wireless-inference/http.yaml down

uv run embodirun --config configs/http-wireless-inference/wireless.yaml init
uv run embodirun --config configs/http-wireless-inference/wireless.yaml up --wait-timeout 600
uv run embodirun --config configs/http-wireless-inference/wireless.yaml down --target control
uv run python benchmarks/inference-transport/benchmark.py run \
  --config configs/http-wireless-inference/wireless.yaml \
  --observations artifacts/so101-input/observations.jsonl \
  --output artifacts/so101-wireless.json
uv run embodirun --config configs/http-wireless-inference/wireless.yaml down

uv run python benchmarks/inference-transport/benchmark.py compare \
  --http artifacts/so101-http.json --wireless artifacts/so101-wireless.json
```

`run` reuses the Python environment recorded by `init` on each control node and
distributes the current worker source and the input over SSH, so nothing has to
be installed on the devices. Pass the same `--state-dir` as the Host when one
was used. `--runtime <id>` selects a single client instead of all of them.

Reports are written under `artifacts/`, which is not tracked. They record the
per-request latency, the sample index, the action row count, model timings, and
the input and source hashes that `compare` checks.

## Profiling one request

```bash
uv run python benchmarks/inference-transport/profile_inference.py analyze \
  --reports artifacts/so101-http.json artifacts/so101-wireless.json \
  --output artifacts/so101-profile-summary.json
```

`analyze` derives each stage per request before summarizing, keeps negative
differences, and never subtracts two P95 values. The stage definitions and the
measured breakdown are in
[Inference transport](../../docs/en/inference-transport.md).
