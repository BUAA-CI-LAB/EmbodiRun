# Benchmarks

Measure inference, transport, and runtime behavior with defined workloads and
machine-readable results. For robot and simulator task setup, use
[Recipes](../examples/README.md).

## Available measurements

| Benchmark | Workload | Results |
|---|---|---|
| [SO-101 engine comparison](engine-comparison/README.md) | π0.5 grasping with EmbodiInfer HTTP/WirelessComm, SGLang, and native LeRobot | [Timing report](../docs/en/demos/engine-e2e-contrast.md), summary JSON, and SVG generator |
| [Inference transport](inference-transport/README.md) | HTTP / WirelessComm replay against the same model service and recorded observations | [Transport report](../docs/en/inference-transport.md) and measurement commands |

Model-only benchmarks are maintained in
[EmbodiInfer](https://github.com/BUAA-CI-LAB/EmbodiInfer/tree/main/benchmarks).
中文结果：[引擎对比](../docs/zh/demos/engine-e2e-contrast.md) ·
[推理传输](../docs/zh/inference-transport.md)。

## Run a measurement

The [transport guide](inference-transport/README.md) covers input preparation,
measurement, and report comparison:

- `benchmark.py run` measures a replay workload; `compare` checks comparability
  and prints results. `fixture` creates synthetic smoke-check inputs.
- `capture_observations.py` records camera and state inputs from a prepared
  device without commanding motion.
- `profile_inference.py` serves and analyzes inference-stage timings.

Working reports are written under ignored `artifacts/` directories.
Publishable results and figure generators belong beside their benchmark.

## Measurement contract

A result should include the commands, source samples, and configuration needed
to reproduce it:

| Record | Contents |
|---|---|
| Samples | JSON/JSONL or CSV timings, units, trial/client identifiers, completion status, and failures |
| Environment | Hardware, network topology, source revisions, checkpoint/input hashes, dtype, and engine flags |
| Workload | Warmup, measured iterations, concurrency, batch limit, collection window, action horizon, and seed |
| Report | Sample count, P50/P95 latency, completed-request throughput, failure rate, and variation across repeated runs |
| Figure | Deterministic generation command and links to the input data |

Specify timing boundaries for each metric. Keep credentials and private
calibration data local, and record unavailable metadata explicitly.

## Planned benchmarks

| Benchmark | Workload | Status |
|---|---|---|
| Deployment and Runtime comparison | EmbodiRun Host and explicit SSH/process scripts launching the same service topology; cold preparation, warm readiness, and replay-loop overhead | 🟨 待补充 |
| Client scaling | 1, 2, 4, and 8 clients sharing one service; `--max-batch 1` and opt-in π0.5 batching | 🟨 待补充 |
| Physical-node scaling | 1, 2, and 4 inference hosts with a fixed checkpoint and recorded client placement | 🟨 待补充 |
| Archived transport results | Public raw replay reports and a table/figure generator | 🟨 待补充 |

For deployment comparisons, separate environment preparation, model loading,
compilation/warmup, and control initialization. Use the same readiness checks
for every baseline. Measure Runtime overhead with identical observations,
inference endpoints, action horizons, and a simulated executor.

Scaling sweeps should report aggregate and per-client throughput, P50/P95
latency, queueing, failures, GPU utilization, and memory. Record `--max-wait-ms`
and keep checkpoint, dtype, decode steps, and observation inputs fixed. Report
replay and action-playback workloads separately, since playback can limit
closed-loop throughput.
