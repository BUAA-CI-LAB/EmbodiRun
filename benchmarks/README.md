# Benchmarks

Measure inference, transport, and runtime behavior with defined workloads and
machine-readable results. For robot and simulator task setup, use
[Recipes](../examples/README.md).

## Available measurements

| Benchmark | Workload | Results |
|---|---|---|
| [SO-101 engine comparison](engine-comparison/README.md) | π0.5 grasping with EmbodiInfer HTTP/WirelessComm, SGLang, and native LeRobot | [Timing report](../docs/en/demos/engine-e2e-contrast.md), summary JSON, and SVG generator |
| [Inference transport](inference-transport/README.md) | HTTP / WirelessComm replay against the same model service and recorded observations | [Transport report](../docs/en/inference-transport.md) and measurement commands |
| [StreamVLN navigation](streamvln-navigation/benchmark.py) | Habitat closed-loop R2R; serialized B1 or explicit tensor batches 1/2/4/8 | Per-episode SR/SPL evidence, primitive-frame history, actions, token traces and model/transport timing |
| [ActiveVLN navigation](activevln-navigation/benchmark.py) | Habitat R2R/RxR with the checkpoint's action grammar; tensor batches 1/2/4/8 | Per-episode SR/SPL, executed trajectories, batch identity and timing |

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

For StreamVLN, use the benchmark model service in
`third_party/embodiinfer/benchmarks/streamvln-benchmark/serve.py`. Supply its frozen
configuration and readiness JSON to the navigation runner, together with a
private token file. The manifest must contain exactly the declared number of
unique episodes; `--episode-limit` produces a smoke result, never a full report.

```bash
PYTHONPATH=src python benchmarks/streamvln-navigation/benchmark.py \
  --root artifacts/manifests --episode-count 256 --datasets r2r \
  --habitat-python /path/to/habitat/bin/python --scenes /path/to/scene_datasets \
  --config artifacts/model.json --endpoint http://127.0.0.1:28290 \
  --service-ready artifacts/model-ready.json --token-file artifacts/service-token \
  --simulator-gpu 0 --output artifacts/navigation
```

The protocol uses 0.25 m forward steps, 15° turns, at most 500 primitive steps,
and STOP within 3 m for success. StreamVLN's four-symbol chunks use metres for
forward movement. Simulator and HTTP latency are recorded separately from model
latency. Keep checkpoint, precision, sensor settings and manifest identical when
comparing devices. The runner transmits every intervening primitive observation;
the model resets its fast window every 32 executed primitive actions, independently
of HTTP request count. Images and generated tokens are hashed or recorded so an
early trajectory divergence can be located.

For explicit tensor batches use the model-side `serve_batch.py` and matching
`--batch-size 1`, `2`, `4`, or `8`. Every live episode retains its own session,
slot and primitive-frame history. All responses in a round must share one tensor
invocation ID; a split round fails the run. Summaries count a batch's model time
once and report actual occupancy and amortized throughput separately. Per-request
HTTP queueing and transport timers are not required to match across batch rows.

For Jetson validation, copy the complete checkpoint to the Jetson's local NVMe
before starting the model service. Dereference source symlinks during the copy,
verify every shard and configuration file against the checkpoint SHA-256
manifest, and record the resolved filesystem and device. Do not use an SSHFS or
other remote checkpoint mount for this validation. A simulator on a separate
host may send observations over HTTP; the Jetson model service must read its
weights locally. Record transfer, model loading and warmup durations separately
from inference timing.

The ActiveVLN runner accepts the same external service arguments and obtains its
action grammar from the frozen configuration's `action_space` (`r2r` or `rxr`).
`--action-space` may explicitly confirm that value; mismatches are rejected.
The action-space names identify grammars, not evaluation datasets: `r2r` uses
15° primitives and the optional `rxr` grammar uses 30°. The published ActiveVLN
R2R and RxR checkpoints both use `action_space: r2r`: the official
[RxR evaluation script](https://github.com/arvillion/ActiveVLN/blob/3a0c63b00e4f42c828cc74c3554afce17641da60/examples/vlnce/eval_rxr.sh)
retains the evaluator's R2R default, and its
[simulator configuration](https://github.com/arvillion/ActiveVLN/blob/3a0c63b00e4f42c828cc74c3554afce17641da60/vlnce_server/VLN_CE/habitat_extensions/config/vlnce_task_activevln_rxr_test.yaml)
sets 15° turns. Evaluating the RxR checkpoint on RxR therefore requires its
dataset-matched weights with the R2R prompt and 15° simulator steps. Do not infer
the action grammar from `--datasets`. Both grammars use 0.25 m forward primitives
and a 500-step cap. Record the checkpoint, evaluation split and action grammar
separately, and calibrate INT8 with the same prompt used for inference.
The legacy HTTP action schema name is retained for compatible numeric chunks;
the report's `checkpoint_action_grammar` and `turn_degrees` identify their meaning.
For `--batch-size 2`, `4`, or `8`, the runner maintains independent episodes and
requires all responses in each round to share one model tensor batch. Concurrent
batch-one HTTP calls do not satisfy this check.

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

## Deployment work procedure

[Deployment and Runtime work](deployment-runtime/README.md) supplies six paired
procedures, an existing-command software service comparison, and empty CSV,
environment and report templates. An Agent has executed
[three prepared Control startup pairs and a separate port fault/recovery](deployment-runtime/results/2026-10-08-thor-software/report.md)
on Thor; these coarse software samples do not establish a performance benefit.
It separates direct
versus Host management of the same simulated Control service from the full
same-model LeRobot comparison, which still needs assets and a compatible adapter.
中文职责说明：[Runtime 的职责与部署成本](../docs/zh/runtime-value.md)。

## Planned benchmarks

| Benchmark | Workload | Status |
|---|---|---|
| [Deployment and Runtime work](deployment-runtime/README.md) | Same-service preparation, readiness, updates and diagnosis; six full-policy deployment costs | [Prepared software lifecycle samples](deployment-runtime/results/2026-10-08-thor-software/report.md) available; full same-model LeRobot costs remain pending |
| Client scaling | 1, 2, 4, and 8 clients sharing one service; `--max-batch 1` and opt-in π0.5 batching | 🟨 待补充 |
| Physical-node scaling | 1, 2, and 4 inference hosts with a fixed checkpoint and recorded client placement | 🟨 待补充 |
| Archived transport results | Public raw replay reports and a table/figure generator | 🟨 待补充 |

For deployment comparisons, separate environment preparation, model loading,
compilation/warmup, and control initialization. Use the same readiness checks
for every comparison route. Measure Runtime overhead with identical observations,
inference endpoints, action horizons, and a simulated executor.

Scaling sweeps should report aggregate and per-client throughput, P50/P95
latency, queueing, failures, GPU utilization, and memory. Record `--max-wait-ms`
and keep checkpoint, dtype, decode steps, and observation inputs fixed. Report
replay and action-playback workloads separately, since playback can limit
closed-loop throughput.
