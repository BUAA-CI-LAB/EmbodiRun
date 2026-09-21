# Benchmarks

Benchmarks measure a defined part of the system with repeatable inputs and
machine-readable results. For end-to-end robot and simulator tasks, use
[examples](../examples/README.md).

## Inference transport

[inference-transport](inference-transport/README.md) compares HTTP and
WirelessComm against the same model service and recorded observations.

- `benchmark.py run` performs measurements; `compare` checks comparability
  and prints results. `fixture` creates synthetic input for smoke checks.
- `capture_observations.py` collects camera/state input without commanding
  motion. It still accesses hardware and requires operator preparation.
- `profile_inference.py` serves and analyzes inference-stage timing.

These supporting scripts stay with their benchmark: their input formats and
timing definitions are specific to this experiment, not general utilities.

See the [running instructions](inference-transport/README.md) for preparation,
input capture, commands, and outputs. Reports go under ignored `artifacts/`
directories.
