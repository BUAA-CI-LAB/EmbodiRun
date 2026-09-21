# Comparing inference engines on SO-101

Run the same bounded pick-and-place task with three packaged profiles:

| Manifest | Model service | Transport |
|---|---|---|
| `embodiinfer-http.yaml` | EmbodiInfer π0.5 | HTTP |
| `embodiinfer-wireless.yaml` | EmbodiInfer π0.5 | WirelessComm |
| `sglang-http.yaml` | SGLang with LeRobot processors | HTTP |

The native LeRobot panel in the video requires a separate inference-service
adapter; it is not one of this launcher's profiles.

## Prepare matching configurations

Follow the [SO-101 preparation steps](../multi_robot_serving/README.md), using
one robot computer and one GPU host. Copy each selected manifest and its
matching `*.deployment.yaml` to `*.local.yaml`, then update the manifest's
`parameters.deployment` reference. For example:

```bash
cp examples/engine_comparison/embodiinfer-http.yaml examples/engine_comparison/http.local.yaml
cp examples/engine_comparison/embodiinfer-http.deployment.yaml examples/engine_comparison/http.deployment.local.yaml
```

Set `parameters.deployment: http.deployment.local.yaml`. Replace the deployment's
`REPLACE_*` fields. Use the same checkpoint, robot, calibration, image mapping,
instruction, 50-action horizon, and 20 Hz playback rate across profiles.
Set `metadata.deploy-commit` to the same published EmbodiRun commit in all three
deployments to pin the runtime and its EmbodiInfer source for the comparison.

EmbodiInfer requests BF16, 10 denoising steps, and full-loop CUDA graphs.
SGLang uses the `LeRobotPi05Pipeline` compatibility integration for checkpoint
processors and six-dimensional SO-101 output. Its optional environment is
installed separately by Host; prepare compatible CUDA/PyTorch packages on the
GPU host as described in the [installation guide](../../docs/en/installation.md).

## Run one profile at a time

```bash
CONFIG=examples/engine_comparison/http.local.yaml
bash examples/run.sh "$CONFIG" validate
bash examples/run.sh "$CONFIG" plan
bash examples/run.sh "$CONFIG" setup
bash examples/run.sh "$CONFIG" up --allow-hardware
bash examples/run.sh "$CONFIG" run --allow-hardware
bash examples/run.sh "$CONFIG" down
```

For warmup, temporarily set `parameters.chunks: 1`, run with the operator
present, and reset the cube afterwards. Restore `chunks: 15` for the task.
Keep the service running between warmup and the task; stop it before switching
engines. Warmup here executes robot actions, not just inference.

Each profile writes its own Host state, `run.json`, and command log under
`artifacts/examples/engine-<profile>/`. The same Wi-Fi connection can carry both
HTTP and WirelessComm; transport selection does not change the physical network.

For repeated inference-only timing without moving the arm, use the
[transport benchmark](../../benchmarks/inference-transport/README.md). Its
latency report and a physical task's completion time measure different things.

[Manifest reference](../README.md) ·
[Demo video](https://embodirun.readthedocs.io/en/latest/demos/engine-e2e-contrast/)
