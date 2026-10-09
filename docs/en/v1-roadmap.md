# V1 open-source scope and delivery checklist

This page defines the public V1 scope, reproducible entry points, and evidence to collect. Missing measurements stay marked as pending; they are not replaced with estimates.

## Positioning

EmbodiRun separates robot control from model compute. Control can run beside the robot while inference runs on an edge GPU, a shared GPU service, or the cloud. A deployment can use independent backends, share one backend across devices, or mix both; one-controller/many-device and many-controller/many-device layouts are expressed in configuration. EmbodiInfer owns model execution, batching, and the serving contract; EmbodiRun owns devices, tasks, action execution, and lifecycle.

This separation fits economical robots and compute reuse: a high-end GPU does not need to be installed on every robot, and a workflow can be rehearsed with a fixture or simulator before hardware is connected.

## Three primary scenarios

| Scenario | Primary demo | Recipe extensions | Status |
|---|---|---|---|
| VLA | SO-101 grasping | ARX5, Franka, Songling | SO-101 configuration exists; others follow the support matrix |
| VLN | MicroDuck navigation | Go2 and other mobile platforms | Experimental example |
| Agent | Mobile base + SO-101 + RPent/Astra | Other mobile-manipulation combinations | End-to-end collection-to-deployment material is pending |

The home page keeps one primary demo per scenario. Other devices, models, and backends are exposed through `examples/` and configuration Recipes.

## Recipe prerequisites

Every hardware Recipe must state the Python/uv or Docker version, robot SDK, calibration files, camera-role mapping, motion limits, model checkpoint, checkpoint binding, action units, route or task files, model endpoint, emergency-stop procedure, and operator takeover path. `check` validates declarations and files; it does not prove device connectivity, calibration, model output, or task success.

## Evidence to publish

| Evidence | Record | Status |
|---|---|---|
| Hardware budget | `hardware.md` in the relevant Recipe | 🟨 Pending |
| First deployment time and manual steps | `benchmarks/deployment-readiness.md` | 🟨 Pending |
| CPU/GPU/VRAM/RAM usage | Raw benchmark data | 🟨 Pending |
| EmbodiRun versus manual deployment | `benchmarks/deployment-readiness.md` | 🟨 Pending |
| Inference, deployment, and multi-node scaling curves | Raw data and scripts under `benchmarks/` | Partial |
| WirelessComm latency, throughput, failure rate, recovery time | `benchmarks/inference-transport/` | Partial |

A chart without its raw data and generator script is not a final performance claim.

## Ecosystem boundary

EmbodiRun sits between application tasks and low-level drivers. Robot SDKs, ROS 2, camera drivers, and emergency stops remain device-side responsibilities; adapters, observation contracts, and action bindings connect them to the runtime. A formal ROS 2 bridge is still pending. The current documentation describes the boundary and does not claim that bridge exists.

## Contribution entry points

- [ ] `good first issue`: add a fixture or configuration validation
- [ ] `good first issue`: add a bilingual Recipe page
- [ ] `good first issue`: add raw data and a plotting script to an existing benchmark
- [ ] `help wanted`: collect deployment-cost and multi-node scaling evidence
- [ ] `help wanted`: implement hardware inventory, model/device candidate search, and an operator-optimization report

When submitting code, update the English page, Chinese page, support matrix, and Recipe prerequisites together. Keep a pending marker when a measurement is not available.
