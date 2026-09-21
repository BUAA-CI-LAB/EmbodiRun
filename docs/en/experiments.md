# Experiments

These experiments explore inference transport, camera capture, and robot
integration. Each page describes the setup, measurements, and findings.

For installation and operation, start with [Quick start](quickstart.md).
The [support matrix](support-matrix.md) lists available integrations.
Each experiment specifies its inputs and whether it uses replay, simulation,
or physical devices.

## Choose an experiment

| Note | Question | Current conclusion |
|---|---|---|
| [Inference transport](inference-transport.md) | How does transport affect inference latency? | WirelessComm gave roughly 4.5% higher throughput with small JPEG payloads on the tested LAN; model execution and queueing dominated latency. |
| [Camera-only capture](camera-only-experiments.md) | How do capture load and connection faults affect observation delivery? | A read-only camera publisher supports paired replay measurements, live capture, and TCP-disconnect tests. |
| [Transport candidates](transport-experiments.md) | Do shared memory, raw frames, GStreamer, Zenoh, or NIXL/UCX reduce latency? | Implemented and locally tested as opt-in paths. None has yet demonstrated a speedup or better recovery on the tested hardware. |
| [LightNav-0 with XLeRobot](lightnav0_xlerobot.md) | Can decoded local waypoints drive an XLeRobot base? | The binding and its CPU tests exist. Real-checkpoint evaluation, navigation success, and physical-robot validation are outstanding. |

## Comparing configurations

Use the hardware, software versions, and workload listed on each page as the
starting point for your own measurements. Transport comparisons also record
input fingerprints to check that both paths receive the same data.
