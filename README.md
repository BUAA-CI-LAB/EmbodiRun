<p align="center">
  <img src="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/logo.png" alt="EmbodiRun" width="440">
</p>

<h3 align="center">From model predictions to robot actions.</h3>
<p align="center">
  <a href="https://embodirun.readthedocs.io/">Documentation</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#demos">Demos</a> ·
  <a href="#performance">Performance</a> ·
  <a href="docs/en/support-matrix.md">Support matrix</a> ·
  <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python"></a>
  <a href="https://embodirun.readthedocs.io/"><img src="https://readthedocs.org/projects/embodirun/badge/?version=latest" alt="Documentation"></a>
</p>

**EmbodiRun is a deployment and execution runtime for embodied AI.** Describe
your devices, inference services, and compute nodes in YAML, then run the
observation–inference–action loop through a shared runtime. Keep control beside
the robot and place inference on a GPU host, or run both on one machine.

Use [EmbodiInfer](https://github.com/BUAA-CI-LAB/EmbodiInfer) as the inference
engine, connect an external model service, or bring an agent with its own
planning loop.

## Demos

| Three robots, one inference service | Comparing inference engines on SO-101 |
|---|---|
| [![Three SO-101 recordings](https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/multi_robot_serving.jpg)](https://embodirun.readthedocs.io/en/latest/demos/multi-robot-serving/) | [![Engine comparison on SO-101](https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/engine_e2e_contrast.jpg)](https://embodirun.readthedocs.io/en/latest/demos/engine-e2e-contrast/) |
| Three SO-101 arms using a shared π0.5 inference service, with one rollout process per device. | π0.5 on a Jetson AGX Thor with an SO-101 arm: EmbodiInfer over HTTP and WirelessComm, SGLang, and native LeRobot. |

| Fetching a snack with XLeRobot | Language-guided navigation with MicroDuck |
|---|---|
| [<img src="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/xlerobot_snack_delivery/xlerobot_snack_delivery_overview.jpg" alt="XLeRobot at the snack table" width="356">](https://embodirun.readthedocs.io/en/latest/demos/xlerobot-snack-delivery/) | [<img src="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/microduck_simulation_one_click_demo.jpg" alt="MicroDuck simulation views and execution log" width="200">](https://embodirun.readthedocs.io/en/latest/demos/microduck-vln/) |
| An agent connects base motion, VLA grasping, and the return trip through EmbodiRun. | Launch a MuJoCo navigation run with ActiveVLN, first- and third-person views, and step-by-step execution logs. |

Click a preview to watch the video and explore the setup.

[Reproduce the demos](docs/en/examples.md) with versioned YAML configurations and
the shared `examples/run.sh` launcher.

## Why EmbodiRun?

Keep control close to the robot. Share compute where it counts.

<table>
<tr>
<td width="50%" valign="top">
<h3>🌐 One config, multiple machines</h3>
<p>Place robot control on the edge and inference on a GPU host. One YAML describes the nodes, environments, devices, and bindings; the Host CLI handles preparation and service lifecycle.</p>
<a href="docs/en/architecture.md">Deployment architecture →</a>
</td>
<td width="50%" valign="top">
<h3>🦾 Multiple robots, shared inference</h3>
<p>Connect independent device loops to a shared model endpoint. Each robot keeps its own session and execution flow while using the same inference service.</p>
<a href="docs/en/demos/multi-robot-serving.md">See three SO-101 arms in action →</a>
</td>
</tr>
<tr>
<td valign="top">
<h3>🔌 Two transports, one contract</h3>
<p>Choose HTTP or WirelessComm for inference without changing the model-facing observation and action contract. Session and step semantics stay consistent across transports.</p>
<a href="docs/en/inference-transport.md">Transport design and measurements →</a>
</td>
<td valign="top">
<h3>📷 Capture once, reuse across consumers</h3>
<p>Shared camera and state snapshots feed inference, agent observations, and recording. The runtime owns device connections, so each consumer does not need to open the hardware again.</p>
<a href="docs/en/architecture.md">Device and observation ownership →</a>
</td>
</tr>
<tr>
<td valign="top">
<h3>🧩 Your planner, a ready robot API</h3>
<p>Observe, request policy proposals, execute actions, inspect jobs, and cancel through a dependency-free Python client. Bring your own planning loop; reuse the runtime underneath.</p>
<a href="agents/CLIENT.md">Agent client →</a>
</td>
<td valign="top">
<h3>🎛️ Execution with operator control</h3>
<p>Action validation, execution arbitration, and manual takeover sit between policy output and hardware. Robot adapters and policy bindings keep motion details out of application code.</p>
<a href="docs/en/safety.md">Execution controls and hardware setup →</a>
</td>
</tr>
</table>

## How it works

![Host deploys Control and inference; Control connects applications to robots and model services](docs/assets/runtime-overview.svg)

**Host** prepares and launches the deployment. **Control** owns robot connections,
observations, and action execution; **Simulation** serves simulator environments.
**Inference** turns observations into predictions. These services can run on
separate machines. See [Architecture](docs/en/architecture.md) for the full design.

## Performance

### π0.5 on a real SO-101 arm

On a Jetson AGX Thor, the recorded comparison reduced median inference latency
from **1,061 ms to 162 ms** and the complete control-loop chunk from
**3,592 ms to 2,660 ms**. Faster inference shortens the loop; action playback
still accounts for about 2.45 seconds per chunk.

| Engine | Transport | Inference latency | Full chunk time |
|---|---|---:|---:|
| **EmbodiInfer** | WirelessComm | **162 ms** | **2,660 ms** |
| EmbodiInfer | HTTP | 170 ms | 2,666 ms |
| SGLang | HTTP | 194 ms | 2,713 ms |
| Native LeRobot | HTTP | 1,061 ms | 3,592 ms |

Medians over 15 chunks per run, with the same SO-101 checkpoint, 10 denoising
steps, two cameras, and 50-step action chunks at 20 Hz. EmbodiInfer uses its
optimized path, SGLang uses upstream defaults, and LeRobot uses eager execution.
The [demo report](docs/en/demos/engine-e2e-contrast.md) describes the hardware,
engine settings, and timing breakdown.

For transport measurements, see the [HTTP/WirelessComm experiment](docs/en/inference-transport.md).
For model-only benchmarks, see [EmbodiInfer](https://github.com/BUAA-CI-LAB/EmbodiInfer#performance).

## Quick start

### Try the runtime on your laptop

Install from source with Python 3.10+ and [uv](https://docs.astral.sh/uv/) 0.12.x:

```bash
git clone https://github.com/BUAA-CI-LAB/EmbodiRun.git
cd EmbodiRun
uv sync --frozen
```

Try the local simulated-device walkthrough:

```bash
uv run python examples/run_shared_device_fake.py
```

It starts a local Control service with simulated joints and a fake camera,
exercises observation, execution, recording, and cancellation, and shuts the
service down. No robot, model checkpoint, or GPU is required.

### Connect your robot or simulator

Continue with [Quick start](docs/en/quickstart.md) to use the deployment CLI.
For hardware, choose a combination from the [support matrix](docs/en/support-matrix.md),
configure its devices and calibration, and read [Safety](docs/en/safety.md)
before execution.

## Support at a glance

✓ **Software-tested** · ◐ **Experimental** · ○ **Planned**

<table>
<tr>
<th align="left">🧪 Simulators</th>
<th align="left">🦾 Robots</th>
<th align="left">🧠 Models</th>
</tr>
<tr>
<td valign="top">
<p>✓ <b>LIBERO</b></p>
<p>◐ VLABench<br>◐ Habitat<br>◐ Isaac Sim</p>
<a href="docs/en/support-matrix.md#simulators">Simulator setup →</a>
</td>
<td valign="top">
<p>✓ <b>SO-101</b> · real-robot demos<br>✓ <b>Bi-SO-101</b><br>✓ <b>Franka FR3</b></p>
<p>◐ ARX5<br>◐ Unitree Go2<br>◐ XLeRobot</p>
<a href="docs/en/support-matrix.md#robots">Robot setup →</a>
</td>
<td valign="top">
<p>✓ <b>π0.5</b></p>
<p>◐ DM0.5 · ARX5 binding<br>◐ StreamVLN · navigation<br>◐ LightNav-0 · external binding</p>
<a href="docs/en/support-matrix.md#models">Model connections →</a>
</td>
</tr>
</table>

These are EmbodiRun integration statuses. Choose a model–device pairing in the
[deployment recipes](docs/en/support-matrix.md#deployment-recipes).
For an experimental MuJoCo navigation workflow, see the
[MicroDuck VLN recipe](docs/en/microduck-vln.md).
For the inference engine's broader model catalog, see
[EmbodiInfer](https://github.com/BUAA-CI-LAB/EmbodiInfer#supported-models).

**Bring your own application:** use the [Python client](agents/CLIENT.md),
the experimental [RPent adapter](docs/en/rpent-integration.md), or connect an
[external inference service](docs/en/http_api.md).

### Planned support

- [ ] 🦾 **AgileX PiperX** — robot adapter and policy binding.
- [ ] 🧠 **SmolVLA** — inference adapter and deployment integration.
- [ ] 🧠 **OpenVLA** — base-model support, separate from the existing OpenVLA-OFT inference adapter.
- [ ] 🧪 **More simulators** — next targets to be selected.

Implementation steps and integration ownership are tracked in the
[roadmap](docs/en/support-matrix.md#roadmap).

## Documentation

| Task | Guide |
|---|---|
| Install and run | [Installation](docs/en/installation.md) · [Quick start](docs/en/quickstart.md) |
| Configure a deployment | [Configuration](docs/en/configuration.md) |
| Operate devices | [Control](docs/en/control.md) · [Safety](docs/en/safety.md) |
| Extend the runtime | [Architecture](docs/en/architecture.md) · [Python API](docs/en/api.md) |
| Review evidence | [Support matrix](docs/en/support-matrix.md) · [Experiments](docs/en/experiments.md) |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and checks.
Use [GitHub issues](https://github.com/BUAA-CI-LAB/EmbodiRun/issues) for bugs and
feature requests, and follow [SECURITY.md](SECURITY.md) for private vulnerability
reports. Community participation follows our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Apache-2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE), and
[third-party notices](THIRD_PARTY_NOTICES.md). Model weights, datasets, robot
SDKs, and simulators retain their own licenses.
