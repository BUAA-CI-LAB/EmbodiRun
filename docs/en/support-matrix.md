# Supported platforms and roadmap

Find a simulator, robot, or model below, then choose a deployment configuration.
The roadmap tracks the next integrations separately from the current catalog.

**✓ Software-tested** — implemented with automated interface tests.<br>
**◐ Experimental** — integration code is available; setup is platform-specific.<br>
**○ Planned** — on the roadmap, not yet implemented.

## Current support

<div class="grid cards support-grid" markdown>

-   ### :material-monitor: Simulators {#simulators}

    ---

    **✓ LIBERO**<br>
    Manipulation with π0.5; EmbodiInfer or SGLang.

    **◐ VLABench**<br>
    Manipulation with π0.5.

    **◐ Habitat · Isaac Sim**<br>
    Navigation with StreamVLN.

    [Browse simulator configurations](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/configs/simulation)

-   ### :material-robot: Robots {#robots}

    ---

    **✓ SO-101** — single-arm; [real-robot demos](demos/multi-robot-serving.md).<br>
    **✓ Bi-SO-101** — coordinated dual-arm policy.<br>
    **✓ Franka FR3** — adapter and π0.5 binding.

    **◐ ARX5** — DM0.5 binding.<br>
    **◐ Unitree Go2** — StreamVLN navigation.<br>
    **◐ XLeRobot** — external hardware owner; dual-arm π0.5 recipe.

    [Explore robot adapters](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/src/embodirun/robots)

-   ### :material-brain: Models {#models}

    ---

    **✓ π0.5**<br>
    SO-101, Bi-SO-101, FR3, and simulator bindings.

    **◐ DM0.5 · StreamVLN**<br>
    ARX5 manipulation and navigation integrations.

    **◐ LightNav-0**<br>
    External XLeRobot binding; model service supplied separately.

    [EmbodiInfer model catalog](https://embodiinfer.readthedocs.io/en/latest/models/)

</div>

## Deployment configurations

=== "Simulators"

    | Environment | Policy / backend | Setup |
    |---|---|---|
    | LIBERO | π0.5 / EmbodiInfer | [Configuration](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/simulation/libero-pi05-embodiinfer.yaml) |
    | LIBERO | π0.5 / SGLang | [Configuration](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/simulation/libero-pi05-sglang.yaml) |
    | VLABench | π0.5 / EmbodiInfer | [Configuration](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/simulation/vlabench-pi05-embodiinfer.yaml) |
    | Habitat | StreamVLN / EmbodiInfer | [Configuration](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/simulation/habitat-streamvln-embodiinfer.yaml) |
    | Isaac Sim | StreamVLN / EmbodiInfer | [Configuration](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/simulation/isaac-streamvln-embodiinfer.yaml) |

    Prepare the simulator assets, model checkpoint, and environment dependencies
    before starting a closed loop. Isaac Sim uses NVIDIA's Isaac Sim EULA.
    See [Installation](installation.md) for dependency groups.

=== "Robots"

    | Robot | Policy / interface | Setup |
    |---|---|---|
    | SO-101 | π0.5, independent single-arm runtimes | [Shared-service configuration](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/http-wireless-inference/http.yaml) |
    | Bi-SO-101 | π0.5, 12-dimensional dual-arm policy | [Dual-arm guide](pi05-bi-so101.md) |
    | Franka FR3 | π0.5 binding | [Adapter](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/src/embodirun/robots/franka/fr3) |
    | ARX5 | DM0.5 binding | [Binding](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/src/embodirun/bindings/arx/x5/dm05) |
    | Unitree Go2 | StreamVLN navigation | [Robot integration](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/src/embodirun/robots/unitree/go2) |
    | XLeRobot | External hardware owner; `lerobot.xlerobot.pi05` for two arms | [Recipe](xlerobot-snack-delivery.md) / [owner package](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/integrations/xlerobot_owner/README.md) |
    | SO-101 (wired teleoperation) | Multi-leader UDP fan-out, follower-side episode collection | [Integration package](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/integrations/so101_wired_teleop/README.md) |

    SO-101 deployments need calibrated arms, camera mappings, and a checkpoint
    trained for the selected binding. FR3 needs its robot SDK; ARX5 needs the
    vendor motor driver. Source links are starting points for custom integration.

    Follow [Safety](safety.md) and [Manual control](control.md) before operating
    physical hardware. [LightNav-0](lightnav0_xlerobot.md) documents the separate
    experimental XLeRobot waypoint binding.

=== "Models and transports"

    **EmbodiInfer** provides the first-party π0.5, DM0.5, and StreamVLN
    inference services. Its [serving guide](https://embodiinfer.readthedocs.io/en/latest/serving/)
    covers checkpoints and model-specific input mappings.

    **SGLang** has a separately installed π0.5 integration. **External
    services** connect through a provider with `lifecycle: external`;
    see [Configuration](configuration.md) and the
    [external-service example](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/examples/external.yaml).

    **HTTP** is the standard transport. **WirelessComm** is an experimental,
    separately installed option; see the
    [WirelessComm configuration](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/http-wireless-inference/wireless.yaml)
    and [transport measurements](inference-transport.md).

## XLeRobot recipe prerequisites {#xlerobot-recipe-prerequisites}

Start with the [software rehearsal](xlerobot-snack-delivery.md). It uses
fixtures and needs none of the physical inputs below. Native setup uses
Python 3.12 / uv 0.12.x; the `xlerobot-software` Docker service needs Docker
Engine and Compose on Linux. Native software/hardware profiles and the
XLeRobot container targets use `examples/xlerobot_snack_delivery/uv.lock`.
The hardware owner uses CPU PyTorch; inference runs in a separate model service.

| Input | When it is needed for your own robot | What software rehearsal uses |
|---|---|---|
| Owner environment / SDK | `setup --mode hardware` installs the owner. Set `sdk_src` to the locked LeRobot directory reported as `environment.sdk_src`; a custom SDK checkout needs its own version check. | Software dependencies; no owner or SDK import |
| Two-arm binding and checkpoint | Before connecting inference, match `lerobot.xlerobot.pi05` to an XLeRobot-adapted π0.5 checkpoint. This binding controls the arms; base routes use separate Control scope. | Fixture proposals; no checkpoint |
| Action feature names and units | Before model use, supply all 12 `left_arm_` / `right_arm_` fields: `shoulder_pan.pos`, `shoulder_lift.pos`, `elbow_flex.pos`, `wrist_flex.pos`, `wrist_roll.pos`, `gripper.pos`. Joint positions are `degrees`; grippers are `range_0_100`. | Named fixture values; no learned action-quality check |
| Camera mapping | Before hardware checks, map `observation.images.front`, `observation.images.left_wrist`, and `observation.images.right_wrist` to owner roles in deployment. Verify actual images and checkpoint input meanings before inference. | No physical cameras |
| Calibration and device paths | Before owner startup, provide your verified calibration, stable serial/camera paths, wheel geometry/directions, and limits in `hardware.local.json`. File existence does not establish calibration validity. | Template placeholders are allowed |
| Routes | `dry-run` accepts the supplied `fixture: true` routes. Hardware preparation needs locally recorded, directed non-fixture chunks; verify each route in the target space before motion. | Zero-motion fixtures |
| Handover | Before supervised delivery, validate `handover.forward_pose` and `gripper_opening` for your calibrated robot and workspace. | Fixture task sequence |
| Model endpoint | Set `deployment.local.yaml`'s `model.endpoint` before startup; verify the service and named outputs separately before model-driven motion. | No model connection |
| Physical stop | With the operator present, obtain fresh owner stop confirmation before motion and after stopping. An accepted stop request or zero wheel speed alone is insufficient. | Simulated feedback |

`check --mode hardware --json` reads local files and declarations. Exit 0 /
`status: "passed"` does not verify device connections, actual calibration,
camera images, model output, fresh observations, stop feedback, or task success.
The default `setup` / `check` mode is hardware; use `--mode software` explicitly
for rehearsal. Linux ARM64 dependencies and optional integrations need checks
on the target. A container image alone does not establish hardware/GPU support.

## Roadmap

<div class="grid cards support-grid" markdown>

-   :material-robot: __Robots__

    ---

    ○ **AgileX PiperX**

    - [ ] Robot connection and observation adapter
    - [ ] Policy-to-robot action binding
    - [ ] Calibration and deployment example

-   :material-brain: __Models__

    ---

    ○ **SmolVLA · OpenVLA**

    - [ ] EmbodiInfer inference adapters
    - [ ] Serving and observation mappings
    - [ ] EmbodiRun deployment bindings

    OpenVLA here means the base model; OpenVLA-OFT already has an
    EmbodiInfer inference adapter.

-   :material-monitor: __Simulators__

    ---

    ○ **Additional simulators**

    - [ ] Select the next environments
    - [ ] Observation and action adapters
    - [ ] Example deployments and closed-loop checks

</div>

## Agents and extensions

- **MicroDuck VLN:** an experimental [MuJoCo navigation example](microduck-vln.md)
  with a separate integration package, HTTP inference, and video recording.
- **XLeRobot snack delivery:** an experimental [task example](xlerobot-snack-delivery.md)
  combining recorded base routes, π0.5 arm proposals, and operator-confirmed handover.
- **Wired SO-101 teleoperation:** an experimental, separately installed
  [multi-leader integration](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/integrations/so101_wired_teleop/README.md)
  with UDP fan-out and follower-side recording.
- **Your planner:** the [public Python client](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/agents/CLIENT.md)
  exposes observation, proposal, execution, inspection, and cancellation.
- **RPent:** an experimental [agent integration](rpent-integration.md).
- **Astra + π0.5:** a [cooperative review loop](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/agents/astra_pi05/README.md)
  with an injectable reviewer; the example uses a mock reviewer.
- **Camera and transport work:** [camera-only capture](camera-only-experiments.md)
  and [transport experiments](transport-experiments.md) cover optional
  GStreamer, shared-memory, Zenoh, and NIXL paths.

For the demo deployments, [the shared launcher](examples.md) handles setup,
startup, and task execution. Node and GPU placement are set in the deployment YAML.
