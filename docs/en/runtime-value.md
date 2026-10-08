# Runtime responsibilities and deployment costs

EmbodiRun prepares configured nodes and environments, manages service processes,
and exposes shared device observations and bounded execution to applications.
Whether that reduces deployment or maintenance work needs a comparison with the
same model, inference backend, hardware, and workload. The existing
[SO-101 engine timings](demos/engine-e2e-contrast.md) compare inference engines
and optimizations; they do not measure the benefit of adding EmbodiRun.

## What LeRobot already provides

This comparison uses upstream **LeRobot v0.6.1**, rather than assuming that a
policy library lacks deployment tools.

| Need | Upstream implementation | EmbodiRun's responsibility |
|---|---|---|
| Evaluate a trained policy | [`lerobot-eval`](https://raw.githubusercontent.com/huggingface/lerobot/v0.6.1/src/lerobot/scripts/lerobot_eval.py) creates supported evaluation environments and collects episode metrics. | Configure supported device/simulator, model-service, and binding combinations; coordinate their execution. It does not replace policy training or dataset evaluation. |
| Deploy on a supported robot | [`lerobot-rollout`](https://raw.githubusercontent.com/huggingface/lerobot/v0.6.1/src/lerobot/scripts/lerobot_rollout.py) supports synchronous/RTC inference and autonomous, recording, and human-in-the-loop strategies. | Keep configured device owners and services running independently of an application's planning loop. |
| Remote policy inference | The [`async policy server`](https://raw.githubusercontent.com/huggingface/lerobot/v0.6.1/src/lerobot/async_inference/policy_server.py) loads policies and exchanges observations/actions over gRPC. | Connect Control to a compatible external endpoint, or prepare and supervise a configured managed provider using HTTP/WirelessComm. |
| Cross-node service operation | The policy and robot commands are available; the selected deployment still needs its environments, addresses, processes, and recovery procedure. | Host implements node preparation, source synchronization, health waits, logs, and identity-checked shutdown from deployment YAML. |
| Application/Agent access | A caller can use LeRobot's policy and robot APIs in its application. | The [Control client](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/agents/CLIENT.md) offers shared observations, proposals, bounded actions, job inspection, cancellation, and recording through one service boundary. |

The last two rows describe this repository's scope, not a measured claim that
LeRobot requires more work. See the current implementations of
[environment preparation](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/src/embodirun/deployment/operations/init.py),
[service startup](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/src/embodirun/deployment/operations/up.py),
and [providers](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/src/embodirun/model_services/providers.py).
Record the checked-out Git revision and local changes with each measurement.

## ROS 2, Agents, and EmbodiInfer

ROS 2 supplies a distributed robotics graph with topics, services, actions,
parameters, and discovery. Its launch system already starts and configures
multiple nodes and handles process events. This description is based on the
Jazzy documentation at commit
[`f8b12f7`](https://github.com/ros2/ros2_documentation/commit/f8b12f7b30ce1c0d7865bbadc0c210aec79114da):
[nodes](https://raw.githubusercontent.com/ros2/ros2_documentation/f8b12f7b30ce1c0d7865bbadc0c210aec79114da/source/Concepts/Basic/About-Nodes.rst)
and [launch](https://raw.githubusercontent.com/ros2/ros2_documentation/f8b12f7b30ce1c0d7865bbadc0c210aec79114da/source/Tutorials/Intermediate/Launch/Launch-Main.rst).

EmbodiRun currently has no packaged ROS 2 bridge. Using a ROS 2 robot would
require an adapter that maps observations, timestamps, command limits,
ownership, and stop/cancellation feedback to the existing device interface,
plus tests and device validation. ROS 2 interoperability remains an integration
task; the current HTTP/WirelessComm interfaces are not ROS 2 interfaces.

An Agent supplies the planning or review loop. Control owns device connections
and execution state; an accepted request is not a completed action or verified
physical stop. [Agent workflow](agent-workflow.md) explains how to inspect an
uncertain outcome. EmbodiInfer owns model loading, inference, and engine
optimization. Its model latency or memory improvements must be measured
separately from Host/Control deployment costs.

## When the extra runtime helps

The existing interfaces address deployments with several nodes or consumers,
an inference process separate from device ownership, or applications that need
to inspect and cancel jobs. A single supported robot running one LeRobot policy
may need fewer configuration files and processes through the upstream rollout
command. EmbodiRun adds YAML configuration, adapter/binding constraints,
deployment state, service processes, and network requests. Its supported
combinations are listed in the [support matrix](support-matrix.md); adding an
unsupported robot or backend still requires implementation and validation.

These tradeoffs are reasons to measure both routes, not predictions of savings.
Keep negative results, failed installations, retries, and recovery work in the
report.

## Measure six kinds of work

The [deployment measurement procedure](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/benchmarks/deployment-runtime/README.md)
contains executable commands for the current software scope, six paired
procedures, collection fields, and an empty report template.

| Scenario | Compare under the same starting and ending conditions |
|---|---|
| First deployment | Fresh task environment to the first ready observation; separate downloads, installation, model loading, warmup, and Control initialization. |
| Manual work | Operator actions, files edited, edit operations, help requests, and retries across all scenarios. Do not infer human effort from command count. |
| Change backend | The same initial backend A and final backend B, checkpoint, inputs, policy settings, and readiness condition in both routes. |
| Add a device | One to two equivalent devices, with the same placement and observation/inference workload. |
| Update and restart | The same source-version change; measure stopped time, preparation, readiness, and retained/reinitialized state. |
| Diagnose a failure | The same deliberate failure, time to identify its cause, and time to restore the same ready condition. |

**Results are pending.** The current repository has no ready native LeRobot
gRPC-to-EmbodiRun inference adapter or matched full comparison configuration.
Before a full model comparison, select a supported policy/device, obtain the
checkpoint and inputs, implement the required protocol and feature mapping,
and verify equivalent outputs and action limits. Those are explicit remaining
tasks; the engine-comparison Recipe cannot supply the missing adapter.
Its recorded native LeRobot/HTTP panel is a published measurement, not a
packaged native LeRobot launcher profile in this repository.

The smaller available procedure compares direct startup and Host management of
the **same real Control service with `simulated.joints` and a fake camera**.
It can measure service preparation, readiness, shutdown, and software failure
diagnosis without model assets. Model loading, warmup, backend changes, and
physical-device conclusions remain unavailable in that scope. No recorded
software or full-model cost results are supplied by the empty templates.
