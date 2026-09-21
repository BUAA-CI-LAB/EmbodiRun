# Configuration

One deployment YAML describes nodes, robots or simulators, sensors, inference
models, and runtimes. Start from a checked-in example and replace the
placeholders.

```text
metadata:   # deployment name and pinned revisions
nodes:      # machines and how Host reaches them
robots:     # hardware adapters
simulators: # simulator adapters
sensors:    # cameras
models:     # inference services
runtimes:   # robot/simulator + model + binding + sensor wiring
```

`embodirun validate` checks the file statically. `embodirun init` records a
digest of the file; `up` refuses to run if the file changed afterwards.

## metadata

| Field | Meaning |
|---|---|
| `name` | Deployment name; also used for local state and remote paths. |
| `deploy-commit` | EmbodiRun revision to check out on nodes. Use a real commit for managed deployments. |

## nodes

```yaml
nodes:
  robot-compute:
    type: workstation
    connection:
      type: local
  compute:
    type: jetson.agx-thor-128gb
    connection:
      type: ssh
      host: 10.0.0.10          # management address used for SSH
      port: 22
      username: operator
      accept_new_host_key: false
    address: 192.168.1.20      # data-plane address advertised to peers
```

`connection.type` is `local` or `ssh`. SSH supports `identity_file`,
`password_env`, `connect_timeout_s`, `command_timeout_s`, and an optional
`proxy_command`. `address` overrides the data-plane address advertised to
peers when the SSH host is a management address or tunnel.

## robots

```yaml
robots:
  so101-1:
    type: lerobot.so101
    node: robot-compute
    port: /dev/serial/by-id/REPLACE_ARM
    calibration_id: follower
    calibration_dir: /path/to/lerobot/calibration/robots/so_follower
    disable_torque_on_disconnect: true
    max_joint_step_deg: 5.0
    max_gripper_step: 10.0
    step_limit_mode: clip
```

A dual-arm robot declares `left_port` and `right_port` plus
`left_calibration_id` and `right_calibration_id`. `step_limit_mode` is `clip`
(bound each value) or `reject` (refuse an oversized target). Clipping is a
per-row rate limit, not collision avoidance. `resource` groups robots that
share one physical device owner.

ARX5 additionally uses `sdk_path` and requires `operator_confirmed: true` for
physical connection.

## sensors

```yaml
sensors:
  front:
    type: v4l2
    node: robot-compute
    device: /dev/v4l/by-id/REPLACE_FRONT_CAMERA
    width: 640
    height: 480
    fps: 30
```

Each sensor runs on its node and is mapped to a policy image field by a
runtime's `inputs`.

## models

```yaml
models:
  pi05:
    backend: vvla            # provider: vvla or sglang
    transport: http          # http or wireless
    type: pi05
    node: compute
    gpu: cuda:0
    environment: .venv-vvla-pi05
    source: /models/pi05-checkpoint
    server:
      bind: 0.0.0.0          # reachable from the robot node on a trusted network
      port: 8000
    policy_kwargs:
      attention: eager
    server_args:
      - --dtype
      - auto
```

| Field | Meaning |
|---|---|
| `backend` / `provider` | Inference provider. `vvla` (EmbodiInfer) or `sglang`. |
| `transport` | `http` or `wireless`. |
| `lifecycle` / `service` | `managed` (Host starts it) or `external` (you run it). |
| `endpoint` | Required for an external service. |
| `node`, `gpu`, `environment`, `source` | Placement, GPU, environment path, checkpoint. |
| `server` | `bind` and `port` of the inference listener. |
| `policy_kwargs` | Extra policy options (inference service only); validated and merged into the generated adapter. |
| `image_keys` | Map policy image fields to checkpoint feature names. |
| `adapter_config` | Advanced: point at an explicit adapter JSON instead of a binding-generated one. |
| `environment_packages` | Extra packages to install in the model environment. |

`policy_kwargs` must be an object with non-empty string keys, and it requires a
binding-generated adapter configuration.

## runtimes

```yaml
runtimes:
  so101-1-runtime:
    robot: so101-1
    model: pi05
    binding: lerobot.so101.pi05
    server:
      bind: 127.0.0.1
      port: 8100
    inputs:
      observation.images.front: front
```

A model-backed runtime binds a robot or simulator to a model and a policy
binding, maps sensors to policy inputs, and exposes a Control HTTP API for the
Host. Device-only runtimes can omit the model; see
[`device-only.yaml`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/examples/device-only.yaml).
Wireless runtimes additionally declare `inference_client.bind` and
`inference_client.port`, and each listener on a node must use a distinct port.

## Providers

Inference backends are registered providers. A provider declares which
transports it supports, its managed launch command, its environment group, and
which model options it accepts. The built-in providers are `vvla` and `sglang`.
Adding a backend means registering a provider and shipping a client; the
deployment and execution path does not change.

## Portability

Public examples use placeholders such as `REPLACE_ARM`, `/models/...`, and
private RFC1918 addresses. Replace these values for your deployment.
The fragments on this page explain fields; use a complete checked-in example
as the starting file. Every referenced camera must be declared, and its mapping
must match the checkpoint. For remote inference, the model listener must be
reachable from the Control node; a loopback listener only accepts local traffic.

Check the configuration with `validate`, use `probe` for node connectivity,
then inspect startup logs and camera observations before execution.
