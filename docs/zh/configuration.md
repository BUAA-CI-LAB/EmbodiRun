# 部署配置

用一份 YAML 配置计算节点、机器人或仿真器、传感器、推理服务及它们之间的连接。
从仓库中的完整示例开始，按实际部署替换占位符。

```text
metadata:   # 部署名称与固定的修订版本
nodes:      # 机器，以及 Host 如何访问它们
robots:     # 硬件适配器
simulators: # 仿真器适配器
sensors:    # 相机
models:     # 推理服务
runtimes:   # 机器人/仿真器 + 模型 + 绑定 + 传感器接线
```

`embodirun validate` 对文件做静态检查。`embodirun init` 记录文件的
摘要；若文件之后发生变更，`up` 会拒绝运行。

## 部署信息：metadata {#metadata}

| 字段 | 含义 |
|---|---|
| `name` | 部署名称；也用于本地状态和远程路径。 |
| `deploy-commit` | 节点使用的 EmbodiRun 版本；自动部署时填写有效的 Git 提交号。 |

## 计算节点：nodes {#nodes}

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
      host: 10.0.0.10          # 用于 SSH 的管理地址
      port: 22
      username: operator
      accept_new_host_key: false
    address: 192.168.1.20      # 向对等节点通告的数据面地址
```

`connection.type` 为 `local` 或 `ssh`。SSH 支持 `identity_file`、
`password_env`、`connect_timeout_s`、`command_timeout_s`，以及可选的
`proxy_command`。当 SSH 主机是管理地址或隧道时，`address` 会覆盖向
对等节点通告的数据面地址。

## 机器人：robots {#robots}

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

双臂机器人需声明 `left_port` 和 `right_port`，以及
`left_calibration_id` 和 `right_calibration_id`。`step_limit_mode` 为 `clip`
（逐值限幅）或 `reject`（拒绝超限目标）。限幅约束每步的运动变化量，不负责避障。
共用同一物理设备管理进程的机器人通过 `resource` 分组。

ARX5 还使用 `sdk_path`，并且要求 `operator_confirmed: true` 才能建立
物理连接。

## 相机：sensors {#sensors}

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

每个传感器运行在其所在节点上，并由运行时的 `inputs` 映射到策略的图像
字段。

## 推理服务：models {#models}

```yaml
models:
  pi05:
    backend: embodiinfer            # provider：embodiinfer 或 sglang
    transport: http          # http 或 wireless
    type: pi05
    node: compute
    gpu: cuda:0
    environment: .venv-embodiinfer-pi05
    source: /models/pi05-checkpoint
    server:
      bind: 0.0.0.0          # 在可信网络中可从机器人节点访问
      port: 8000
    policy_kwargs:
      attention: eager
    server_args:
      - --dtype
      - auto
```

| 字段 | 含义 |
|---|---|
| `backend` / `provider` | 推理后端：`embodiinfer`（EmbodiInfer）或 `sglang`。 |
| `transport` | `http` 或 `wireless`。 |
| `lifecycle` / `service` | `managed`（由 Host 启动）或 `external`（由你运行）。 |
| `endpoint` | 外部服务必填。 |
| `node`、`gpu`、`environment`、`source` | 服务所在节点、GPU、环境路径和检查点。 |
| `server` | 推理服务的监听地址 `bind` 和端口 `port`。 |
| `policy_kwargs` | 额外的模型参数，校验后合并到生成的适配器配置中。 |
| `image_keys` | 将策略图像字段映射到检查点的特征名。 |
| `adapter_config` | 手动指定适配器 JSON 文件，替代自动生成的配置。 |
| `environment_packages` | 要在模型环境中安装的额外包。 |

`policy_kwargs` 必须是以非空字符串为键的对象，并且需要由绑定生成的
适配器配置。

## 运行时：runtimes {#runtimes}

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

运行时把机器人或仿真器、模型和策略绑定组合起来，通过 `inputs` 指定传感器与模型输入的对应关系，
并提供供 Host 调用的 Control HTTP API。只访问设备、不运行模型时，可以省略模型配置；参见
[`device-only.yaml`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/examples/device-only.yaml)。
无线运行时还需声明 `inference_client.bind` 和
`inference_client.port`，并且节点上的每个监听器必须使用不同的端口。

## 扩展推理后端：provider {#provider}

每个推理后端通过 provider 注册，声明支持的传输协议、启动命令、依赖组和模型参数。
内置后端为 `embodiinfer` 和 `sglang`。接入新后端时，注册 provider 并提供对应客户端即可复用现有部署和执行流程。

## 应用到实际部署

将示例中的 `REPLACE_ARM`、`/models/...` 和内网 IP 替换为实际值。
本页只展示各字段的用法，完整配置请从仓库示例复制。
所有引用的相机都需在配置中声明，映射名称应与检查点一致。
使用远程推理时，确认 Control 节点能访问模型服务的监听地址；回环地址只能从本机访问。

用 `validate` 检查配置，用 `probe` 检查节点连通性，然后在执行前
检查启动日志和相机观测。
