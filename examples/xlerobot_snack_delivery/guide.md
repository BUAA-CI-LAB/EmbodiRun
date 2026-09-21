# XLeRobot setup and running guide

[Demo overview](README.md) · [Hardware and software](hardware.md) · [Agent interfaces](architecture.md)

## Bring-up and calibration

1. 按 [XLeRobot 官方装配资料](https://github.com/Vector-Wangel/XLeRobot) 搭建双轮底盘、双 SO-101 机械臂、供电和急停，安装前视、左右腕相机。硬件选择和预算见 [清单](hardware.md)。
2. 复制 [hardware.example.json](hardware.example.json)，填写稳定串口名、相机路径、SDK 路径、标定文件、轮向、轮径和轮距。保持 `allow_motion: false` 完成读数检查。确认相机角色、关节单位、限位和停止反馈后再开启运动；不要直接照搬示例轮向和递交姿态。
3. 在机器人 Linux 主机运行 `bash examples/run.sh examples/xlerobot_snack_delivery/example.yaml setup`。安装器用项目 `uv.lock` 安装核心和 Host 依赖；硬件 SDK、相机和录制 extras 由独立 owner 包安装。仅做离线演练时，使用 Host 环境即可。
4. 在 GPU 主机启动与该机器人训练配置匹配的 π0.5 服务，参考 [模型部署示例](../../configs/pi05/bi-so101-embodiinfer.yaml)。这里复用的是服务启动方式；本 recipe 的 binding 是 `lerobot.xlerobot.pi05`。模型须输出带 feature names 的本机标定绝对位置：关节 degrees、夹爪 range_0_100，左右各六维。通用权重或匿名 50×12 数组不能自动视为适配完成。
5. 复制 [deployment.example.yaml](deployment.example.yaml)，填写硬件配置路径、模型服务 endpoint 和 owner 相机角色。模型 endpoint 可经 SSH 转发到机器人主机。底盘和双臂 Control 默认分别使用 `127.0.0.1:8100`、`127.0.0.1:8101`；owner 使用 `8766`，提供不同的 `/robot/*` API。

先生成并校验配置，不启动硬件：

```bash
.venv-xlerobot-snack/bin/python examples/xlerobot_snack_delivery/services.py \
  --deployment examples/xlerobot_snack_delivery/deployment.local.yaml \
  --state-dir artifacts/examples/xlerobot-snack-delivery/services --write-only
```

启动唯一 owner 与两个 Control 服务，在该终端保持运行：

```bash
bash examples/run.sh examples/xlerobot_snack_delivery/example.local.yaml up --allow-hardware
```

`up` 自动生成仅本机可读的 owner 认证文件和 Control 配置。生成文件及日志默认放在 `artifacts/examples/xlerobot-snack-delivery/services/`。端口已被占用时命令拒绝启动，不会替换已有 owner。退出后检查急停和 owner 的停止反馈。

owner 页面使用 `service-0.log` 中的短期配对码登录，无需填写 API token。通过 owner 的显式停止操作取得已确认的停止反馈，再开始 recipe。仅观察到轮速为零不会清除 `stop_unconfirmed`。owner 的生命周期和操作入口见 [owner 文档](../../integrations/xlerobot_owner/README.md)。

## RPent 配置

[config.example.json](config.example.json) 的 `agent.factory` 指向 `agents.rpent.snack_agent:create_agent`。这个 RPent 侧适配器使用已有登录状态的 Codex/Astra，读取 Control 中同一个 observation 的图像，给出 `proceed` / `hold`、VLA 指令和短段步数；无相机、超时、错误身份或非法输出均停止。

把 `agent.options.codex_bin` 改为运行主机上可用的 Codex 路径。环境需事先具有 Astra 的访问权限；recipe 不接收 API key 或 token。若完整 RPent 服务在另一台主机，把 `agent.factory` 指向自己的 `module:create_agent`，并在该工厂中连接 RPent。接口及字段见 [架构说明](architecture.md)。

Agent 决策之后，recipe 等待一份更新的观测，再调用 VLA；VLA 返回后复查原观测的 freshness。推理超过 Control 的观测期限会停止，不会给旧动作换上新 observation_id。应通过模型部署优化满足该期限。

## Route input choices

### 录制路线

在自己的场地用 owner 的遥操作入口分别录制去程和返程。将底盘指令按录制频率重采样，导出为 `chunks`。每个 chunk 是一小段动作，长度 1–100；每段结束会请求停止并验证反馈。`navigation.control_hz` 必须与导出频率一致，`timestamp_s` 本身不控制播放间隔。若原始轨迹连续行驶，需在这些停顿点重新验证，不应无条件照搬长轨迹。

```json
{
  "name": "outbound",
  "source": "recorded_route",
  "fixture": false,
  "chunks": [{"actions": [{
    "timestamp_s": 0.0,
    "values": {"x.vel": 0.03, "theta.vel": 0.0},
    "metadata": {
      "action_space": "lerobot.xlerobot.external_owner.v1",
      "units": {"x.vel": "m/s", "theta.vel": "deg/s"}
    }
  }]}]
}
```

这里的一帧只演示格式，不是一条可行驶路线。配置 `navigation.routes.outbound` / `return` 指向本地文件。仓库自带零速度 fixture，只用于 dry-run；硬件模式拒绝它。到站由现场确认或部署提供的导航证据判断，动作发完不代表到站。

### 路线示意图

设置 `planning.mode: "diagram"`，指定 `planning.route_diagram`，并提供已经录制、标明方向和起终点的 `planning.segments`：

```json
{
  "mode": "diagram",
  "route_diagram": "/path/to/room.png",
  "segments": {
    "home_to_table": {"description": "交付点到取物桌，正向", "path": "/path/to/outbound.json"},
    "table_to_home": {"description": "取物桌到交付点，正向", "path": "/path/to/return.json"}
  }
}
```

RPent 根据图和片段注释选择去程、返程的片段序列；无法判断时返回 hold。无比例尺示意图不能直接生成安全的米制速度轨迹。需要直接从图导航时，由外部 RPent 导航 skill 提供经过本地验证的同格式动作，recipe 不实现导航算法。

## 运行场景

复制 recipe 配置，填写两个 Control endpoint、自己的路线、RPent 工厂和本机标定的 `handover.forward_pose` / `gripper_opening`。安装与脚本从仓库根目录执行：

```bash
CONFIG=examples/xlerobot_snack_delivery/example.local.yaml
bash examples/run.sh "$CONFIG" validate
bash examples/run.sh "$CONFIG" plan
bash examples/run.sh "$CONFIG" dry-run
bash examples/run.sh "$CONFIG" run --allow-hardware
```

- `validate` 校验配置，`plan` 预览运行命令与输出路径，均不连接设备、模型或 RPent。
- `dry-run` 使用标明 fixture 的状态和动作，只演练任务顺序。它不调用 Astra，也不证明模型或硬件可用。
- `run --allow-hardware` 先检查两个 Control 服务的 runtime、scope 和能力，然后执行“去程 → RPent 判断 → VLA 短段抓取 → 持物确认 → 返程 → 伸臂 → 松夹爪 → 接收确认”。人工确认之后会重新观测；只允许 dry-run 自动确认。

默认 evidence 路径位于 `robot.metadata`。owner 提供 `safety.base_control_ready`、`safety.stopped`、`safety.stop_confirmed`、`safety.arms_stop_confirmed` 和 `navigation.zero_velocity`。`grasp_confirmed` 和 `route_arrived` 默认需要人工现场确认，并明确记为 `manual_confirmation`；有传感器/导航检测器时可映射相应路径并清空人工确认列表。人工确认不能替代底盘安全和停止反馈。

每段必须先得到 `status=completed` 和 `dispatch_status=completed`。`stop_requested` 表示停止还在处理，recipe 会等待执行之后采集的状态和停止确认；`accepted`、`running`、`unknown`、`stop_unconfirmed` 不会直接推进。Ctrl-C 或异常会对同一个执行 request_id 请求停止，记录结果，并结束当前运行，不自动恢复或重放。

## 结果与证据

输出包括 `status.json`、`events.jsonl`、RPent 决策和 VLA 提议。保留未执行的 proposal 尾部用于分析，但执行只采用有界前缀。

`completed` 表示流程完成。抓到薯片、回到交付点和人收到物品需各自的现场证据；没有配置相应最终证据时，`task_success` 为 `unverified`，`physical_success` 为 null。软件测试、fixture 和 HTTP 200 都不能作为真机演示成功率。

对外分享视频时，标注路线来源、模型/检查点、计算设备、播放倍速；延迟和成功率附测量记录。不要分享生成的认证文件、室内私有地图、设备标识和原始现场录像中的个人信息。
