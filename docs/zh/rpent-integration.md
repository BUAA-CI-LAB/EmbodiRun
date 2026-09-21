# 接入 RPent 任务规划器

RPent 负责任务规划与技能选择；EmbodiRun 负责部署服务并执行
动作。该集成通过 Control HTTP API 将二者连接起来。

## 注册机器人适配器

RPent 通过其顶层 `robots/<name>/` 包发现机器人。
在其中注册 `RobotSpec` 与 `Toolkit`，并使用参考客户端
[`agents/rpent`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/agents/rpent/README.md)。

RPent 侧适配器位于 **`BUAA-CI-LAB/RPent` 分支仓库**的
`embodirun-integration` 分支上：`robots/embodirun/` 实现了
`get_robot_spec`/`get_toolkit`，连接到 Control HTTP API，并从
`init_runtime` 返回
`([], runtime_kwargs)`。

## 接口对应关系

| RPent 接口 | EmbodiRun Control API |
|---|---|
| 机器人身份 / 能力 | `GET /v1/describe`（包括 `binding.kind`、`binding.maximum_chunk_steps`、`binding.action_feature_names`） |
| `reset()` | 仅 `observe()` — 零运动 |
| `step` / `chunk_step` | 一次有界的 `POST /v1/execute`，然后 `observe()` |
| `propose` | 针对保留的 `observation_id` 发起 `POST /v1/propose` |
| `inspect` / `cancel` / `stop` | `GET /v1/jobs/{id}`、`POST /v1/jobs/{id}/cancel`、`POST /v1/jobs/{id}/stop` |
| 任务成功 | 观测或任务元数据 |

执行行为遵循[客户端接口说明](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/agents/CLIENT.md)：
`observe` 和 `propose` 只读取观测或生成动作提议，`execute` 才会执行动作。
结果不明确时保留 `unknown`，不自动重发；取消后丢弃迟到的结果。
停止尚未确认时，即使客户端重启，也会保留未确认状态。

## 示例：π0.5 与仿真设备

本次运行将 RPent 连接到 Control 与一个 π0.5 推理服务。动作在
一台内存中的 `simulated.policy_vector` 设备上执行，该设备带两个合成摄像头
（`physical_robot=false`）。

运行配置：

| 组件 | 配置 |
|---|---|
| 推理主机 | 2×A100 80GB，服务位于 GPU0，`vvla-http`（`π0.5`） |
| 模型适配器 | `state_native`；摄像头 `observation.images.front` + `wrist`；`return_steps=50`；6 个动作特征 |
| Control | 来自部署 YAML 的本地 `ControlService`；通过 SSH 隧道访问外部推理端点 |
| 机器人 | `simulated.policy_vector` 绑定 `simulated.policy_vector.pi05` |
| RPent | `BUAA-CI-LAB/RPent` `embodirun-integration`，机器人 `--robot embodirun` |

输出节选：

```text
describe.binding: {"action_feature_names": ["shoulder_pan.pos", "shoulder_lift.pos",
  "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"],
  "kind": "simulated.policy_vector.pi05", "maximum_chunk_steps": 50}
reset.observation_id: 4090:8:...:o2
reset.task_success: unverified
proposal.status: proposed mapped_actions: 50
fresh observation_id: 4090:8:...:o7
chunk_step 5-tuple -> obs 0.0 False False
info: {"cancelled": false, "discarded_late_result": false,
  "execution_evidence_unknown": false,
  "request_id": "rpent-embodirun:execute:...", "status": "completed",
  "task_success": "unverified"}
inspect.status: completed
```

适配器在执行动作提议前会刷新观测，以避免
`observation_stale` 错误。该仿真设备报告 `task_success: unverified`。

## 应用配置

- **场景重置。** EmbodiRun 没有恢复场景的重置路由，因此适配器
  设置 `supports_exploration=False`，且 `reset()` 从不会恢复场景。
- **真实机器人。** `RobotSpec.is_real_robot` 由机器人包定义。此适配器面向仿真器；
  接入硬件时，需使用真实机器人包或明确配置硬件安全防护。
- **规划器。** 选择 `--robot embodirun`，配置 `--control-*` 标志，
  并提供规划器所需的 LLM 凭据。
- **RPent 基础安装。** RPent 的基础依赖集（例如
  `openai-codex`）必须能从你所使用的索引解析；从
  公共 PyPI 安装即可。
