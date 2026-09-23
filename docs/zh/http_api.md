# 推理 API v1

本页定义 EmbodiRun 与 EmbodiInfer 之间的推理协议。
请求包含编码图像、机器人状态、任务指令和会话标识；响应返回模型动作，
再由 EmbodiRun 的策略绑定转换为设备命令。

Agent 所使用的 Control API 是一个独立的接口。关于观测、
动作提议、执行与状态查询，请参阅 [Agent 执行流程](agent-workflow.md)
以及[公开客户端参考](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/agents/CLIENT.md)。

## 从动作提议到执行

Agent 调用 Control 的 `/v1/propose`，根据已有观测和文本指令请求动作提议。
Control 调用推理服务，将模型结果按绑定转换后返回，但不执行动作；执行需由 Agent 另行提交请求。
当前动作提议流程使用临时推理会话。有状态导航则需在连续推理时复用同一模型会话。

## HTTP 端点

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/healthz` | 检查服务健康状况。 |
| GET | `/v1/capabilities` | 检查适配器能力与动作空间。 |
| POST | `/v1/sessions` | 打开策略会话。 |
| POST | `/v1/sessions/{session_id}/steps` | 按会话步序执行一次推理。 |
| POST | `/v1/sessions/{session_id}/reset` | 清除会话状态并重新开始步编号。 |
| DELETE | `/v1/sessions/{session_id}` | 关闭会话。 |

配置认证后，在健康检查以外的请求上发送
`Authorization: Bearer <token>`。服务端设置与错误语义记录在
[EmbodiInfer 服务指南](https://embodiinfer.readthedocs.io/zh-cn/latest/serving/) 中。

使用包含 `schema: embodiinfer.policy.session.v1`、
`robot_id`、适配器的 `action_space` 以及可选 `metadata` 的 JSON 打开会话。使用
返回的 `session_id` 进行后续操作。

## 请求顺序与幂等处理

同一会话内的请求使用相同的 `session_id`，按顺序递增 `step_id`，每个新请求使用唯一的 `request_id`。
服务端按请求 ID 去重，确保同一步只提交一次会话状态。

## 发送推理请求

`POST /v1/sessions/{session_id}/steps` 使用 multipart 表单数据。

- `metadata`：`application/json`，schema `embodiinfer.policy.step.v1`
- `image_0..N`：编码后的 JPEG/PNG 字节

发送与元数据中 `request_id` 相等的 `Idempotency-Key`。元数据
包含 `session_id`、`request_id`、非负的 `step_id`、非空的
`instruction`、一个 `state` 对象、一个 `images` 列表以及可选的 `metadata`。
每个图像声明都有 `name` 和 `mime_type`；其位置对应
`image_0`、`image_1` 等。请求体与 URL 中的会话 ID 必须一致。

从 `step_id: 0` 开始，每完成一步再加一。传输失败时，使用相同 ID 和内容重试，
在结果确认前保持步编号不变。响应缓存有容量限制，旧响应被移出缓存后，对应的重试可能被拒绝。

## 解析推理结果

```json
{
  "request_id": "step-0-...",
  "session_id": "fr3-episode-1",
  "step_id": 0,
  "session_revision": 1,
  "action_space": "pi05.action_chunk.v1",
  "actions": [
    {
      "type": "action_chunk",
      "values": {
        "data": [
          [0.12, -0.03, 0.45, -2.21, 0.08, 1.12, -0.34, 0.04]
        ],
        "feature_names": [
          "joint_1",
          "joint_2",
          "joint_3",
          "joint_4",
          "joint_5",
          "joint_6",
          "joint_7",
          "gripper_width_m"
        ]
      }
    }
  ],
  "timing": {
    "policy_ms": 61.0
  }
}
```

EmbodiRun 按绑定逐步映射并校验 `pi05.action_chunk.v1` 中的动作，
取每个动作块的前 `run --chunk-steps` 步，按配置的控制频率执行。
机器人状态字段和动作维度由绑定定义。

## 使用 WirelessComm 传输

WirelessComm 使用相同的 `embodiinfer.policy.session.v1`、`embodiinfer.policy.step.v1` 与
结果 schema。RPC 元数据使用 schema `embodiinfer.policy.rpc.v1`、每次尝试的
`rpc_id`，以及 `health`、`capabilities`、`open_session`、
`step`、`reset` 或 `close` 方法之一。

发送推理请求时，元数据和编码图像组成一份结构化消息。
图像直接使用 WirelessComm 字节段传输，不转成 base64 或 HTTP multipart。
`request_id` 用于推理请求去重，`rpc_id` 则标识每次 RPC 尝试，两者用途不同。
