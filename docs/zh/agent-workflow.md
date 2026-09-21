# Agent 执行流程

Agent 通过 Host 命令行读取观测、提交动作、查询执行状态和录制数据。
本页介绍 `describe`、`observe`、`media`、`execute`、`inspect`、`cancel`、`stop`
及录制命令的使用流程。Control API 的完整定义见
[客户端参考](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/agents/CLIENT.md)中；
硬件操作见[手动控制](control.md)和
[安全指南](safety.md)。

## 通过 Host CLI 连接

Host 根据部署 YAML 和初始化记录确定 Control 服务地址，通过 SSH 隧道或本机回环地址连接服务。
设备连接、共享观测和动作队列都由 Control 管理，Agent 只需调用这套接口。

控制和录制命令支持 `--json`，调用方可直接解析标准输出中的 JSON。
如需先验证调用流程，可以运行
[`examples/shared-device-fake.md`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/shared-device-fake.md)。
该示例使用模拟设备启动 `ControlHttpServer`，无需机器人或模型检查点。

## 保持调用方与会话标识一致

同一工作流中的调用应使用相同的 `--caller-id` 和 `--session-id`。
服务根据这两个标识判断控制权、可访问的观测和可取消的任务；中途更换标识会失去与原有状态的关联。

每个新动作使用新的 `--request-id`，后续查询、取消或停止该动作时沿用这个 ID。
它也是幂等键：同一请求只会提交一次，使用相同 ID 重试不会重复执行动作。

## 读取观测

用 `describe` 查看运行时支持的功能和当前控制权限，再用 `observe` 获取共享观测及其 `observation_id`。
`media` 可按这个 ID 查询图像；加上 `--include-data` 才会返回编码后的帧数据。
如果观测已过期或不存在，重新调用 `observe`。

## 提交动作

`execute` 从文件读取动作 JSON，也支持用 `--action -` 从标准输入读取。
动作发往机器人前，会先按绑定配置检查限位：

```sh
printf '%s\n' '{"timestamp_s":0,"values":{"type":"joint_position","joint_positions_deg":[0,0,0,0,0],"gripper_position":25},"metadata":{"action_space":"simulated.so101.position.v1"}}' \
  | embodirun --config "$CONFIG" execute --runtime "$RUNTIME" \
      --caller-id "$CALLER" --session-id "$SESSION" \
      --request-id action-1 --action - --steps 1 --json
```

用 `--observation-id` 指定动作依据的观测，用 `--max-age-ns` 和 `--max-skew-ns`
限制观测时效和时间偏差。省略这些参数时，运行时使用最新的共享观测。

## 查询执行状态

`execute` 返回请求已接受时，动作可能仍在执行。如果请求
超时，将结果视为未知：保留原始请求 ID，运行
`inspect`，并根据报告的状态作出判断。不要在
超时后重发动作——用新 ID 重发可能执行两次，用
相同 ID 重发只会重新读取已提交的结果。

提交结果不明确时，使用 `inspect` 查询。向上层返回结果时，保留
`status`、`physical_status`、`error`，以及相关任务和媒体信息。

## 取消或停止请求

调用 `cancel` 或 `stop` 时，提供该任务的调用方、会话和请求 ID，并检查返回状态。
如果连接中断，重连后查询任务，确认停止是否完成。

## 通过服务录制

`recording-status` 只查询录制状态，不改变控制权。服务支持录制时，可用
`recording-start`、`recording-stop` 和 `recording-get` 启停录制并获取结果。
录制器由服务统一管理；未配置录制器时返回 `unsupported`，客户端无需另建录制器。

## 处理异常状态

动作是否超限、观测是否过期、控制权是否被占用，以及停止和录制是否完成，都以 Control 的返回结果为准。
应用应保留 `unknown`、`uncertain`、`stop_unconfirmed`、`stale`、`busy` 和 `unsupported`
等状态，再通过 Control API 重查或交由操作员处理。设备连接始终由服务管理。
