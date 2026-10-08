# Runtime 的职责与部署成本

EmbodiRun 根据配置准备节点与环境、管理服务进程，并向应用提供共享观测和有界执行接口。
是否能减少部署和维护工作，需要在同模型、同推理后端、同硬件、同工作负载下比较。
现有 [SO-101 引擎耗时](demos/engine-e2e-contrast.md)比较了推理引擎与优化配置，
不能据此得出加入 EmbodiRun 后部署成本降低的结论。

## LeRobot 已经提供什么

这里核对的是上游 **LeRobot v0.6.1**，不能把它当作只有策略代码、没有部署工具的库。

| 需求 | 上游实现 | EmbodiRun 的职责 |
|---|---|---|
| 评估训练后的策略 | [`lerobot-eval`](https://raw.githubusercontent.com/huggingface/lerobot/v0.6.1/src/lerobot/scripts/lerobot_eval.py)创建支持的评估环境并收集 episode 指标。 | 配置支持的设备/仿真器、模型服务与绑定，并协调执行；不代替策略训练和数据集评估。 |
| 在支持的机器人上部署 | [`lerobot-rollout`](https://raw.githubusercontent.com/huggingface/lerobot/v0.6.1/src/lerobot/scripts/lerobot_rollout.py)支持同步/RTC 推理，以及自主运行、录制和人在回路中的策略。 | 让配置中的设备 owner 和服务独立于上层应用的规划循环持续运行。 |
| 远程策略推理 | [`异步策略服务`](https://raw.githubusercontent.com/huggingface/lerobot/v0.6.1/src/lerobot/async_inference/policy_server.py)通过 gRPC 加载策略、交换观测与动作。 | Control 连接兼容的外部端点，或由 Host 准备并管理已配置的 HTTP/WirelessComm provider。 |
| 跨节点服务运行 | 策略与机器人命令已经存在，具体部署仍需确定环境、地址、进程和恢复流程。 | Host 根据部署 YAML 准备节点、同步源码、等待健康状态、定位日志并核对进程身份后关闭服务。 |
| 应用/Agent 调用 | 应用可以调用 LeRobot 的策略和机器人 API。 | [Control 客户端](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/agents/CLIENT.md)提供共享观测、策略提案、有界动作、任务查询、取消和录制接口。 |

后两行说明本仓库的职责，并不代表已经测出 LeRobot 需要更多手工操作。
实现依据见[环境准备](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/src/embodirun/deployment/operations/init.py)、
[服务启动](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/src/embodirun/deployment/operations/up.py)与
[provider](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/src/embodirun/model_services/providers.py)。
每次测量记录实际 checkout 的 Git 版本及本地改动。

## ROS 2、Agent 和 EmbodiInfer 的关系

ROS 2 提供由节点、topic、service、action、参数和发现机制组成的分布式机器人通信体系；
其 launch 系统已经能启动、配置多个节点并处理进程事件。这里依据 Jazzy 文档的
[`f8b12f7`](https://github.com/ros2/ros2_documentation/commit/f8b12f7b30ce1c0d7865bbadc0c210aec79114da)版本，
参见[节点说明](https://raw.githubusercontent.com/ros2/ros2_documentation/f8b12f7b30ce1c0d7865bbadc0c210aec79114da/source/Concepts/Basic/About-Nodes.rst)和
[launch 说明](https://raw.githubusercontent.com/ros2/ros2_documentation/f8b12f7b30ce1c0d7865bbadc0c210aec79114da/source/Tutorials/Intermediate/Launch/Launch-Main.rst)。

EmbodiRun 当前没有打包可用的 ROS 2 bridge。接入 ROS 2 机器人仍需实现 adapter，
把观测、时间戳、指令限位、控制权及停止/取消反馈映射到现有设备接口，并完成测试与设备验证。
现有 HTTP/WirelessComm 接口不能直接当作 ROS 2 接口。

Agent 提供规划或复核循环；Control 持有设备连接和执行状态。请求已接受不等于动作已完成，
也不等于真机已停。[Agent 执行流程](agent-workflow.md)说明了如何查询不确定结果。
EmbodiInfer 负责模型加载、推理与引擎优化；其延迟和显存收益要与 Host/Control 的部署成本分别测量。

## 何时值得增加这一层

已有接口适用于多个节点或调用方、设备 owner 与推理进程分离、应用需要查询和取消任务等场景。
若只是一个上游已支持的机器人运行一个 LeRobot 策略，直接 rollout 可能需要更少配置和进程。
EmbodiRun 增加了 YAML 配置、adapter/绑定约束、部署状态、服务进程和网络请求；
实际支持范围见[支持矩阵](support-matrix.md)。新机器人或新后端仍需实现并验证。

这些取舍需要实测，不能预设一定省事。失败安装、重试、恢复工作以及不利结果都应保留。

## 测量六类工作

[部署成本测量流程](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/benchmarks/deployment-runtime/README.md)
包含当前软件范围可执行的命令、六项成对流程、采集字段和空报告模板。

| 场景 | 两条路线保持相同的起点和终点 |
|---|---|
| 首次部署 | 从新的任务环境到首次就绪观测；分别记录下载、安装、模型加载、warmup 和 Control 初始化。 |
| 手工工作 | 各场景的操作者步骤、修改文件数、编辑次数、求助次数和重试次数；不能从命令数推算人工成本。 |
| 更换后端 | 两侧使用相同的初始后端 A、最终后端 B、checkpoint、输入、策略设置及就绪条件。 |
| 扩展设备 | 从一台到两台等价设备，保持部署位置及观测/推理工作负载一致。 |
| 更新重启 | 使用同一源码版本变更，记录停机、准备、就绪以及保留/重建状态的成本。 |
| 故障定位 | 注入同一个具体故障，分别记录找到原因和恢复至相同就绪条件的时间。 |

**目前结果待测。** 本仓库没有现成的原生 LeRobot gRPC 到 EmbodiRun 推理接口的兼容 adapter，
也没有完整匹配的对照配置。正式同模型比较前，需要选定支持的策略/设备，取得 checkpoint 与输入，
实现协议及特征映射，并核对输出和动作限位是否等价。这些是明确的剩余工作，
现有引擎对比 Recipe 不能提供缺失的 adapter。
其中原生 LeRobot/HTTP 面板是已发布的测量材料，不是本仓库打包可用的原生 LeRobot 启动 profile。

较小的现有流程比较**同一个真实 Control 服务、`simulated.joints` 与假相机**的直接启动和 Host 管理。
它不需要模型资产，可测服务准备、就绪、关闭和软件故障定位。
此范围不能得出模型加载、warmup、换后端或真机结论；采集空模板也不代表已经完成任何一次成本实测。

## 准备同原生 LeRobot 后端的对照

第一个候选组合是 **Pi0.5、SO-101 命名观测与录制的相机输入，不向机器人下发动作**。
两侧使用相同版本的 LeRobot v0.6.1 policy server、checkpoint 与前后处理：
一侧由 gRPC 回放客户端直接调用，另一侧经过可选的 HTTP 到 gRPC bridge。
这项计划回放用于核对协议和策略输出是否等价，不是原生异步机器人 rollout，
也不能算作六类部署成本已经完成实测。

上游 [`AsyncInference` 服务](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/transport/services.proto#L38)
提供 `Ready`、`SendPolicyInstructions`、`SendObservations` 与 `GetActions`。
服务的 [`Ready`](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/async_inference/policy_server.py#L108)
会重置全局队列状态，不会创建 HTTP 式独立 session。
每条路线、每次试验都应使用新服务进程，由一个客户端独占；
不能把 `Ready` 当健康探针，也不能让两侧客户端同时连接来做此对照。

| 边界 | 已有能力 | 候选对照还需要什么 |
|---|---|---|
| 观测特征 | [`PolicyObservation`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/src/embodirun/model_services/contracts.py#L48)包含状态、编码图片、指令与 metadata。 | 按 checkpoint 顺序恢复 SO-101 六个 `.pos` 键，并统一关节/夹爪单位、相机名和 RGB 数组。回放用无损 PNG；缩放、归一化、分词与后处理仍由原生服务完成一次。先核对 checkpoint 的实际特征。 |
| 动作 chunk | [SO-101 mapper](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/src/embodirun/bindings/lerobot/so101/pi05/mapper.py#L63)核对数值行和特征名，binding 最多执行 50 步。 | 将原生 tensor 转成有限数值行，保持六个特征名、horizon 与 `pi05.action_chunk.v1`。保留原生 timestep/timestamp 供回放检查；当前 mapper 会重新赋 wall-clock 时间戳，不保留原生调度语义。 |
| Session 与重试 | [HTTP 客户端](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/src/embodirun/model_services/backends/embodiinfer/http.py#L50)提供 session/reset/close，并核对 request/session/step 身份。 | bridge 独占一个 session，把请求与 chunk 对应起来，限制等待时间并保留已有幂等行为。不自动重发结果不确定的原生请求。模型状态重置需要新建归本任务管理的后端，或已验证的显式 reset；仅清队列不够。 |
| 异步执行 | 当前[模型循环](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/src/embodirun/application/model_loop.py#L125)等待 chunk，再按 `control_hz` 执行选定前缀。 | 原生 [robot-client 队列](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/async_inference/robot_client.py#L224)会丢弃已消费 timestep 并聚合重叠 chunk。要对齐还需调度、观测发送阈值、取消/停止反馈和时钟处理，不是加一个 HTTP wrapper 就完成。 |

现有 HTTP payload 能表达候选回放的观测和数值 chunk，但 provider 注册表没有原生 LeRobot adapter。
未来 bridge 的依赖应放在可选 integration 进程，不能塞入通用 Runtime。
LeRobot v0.6.1 的[依赖声明](https://github.com/huggingface/lerobot/blob/v0.6.1/pyproject.toml#L60)
使用 NumPy 2，`pi` extra 使用 Transformers 5.4–5.5，与 MicroDuck 的 NumPy 1/Transformers 4 配置不同。
两条对照路线复用相同的隔离后端环境；各 Recipe 不强行合成一个环境。
此实验不能把原生后端换成 EmbodiInfer 引擎。

核对过的 server 与 robot client 将配置、观测和动作以 pickle 放入 insecure gRPC，
这些入口没有应用认证。因此这段连接只允许受信任的本机 loopback，
或通过已认证 SSH 转发连接到可信对端；不接收不可信 pickle，也不公开 gRPC 监听。
HTTP 的认证、限额和请求检查仍保留。依据见
[server](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/async_inference/policy_server.py#L125)与
[client](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/async_inference/robot_client.py#L110)。

输出等价需要逐层比较解码后的 RGB/状态/task、原生前处理 tensor 和后处理后的命名 chunk，
并核对单位、horizon 和时间信息。[Pi0.5 采样](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/pi05/modeling_pi05.py#L631)还需要对齐随机状态或明确重复试验比较方法；
v0.6.1 server CLI 没有 seed 参数。要保留权重成功加载的记录：核对过的
[Pi0.5 loader](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/pi05/modeling_pi05.py#L810)
可能在 checkpoint 加载失败后返回初始化模型；所以 gRPC 能连上或返回数值 chunk 也不能证明 checkpoint 已加载。
[测量流程](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/benchmarks/deployment-runtime/README.md#native-backend-preparation)
列出已按源码核对的服务命令及后续实现检查。bridge、回放客户端、授权 checkpoint/输入集与目标机器执行仍待完成。
