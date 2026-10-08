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
