# 支持的平台与路线图

按仿真器、机器人或模型查找当前支持的组合，再选择对应部署配置。
计划接入的设备和模型列在下方路线图中。

**✓ 已通过软件测试** — 已实现，并有自动化接口测试。<br>
**◐ 实验性支持** — 集成代码可用，需按平台配置。<br>
**○ 计划支持** — 尚未实现。

## 场景指南

| 场景 | 入门 | 相关配置 |
|---|---|---|
| VLA | [SO-101 抓取](demos/so101-grasping.md) | [多臂共享推理](demos/multi-robot-serving.md)、[引擎变体](demos/engine-e2e-contrast.md)、[Bi-SO-101](pi05-bi-so101.md) |
| VLN | [MuJoCo 中的 MicroDuck / ActiveVLN](demos/microduck-vln.md) | 下方的仿真器与机器人配置 |
| Agent | [XLeRobot 零食递送](demos/xlerobot-snack-delivery.md) | [Agent 工作流](agent-workflow.md)、[RPent 集成](rpent-integration.md) |

完整任务步骤见 [Recipe 索引](examples.md)。下方目录说明适配器支持状态，
各平台的依赖与标定要求见对应指南。

π0.5 HTTP 与 WirelessComm 服务通过 `--max-batch` 支持跨会话批处理，默认值为 1。
DM0.5 与 StreamVLN 当前逐请求执行。MicroDuck 使用其专用集成中的 ActiveVLN HTTP 服务。

## 当前支持

<div class="grid cards support-grid" markdown>

-   ### :material-monitor: 仿真器 {#simulators}

    ---

    **✓ LIBERO**<br>
    π0.5 操作任务，支持 EmbodiInfer 和 SGLang 后端。

    **◐ VLABench**<br>
    π0.5 操作任务。

    **◐ MuJoCo / MicroDuck** — ActiveVLN 导航。<br>
    **◐ Habitat · Isaac Sim**<br>
    使用 StreamVLN 进行导航。

    [浏览仿真器配置](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/configs/simulation)

-   ### :material-robot: 机器人 {#robots}

    ---

    **✓ SO-101** — 单臂；[真机演示](demos/multi-robot-serving.md)。<br>
    **✓ Bi-SO-101** — 双臂协同控制。<br>
    **✓ Franka FR3** — 适配器与 π0.5 绑定。

    **◐ ARX5** — DM0.5 绑定。<br>
    **◐ Unitree Go2** — StreamVLN 导航。<br>
    **◐ XLeRobot** — 独立硬件服务；双臂 π0.5 Recipe。

    [查看机器人适配器](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/src/embodirun/robots)

-   ### :material-brain: 模型 {#models}

    ---

    **✓ π0.5**<br>
    SO-101、Bi-SO-101、FR3 以及仿真器绑定。

    **◐ DM0.5 · StreamVLN**<br>
    ARX5 操作与导航集成。

    **◐ LightNav-0**<br>
    外部 XLeRobot 绑定；模型服务单独提供。

    [EmbodiInfer 模型目录](https://embodiinfer.readthedocs.io/zh-cn/latest/models/)

</div>

## 部署配置 {#deployment-recipes}

=== "仿真器"

    | 环境 | 策略 / 后端 | 配置入口 |
    |---|---|---|
    | LIBERO | π0.5 / EmbodiInfer | [配置](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/simulation/libero-pi05-embodiinfer.yaml) |
    | LIBERO | π0.5 / SGLang | [配置](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/simulation/libero-pi05-sglang.yaml) |
    | VLABench | π0.5 / EmbodiInfer | [配置](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/simulation/vlabench-pi05-embodiinfer.yaml) |
    | Habitat | StreamVLN / EmbodiInfer | [配置](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/simulation/habitat-streamvln-embodiinfer.yaml) |
    | Isaac Sim | StreamVLN / EmbodiInfer | [配置](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/simulation/isaac-streamvln-embodiinfer.yaml) |

    在启动闭环之前，准备仿真器资产、模型检查点和环境依赖。
    Isaac Sim 使用 NVIDIA 的 Isaac Sim EULA。
    依赖组见[安装](installation.md)。

=== "机器人"

    | 机器人 | 策略 / 接口 | 配置入口 |
    |---|---|---|
    | SO-101 | π0.5，独立的单臂运行时 | [共享服务配置](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/http-wireless-inference/http.yaml) |
    | Bi-SO-101 | π0.5，12 维双臂策略 | [双臂指南](pi05-bi-so101.md) |
    | Franka FR3 | π0.5 绑定 | [适配器](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/src/embodirun/robots/franka/fr3) |
    | ARX5 | DM0.5 绑定 | [绑定](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/src/embodirun/bindings/arx/x5/dm05) |
    | Unitree Go2 | StreamVLN 导航 | [机器人集成](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/src/embodirun/robots/unitree/go2) |
    | XLeRobot | 独立硬件服务；双臂使用 `lerobot.xlerobot.pi05` | [Recipe](xlerobot-snack-delivery.md) / [owner 集成包](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/integrations/xlerobot_owner/README.md) |
    | SO-101（有线遥操作） | 多主臂 UDP 分发，从臂侧 episode 采集 | [集成包](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/integrations/so101_wired_teleop/README.md) |

    SO-101 部署需要已标定的机械臂、相机映射，以及针对所选绑定
    训练的检查点。FR3 需要其机器人 SDK；ARX5 需要厂商电机
    驱动。源码链接是自定义集成的起点。

    在操作物理硬件之前，请遵循[安全](safety.md)和[手动控制](control.md)。
    [LightNav-0](lightnav0_xlerobot.md) 记录了独立的实验性
    XLeRobot 路点绑定。

=== "模型与传输"

    **EmbodiInfer** 提供第一方的 π0.5、DM0.5 和 StreamVLN
    推理服务。其[服务指南](https://embodiinfer.readthedocs.io/zh-cn/latest/serving/)
    涵盖检查点和模型特定的输入映射。

    **SGLang** 有单独安装的 π0.5 集成。**外部
    服务**通过带 `lifecycle: external` 的 provider 连接；
    参见[配置](configuration.md)和
    [外部服务示例](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/examples/external.yaml)。

    **HTTP** 是标准传输。**WirelessComm** 是实验性的、
    单独安装的选项；参见
    [WirelessComm 配置](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/http-wireless-inference/wireless.yaml)
    和[传输测量](inference-transport.md)。

## XLeRobot Recipe 前提 {#xlerobot-recipe-prerequisites}

先从[软件演练](xlerobot-snack-delivery.md)开始。它使用 fixture，不需要下表中的实体前提。
native setup 使用 Python 3.12 / uv 0.12.x；Docker `xlerobot-software` 服务需要
Linux 上的 Docker Engine 与 Compose。native 软件/硬件 profile 与 XLeRobot 容器目标
都使用 `examples/xlerobot_snack_delivery/uv.lock`。硬件 owner 使用 CPU PyTorch，
模型推理由独立服务承担。

| 输入 | 准备自己的小车时何时需要 | 软件演练使用什么 |
|---|---|---|
| owner 环境 / SDK | `setup --mode hardware` 安装 owner；将 `sdk_src` 指向诊断中 `environment.sdk_src` 给出的锁定 LeRobot 目录。自备 SDK checkout 需单独确认版本。 | 软件依赖，不导入 owner 或 SDK |
| 双臂 binding 与检查点 | 接入推理前，为 `lerobot.xlerobot.pi05` 准备适配 XLeRobot 的 π0.5 检查点。该 binding 控制双臂；底盘路线使用独立 Control scope。 | Fixture 提议，无需检查点 |
| 动作 feature names 与单位 | 调用模型前，提供左右各六个字段：`left_arm_` / `right_arm_` 加 `shoulder_pan.pos`、`shoulder_lift.pos`、`elbow_flex.pos`、`wrist_flex.pos`、`wrist_roll.pos`、`gripper.pos`。关节位置用 `degrees`，夹爪用 `range_0_100`。 | 具名 fixture 数值，不验证学习动作质量 |
| 相机映射 | 硬件检查前，在 deployment 中将 `observation.images.front`、`observation.images.left_wrist`、`observation.images.right_wrist` 映射到 owner 角色。推理前确认实际图像与检查点输入语义。 | 无真实相机 |
| 标定与设备路径 | owner 启动前，在 `hardware.local.json` 提供本机已验证的标定、稳定串口/相机路径、轮向/几何和限位。文件存在不代表标定有效。 | 允许模板占位值 |
| 路线 | `dry-run` 可用自带 `fixture: true` 路线。硬件准备需要本场地录制、带方向的非 fixture 片段；运动前验证每条路线。 | 零运动 fixture |
| Handover | 监督递送前，按本机标定与场地验证 `handover.forward_pose` / `gripper_opening`。 | Fixture 任务顺序 |
| 模型 endpoint | 启动前填写 `deployment.local.yaml` 的 `model.endpoint`；模型驱动运动前单独验证服务与具名输出。 | 不连接模型 |
| 真机停止 | 操作员在场，运动前和停止后取得 owner 的新鲜停止确认。停止请求被接受或轮速为零仍不足以确认。 | 模拟反馈 |

`check --mode hardware --json` 只读本地文件和声明。退出 0 / `status: "passed"`
不验证设备连接、实际标定、相机图像、模型输出、新鲜观测、停止反馈或任务成功。
`setup` / `check` 默认是 hardware，软件演练每次显式使用 `--mode software`。
Linux ARM64 依赖与可选集成需在目标机器检查；仅有容器镜像不构成硬件/GPU 支持证明。

## 路线图 {#roadmap}

<div class="grid cards support-grid" markdown>

-   :material-robot: __机器人__

    ---

    ○ **AgileX PiperX**

    - [ ] 机器人连接与观测适配器
    - [ ] 策略到机器人的动作绑定
    - [ ] 标定与部署示例

-   :material-brain: __模型__

    ---

    ○ **SmolVLA · OpenVLA**

    - [ ] EmbodiInfer 推理适配器
    - [ ] 服务与观测映射
    - [ ] EmbodiRun 部署绑定

    这里的 OpenVLA 指基础模型；OpenVLA-OFT 已经有
    EmbodiInfer 推理适配器。

-   :material-monitor: __仿真器__

    ---

    ○ **更多仿真器**

    - [ ] 选择下一批环境
    - [ ] 观测与动作适配器
    - [ ] 示例部署与闭环检查

</div>

### 计划补充的 Recipe 与基准

| 方向 | 内容 | 状态 |
|---|---|---|
| VLA | Bi-SO-101、Franka FR3、ARX X5 与操作仿真器的完整 Recipe | 🟨 待补充 |
| VLA | SO-101 专用单臂视频与多次任务成功率测量 | 🟨 待补充 |
| VLN | Go2、Habitat、Isaac Sim 与 LightNav-0/XLeRobot 的完整 Recipe | 🟨 待补充 |
| Agent | 自动采集流程与视频，包含任务复位和数据集导出 | 🟨 待补充 |
| Agent | 更多移动抓取 Agent Recipe | 🟨 待补充 |
| 性能 | 部署框架对比及客户端／节点扩展曲线 | 🟨 待补充 |

测量工作负载与数据要求见[基准规划](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/benchmarks/README.md#planned-benchmarks)。

## Agent 与扩展

- **MicroDuck VLN：** 一个实验性的 [MuJoCo 导航示例](microduck-vln.md)，
  带独立集成包、HTTP 推理和视频录制。
- **XLeRobot 零食配送：** 一个实验性的[任务示例](xlerobot-snack-delivery.md)，
  结合录制的底盘路线、π0.5 机械臂动作提议和操作员确认的交接。
- **有线 SO-101 遥操作：** 一个实验性的、单独安装的
  [多主臂集成](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/integrations/so101_wired_teleop/README.md)，
  带 UDP 分发和从臂侧录制。
- **自定义规划器：** [Python 客户端](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/agents/CLIENT.md)
  提供观测读取、动作提议、执行、状态查询和取消接口。
- **RPent：** 一个实验性的 [Agent 集成](rpent-integration.md)。
- **Astra + π0.5：** 一个[协作评审循环](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/agents/astra_pi05/README.md)，
  支持替换评审模块，示例默认使用模拟实现。
- **相机与传输实验：** [相机采集](camera-only-experiments.md)
  和[传输实验](transport-experiments.md)涵盖可选的
  GStreamer、共享内存、Zenoh 和 NIXL 路径。

演示通过[统一启动脚本](examples.md)准备环境、启动服务和执行任务。
服务使用的节点与 GPU 在部署 YAML 中指定。
