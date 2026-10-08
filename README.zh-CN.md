<p align="center">
  <img src="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/logo.png" alt="EmbodiRun" width="440">
</p>

<h3 align="center">从模型预测，到机器人行动。</h3>
<p align="center">
  <a href="https://embodirun.readthedocs.io/zh-cn/latest/">文档</a> ·
  <a href="#快速开始与-recipe">快速开始</a> ·
  <a href="#三个场景">场景</a> ·
  <a href="#性能">性能</a> ·
  <a href="docs/zh/support-matrix.md">支持矩阵</a> ·
  <a href="README.md">English</a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python"></a>
  <a href="https://embodirun.readthedocs.io/"><img src="https://readthedocs.org/projects/embodirun/badge/?version=latest" alt="Documentation"></a>
</p>

**EmbodiRun 是面向具身智能的部署与执行运行时。** 它将机器人观测送入模型或 Agent，
执行返回的动作，并记录会话。用 YAML 描述设备、推理服务与计算节点，即可将控制进程放在机器人侧，
将推理服务部署到 GPU 主机。

使用 [EmbodiInfer](https://github.com/BUAA-CI-LAB/EmbodiInfer) 运行模型推理，
接入外部服务，或通过 [Agent 客户端](agents/CLIENT.md)编排任务。

## 三个场景

| VLA · 机械臂抓取 | VLN · 语言导航 | Agent · 移动与抓取 |
|---|---|---|
| [![SO-101 抓取](https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/multi_robot_serving.jpg)](docs/zh/demos/so101-grasping.md) | [![MuJoCo 中的 MicroDuck](https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/microduck_simulation_one_click_demo.jpg)](docs/zh/demos/microduck-vln.md) | [![XLeRobot 零食递送](https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/xlerobot_snack_delivery/xlerobot_snack_delivery_overview.jpg)](docs/zh/demos/xlerobot-snack-delivery.md) |
| **SO-101 + π0.5：** 根据相机图像和关节状态，抓取方块并放入碗中。 | **MicroDuck + ActiveVLN：** 在 MuJoCo 场景中按语言指令导航。 | **移动底盘 + SO-101 双臂：** 结合录制路线、RPent/Astra 场景判断和 VLA 抓取，在操作员监督下完成零食递送。 |
| [观看演示](docs/zh/demos/so101-grasping.md) · [Recipe](examples/so101_grasping.md) | [观看演示](docs/zh/demos/microduck-vln.md) · [Recipe](examples/microduck_vln/README.md) | [流程与架构](docs/zh/demos/xlerobot-snack-delivery.md) · [Recipe](examples/xlerobot_snack_delivery/README.md) |

[Recipe 索引](examples/README.md)提供多臂共享推理、其他推理后端及纯软件示例。
机器人、仿真器与模型的支持情况见[支持矩阵](docs/zh/support-matrix.md)。
Recipe 使用统一的 `embodirun example` 命令行入口；源码 checkout 仍可使用
兼容的 `examples/run.sh`。

## 为什么选择 EmbodiRun

| 特点 | 使用方式 |
|---|---|
| **跨机部署** | 在 YAML 中描述节点、环境、设备和模型服务，由 Host 准备环境并启动服务。 |
| **多机器人共享推理** | 各机器人保持独立会话与控制循环，共用模型服务；π0.5 可按需开启跨会话批处理。 |
| **灵活选择通信方式** | 通过 HTTP 或 WirelessComm 连接服务，使用版本化观测与动作协议，保持一致的会话与步进处理。 |
| **观测、执行与录制** | Agent、推理和录制共用相机与状态快照；执行前校验动作，支持执行仲裁与人工接管。 |

## 从模型预测到机器人执行

![Host 部署 Control 与推理服务，Control 连接应用、机器人及模型服务](docs/assets/runtime-overview.svg)

模型服务生成预测，Agent 选择任务步骤。EmbodiRun 通过共享观测、策略提议、执行任务和会话录制，
将二者接到机器人上，使同一应用能够适配不同设备与服务部署方式。

| 组件 | 职责 |
|---|---|
| **EmbodiRun** | 服务部署、机器人与仿真器观测、会话协调、动作校验、执行与录制。 |
| [**EmbodiInfer**](https://github.com/BUAA-CI-LAB/EmbodiInfer) | 检查点加载、模型推理、优化、批处理与多 GPU 执行。 |
| [**WirelessComm**](https://github.com/BUAA-CI-LAB/WirelessComm) | 控制服务与推理服务之间的可选通信传输。 |
| **RPent / Astra / 自定义 Agent** | 通过 [Agent 客户端](agents/CLIENT.md)进行任务规划与场景判断。 |

服务布局见[架构](docs/zh/architecture.md)，应用接入方式见 [Agent 工作流](docs/zh/agent-workflow.md)。

## 性能

![SO-101 推理与完整 chunk 延迟](docs/assets/performance/so101-engine-comparison.svg)

| 引擎 | 传输 | 推理延迟 | 完整 chunk 时间 |
|---|---|---:|---:|
| **EmbodiInfer** | WirelessComm | **162 ms** | **2,660 ms** |
| EmbodiInfer | HTTP | 170 ms | 2,666 ms |
| SGLang | HTTP | 194 ms | 2,713 ms |
| Native LeRobot | HTTP | 1,061 ms | 3,592 ms |

在 Jetson AGX Thor 上运行 SO-101 抓取任务：相同检查点、双相机、10 个去噪步骤、
50 步动作 chunk、20 Hz 回放；每组运行 15 个 chunk，表中为中位数。
EmbodiInfer 使用 BF16、Inductor、CUDA graph 和 Triton attention；
SGLang 使用上游默认设置，LeRobot 使用 eager 执行。

在这组配置下，EmbodiInfer 配合 WirelessComm 将完整 chunk 时间相对原生 LeRobot 缩短约 **26%**。
其中，动作回放耗时约 2.45 秒。

[条件与计时明细](docs/zh/demos/engine-e2e-contrast.md) ·
[汇总数据与绘图脚本](benchmarks/engine-comparison/README.md) ·
[HTTP / WirelessComm 测量](docs/zh/inference-transport.md) ·
[模型推理基准](https://github.com/BUAA-CI-LAB/EmbodiInfer#performance)

## 快速开始与 Recipe

### 运行本地示例

使用 Python 3.10+ 和 [uv](https://docs.astral.sh/uv/) 0.12.x 从源码安装：

```bash
git clone https://github.com/BUAA-CI-LAB/EmbodiRun.git
cd EmbodiRun
uv sync --frozen
uv run python examples/run_shared_device_fake.py
```

示例启动本地 Control 服务，用模拟关节和虚拟相机演示观测、执行、录制与取消，完成后关闭服务。
普通电脑即可运行，无需机器人、模型检查点或 GPU。

### 运行场景任务

选择 Recipe，按步骤完成环境准备、配置、启动、查看输出与停止：

- **VLA：** [SO-101 抓取](examples/so101_grasping.md)。
- **VLN：** [MuJoCo 中的 MicroDuck 导航](examples/microduck_vln/README.md)。
- **Agent：** [XLeRobot + RPent/Astra 零食递送](examples/xlerobot_snack_delivery/README.md)。
- **其他组合：** [共享推理、其他后端与纯软件示例](examples/README.md)。

[部署快速开始](docs/zh/quickstart.md)介绍 CLI 与 YAML 配置。
启用真机运动前，请完成标定并阅读[硬件安全指南](docs/zh/safety.md)。

## 文档与规划

| 内容 | 入口 |
|---|---|
| 安装与部署 | [安装](docs/zh/installation.md) · [快速开始](docs/zh/quickstart.md) · [配置](docs/zh/configuration.md) |
| 应用接入 | [Agent 工作流](docs/zh/agent-workflow.md) · [推理 API](docs/zh/http_api.md) · [Python API](docs/zh/api.md) |
| 完整任务步骤 | [Recipes](examples/README.md) |
| 部署模板 | [configs/](configs/) |
| 测量与结果 | [benchmarks/](benchmarks/README.md) |
| 已支持与计划支持的集成 | [支持矩阵与路线图](docs/zh/support-matrix.md) |

后续规划包括更多机器人 Recipe、自动数据采集流程、SmolVLA 与 OpenVLA 接入，以及部署和扩展性基准。

## 参与贡献

开发与检查流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。问题与建议请提交
[GitHub issue](https://github.com/BUAA-CI-LAB/EmbodiRun/issues)，漏洞请按 [SECURITY.md](SECURITY.md) 私下报告。
社区遵循[行为准则](CODE_OF_CONDUCT.md)。

## 许可证

Apache-2.0。参见 [LICENSE](LICENSE)、[NOTICE](NOTICE) 和[第三方声明](THIRD_PARTY_NOTICES.md)。
模型权重、数据集、机器人 SDK 和仿真器保留各自的许可证。
