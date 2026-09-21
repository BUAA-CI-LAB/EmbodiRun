<p align="center">
  <img src="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/logo.png" alt="EmbodiRun" width="440">
</p>

<h3 align="center">从模型预测，到机器人行动。</h3>
<p align="center">
  <a href="https://embodirun.readthedocs.io/zh-cn/latest/">文档</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="#演示">演示</a> ·
  <a href="#性能">性能</a> ·
  <a href="docs/zh/support-matrix.md">支持矩阵</a> ·
  <a href="README.md">English</a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python"></a>
  <a href="https://embodirun.readthedocs.io/"><img src="https://readthedocs.org/projects/embodirun/badge/?version=latest" alt="Documentation"></a>
</p>

**EmbodiRun 是面向具身智能的部署与执行运行时。** 用 YAML 描述设备、推理服务和计算节点，
由统一运行时管理观测—推理—动作循环。控制进程可以跑在机器人旁边，推理放在 GPU 主机上，也可以在同一台机器上运行。

你可以使用 [EmbodiInfer](https://github.com/BUAA-CI-LAB/EmbodiInfer) 推理引擎，接入外部模型服务，
或让拥有独立规划循环的 Agent 调用机器人。

## 演示

| 三台机器人，共享一个推理服务 | SO-101 上的推理引擎对比 |
|---|---|
| [![三台 SO-101 的录制](https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/multi_robot_serving.jpg)](https://embodirun.readthedocs.io/en/latest/demos/multi-robot-serving/) | [![SO-101 引擎对比](https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/engine_e2e_contrast.jpg)](https://embodirun.readthedocs.io/en/latest/demos/engine-e2e-contrast/) |
| 三台 SO-101 共享一个 π0.5 推理服务，每台设备运行独立的 rollout 进程。 | Jetson AGX Thor 与 SO-101 上的 EmbodiInfer HTTP/WirelessComm、SGLang 和原生 LeRobot 对比。 |

| XLeRobot 帮你拿零食 | MicroDuck 语言导航 |
|---|---|
| [<img src="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/xlerobot_snack_delivery/xlerobot_snack_delivery_overview.jpg" alt="XLeRobot 在桌边抓取薯片" width="356">](https://embodirun.readthedocs.io/zh-cn/latest/xlerobot-snack-delivery/) | [<img src="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/microduck_simulation_one_click_demo.jpg" alt="MicroDuck 仿真双视角与执行日志" width="200">](https://embodirun.readthedocs.io/zh-cn/latest/microduck-vln/) |
| Agent 通过 EmbodiRun 串起底盘移动、VLA 抓取与携物返程。 | 启动 MuJoCo 中的 ActiveVLN 导航任务，同时查看第一人称画面、第三人称运动与逐步执行日志。 |

点击预览图观看视频，了解运行配置。

[复现演示](docs/zh/examples.md)：使用统一 YAML 配置与 `examples/run.sh` 启动入口。

## 为什么选择 EmbodiRun？

控制贴近机器人，算力按需共享。

<table>
<tr>
<td width="50%" valign="top">
<h3>🌐 一份配置，跨机部署</h3>
<p>机器人控制运行在边缘端，推理部署到 GPU 主机。一份 YAML 描述节点、环境、设备与绑定，Host CLI 负责环境准备和服务生命周期。</p>
<a href="docs/zh/architecture.md">部署架构 →</a>
</td>
<td width="50%" valign="top">
<h3>🦾 多台机器人，共享推理</h3>
<p>多个独立设备循环连接同一个模型端点。每台机器人保留自己的会话与执行流程，共享同一推理服务。</p>
<a href="docs/zh/demos/multi-robot-serving.md">查看三台 SO-101 如何共用推理服务 →</a>
</td>
</tr>
<tr>
<td valign="top">
<h3>🔌 两种传输，一套契约</h3>
<p>在 HTTP 与 WirelessComm 之间切换，无需改动面向模型的观测与动作协议，两种传输的会话与步进语义保持一致。</p>
<a href="docs/zh/inference-transport.md">传输设计与性能测量 →</a>
</td>
<td valign="top">
<h3>📷 一次采集，多方复用</h3>
<p>相机与状态快照只采集一次，供推理、Agent 观测和录制共用。设备连接由运行时统一持有，各使用方无需重复打开硬件。</p>
<a href="docs/zh/architecture.md">设备与观测管理 →</a>
</td>
</tr>
<tr>
<td valign="top">
<h3>🧩 自带规划器，复用现成机器人 API</h3>
<p>用零依赖的 Python 客户端获取观测、请求策略建议、执行动作、查询和取消任务。规划循环由你掌控，底层运行时可直接复用。</p>
<a href="agents/CLIENT.md">Agent 客户端 →</a>
</td>
<td valign="top">
<h3>🎛️ 策略输出，可控执行</h3>
<p>动作校验、执行仲裁和人工接管位于策略输出与硬件执行之间。机器人适配器和策略绑定处理运动细节，让应用专注于任务。</p>
<a href="docs/zh/safety.md">执行控制与硬件配置 →</a>
</td>
</tr>
</table>

## 工作原理

![Host 部署 Control 与推理服务，Control 连接应用、机器人及模型服务](docs/assets/runtime-overview.svg)

**Host** 准备环境并启动部署；**Control** 管理机器人连接、观测和动作执行；**Simulation** 提供仿真环境服务；
**Inference** 把观测变成模型预测。这些服务可以分散在不同机器上。完整设计见[架构文档](docs/zh/architecture.md)。

## 性能

### SO-101 真机上的 π0.5

在 Jetson AGX Thor 的录制对比中，推理延迟中位数从 **1,061 ms 降至 162 ms**，
完整控制循环 chunk 从 **3,592 ms 缩短至 2,660 ms**。推理加速缩短了循环，而每个 chunk 的动作播放仍约为 2.45 秒。

| 引擎 | 传输 | 推理延迟 | 完整 chunk 耗时 |
|---|---|---:|---:|
| **EmbodiInfer** | WirelessComm | **162 ms** | **2,660 ms** |
| EmbodiInfer | HTTP | 170 ms | 2,666 ms |
| SGLang | HTTP | 194 ms | 2,713 ms |
| 原生 LeRobot | HTTP | 1,061 ms | 3,592 ms |

每次运行统计 15 个 chunk 的中位数，使用相同 SO-101 权重、10 步去噪、两个相机，以及按 20 Hz 播放的 50 步动作块。
EmbodiInfer 使用优化路径，SGLang 使用上游默认配置，LeRobot 使用 eager 执行。
[演示报告](docs/zh/demos/engine-e2e-contrast.md)列出硬件、引擎配置与分段计时。

传输性能见 [HTTP/WirelessComm 实验](docs/zh/inference-transport.md)，模型离线性能见
[EmbodiInfer](https://github.com/BUAA-CI-LAB/EmbodiInfer#performance)。

## 快速开始

### 在本机体验运行时

使用 Python 3.10+ 和 [uv](https://docs.astral.sh/uv/) 0.12.x 从源码安装：

```bash
git clone https://github.com/BUAA-CI-LAB/EmbodiRun.git
cd EmbodiRun
uv sync --frozen
uv run python examples/run_shared_device_fake.py
```

示例启动本地 Control 服务，以模拟关节和虚拟相机演示观测、执行、录制与取消，然后关闭服务。
不需要机器人、模型权重或 GPU。

### 接入机器人或仿真器

继续阅读[快速开始](docs/zh/quickstart.md)了解部署 CLI。使用硬件前，先从[支持矩阵](docs/zh/support-matrix.md)选择组合，
配置设备与标定，并阅读[安全说明](docs/zh/safety.md)。

## 支持概览

实验性 MuJoCo 导航工作流见 [MicroDuck VLN 示例](docs/zh/microduck-vln.md)。

✓ **软件测试覆盖** · ◐ **实验性** · ○ **计划支持**

<table>
<tr>
<th align="left">🧪 仿真器</th>
<th align="left">🦾 机器人</th>
<th align="left">🧠 模型</th>
</tr>
<tr>
<td valign="top">
<p>✓ <b>LIBERO</b></p>
<p>◐ VLABench<br>◐ Habitat<br>◐ Isaac Sim</p>
<a href="docs/zh/support-matrix.md#simulators">仿真器接入 →</a>
</td>
<td valign="top">
<p>✓ <b>SO-101</b> · 含真机演示<br>✓ <b>Bi-SO-101</b><br>✓ <b>Franka FR3</b></p>
<p>◐ ARX5<br>◐ Unitree Go2<br>◐ XLeRobot</p>
<a href="docs/zh/support-matrix.md#robots">机器人接入 →</a>
</td>
<td valign="top">
<p>✓ <b>π0.5</b></p>
<p>◐ DM0.5 · ARX5 绑定<br>◐ StreamVLN · 导航<br>◐ LightNav-0 · 外部绑定</p>
<a href="docs/zh/support-matrix.md#models">模型接入 →</a>
</td>
</tr>
</table>

以上是 EmbodiRun 各集成的状态；模型与设备的具体搭配见[部署组合](docs/zh/support-matrix.md#deployment-recipes)。
推理引擎支持的完整模型列表见 [EmbodiInfer](https://github.com/BUAA-CI-LAB/EmbodiInfer#supported-models)。

**接入自己的应用：** 使用 [Python 客户端](agents/CLIENT.md)、实验性 [RPent 适配器](docs/zh/rpent-integration.md)，
或连接[外部推理服务](docs/zh/http_api.md)。

### 计划支持

- [ ] 🦾 **松灵 AgileX PiperX** — 机器人适配器与策略绑定。
- [ ] 🧠 **SmolVLA** — 推理适配器与部署集成。
- [ ] 🧠 **OpenVLA** — 原始模型接入，与已有的 OpenVLA-OFT 推理适配器区分。
- [ ] 🧪 **更多仿真器** — 具体接入目标待选。

实现步骤与各集成的归属见[路线图](docs/zh/support-matrix.md#roadmap)。

## 文档

[安装](docs/zh/installation.md) · [配置](docs/zh/configuration.md) · [人工控制](docs/zh/control.md) ·
[安全](docs/zh/safety.md) · [Python API](docs/zh/api.md) · [实验](docs/zh/experiments.md)

中文站点覆盖全部正文页面；治理与法律页（贡献指南、行为准则、许可证）保留英文原文。

## 参与贡献

开发与检查流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。问题与建议请提交
[GitHub issue](https://github.com/BUAA-CI-LAB/EmbodiRun/issues)，漏洞请按 [SECURITY.md](SECURITY.md) 私下报告。
社区遵循[行为准则](CODE_OF_CONDUCT.md)。

## 许可证

Apache-2.0。参见 [LICENSE](LICENSE)、[NOTICE](NOTICE) 和[第三方声明](THIRD_PARTY_NOTICES.md)。
模型权重、数据集、机器人 SDK 和仿真器保留各自的许可证。
