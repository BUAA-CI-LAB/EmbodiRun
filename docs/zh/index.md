<div class="hero" markdown>

<h1 class="hero-title">
  <img src="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/logo.png" alt="EmbodiRun" class="hero-logo">
</h1>

**一键部署案例、Recipe 与具身部署框架。**

用一份配置连接模型、计算节点与机器人或仿真器，跑通观测、推理和动作执行。

[快速开始](quickstart.md){ .md-button .md-button--primary }
[架构](architecture.md){ .md-button }
[GitHub](https://github.com/BUAA-CI-LAB/EmbodiRun){ .md-button }

</div>

从 [EmbodiInfer](https://github.com/BUAA-CI-LAB/EmbodiInfer) 选择模型与推理配置，
再通过 EmbodiRun 的 Recipe 部署服务、接入机器人或仿真器、执行动作并录制会话。
也可接入外部模型服务，或由 Agent 编排任务循环。

## 演示与对比

| 演示 | 条件与结果 |
|---|---|
| VLA · SO-101 真机 | [三引擎 / 四配置录像与完整 chunk 耗时](demos/engine-e2e-contrast.md) |
| VLN · Habitat / ActiveVLN | [EmbodiInfer、SGLang、vLLM 原速闭环对比](demos/activevln-engines.md) |
| VLN · Go2 真机 | [chunk / RTC 录像，3× 播放](demos/go2-activevln.md)；适配代码待合入 |
| 模型推理性能 | [EmbodiInfer 的 VLA / VLN 基准](https://github.com/BUAA-CI-LAB/EmbodiInfer#performance) |
| 部署组合与后续计划 | [支持矩阵与路线图](support-matrix.md) · [全部 Recipe](examples.md) |

## 三个场景

| 场景 | 任务 | 入口 |
|---|---|---|
| **VLA · 机械臂抓取** | SO-101 + π0.5 抓取方块并放入碗中。 | [演示](demos/so101-grasping.md) |
| **VLN · 语言导航** | MicroDuck + ActiveVLN 在 MuJoCo 中按语言指令导航。 | [演示](demos/microduck-vln.md) |
| **Agent · 移动与抓取** | XLeRobot 结合底盘路线、RPent/Astra 判断和 VLA 抓取，在监督下完成零食递送。 | [流程与架构](demos/xlerobot-snack-delivery.md) |

每个 [Recipe](examples.md)提供环境准备、配置、运行、输出与停止说明。

## 从这里开始

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __部署__

    ---

    从源码安装，运行一个无需设备的示例，然后编写部署 YAML。

    [:octicons-arrow-right-24: 快速开始](quickstart.md)

    [:octicons-arrow-right-24: 配置](configuration.md)

-   :material-robot:{ .lg .middle } __操作机器人__

    ---

    控制权限、有界执行、人工接管和软件急停。

    [:octicons-arrow-right-24: 控制](control.md)

    [:octicons-arrow-right-24: 安全](safety.md)

-   :material-connection:{ .lg .middle } __集成模型或 Agent__

    ---

    通过版本化推理 API 连接策略和 Agent。

    [:octicons-arrow-right-24: 推理 API v1](http_api.md)

    [:octicons-arrow-right-24: RPent](rpent-integration.md)

-   :material-sitemap:{ .lg .middle } __了解系统设计__

    ---

    部署、推理服务和机器人控制如何协同工作。

    [:octicons-arrow-right-24: 架构](architecture.md)

    [:octicons-arrow-right-24: 支持矩阵](support-matrix.md)

</div>

## 系统如何协作

```mermaid
flowchart TB
  agent["Agent / 应用"]
  subgraph embodirun["EmbodiRun"]
    runtime["部署 · 应用 · 设备<br/>配置、协调、仲裁、记录、安全"]
    services["模型服务<br/>版本化推理契约"]
  end
  engine["EmbodiInfer，或 SGLang 等外部后端"]
  robot["机器人或仿真器"]

  agent -- "observe · propose · execute · inspect · cancel · stop" --> runtime
  runtime --> services
  services -- "HTTP 或 WirelessComm" --> engine
  runtime -- "经过校验的动作" --> robot
  robot -- "观测" --> runtime
```

## 选择设备与示例

浏览[支持的机器人、仿真器和模型](support-matrix.md)，或者
从[三个场景及其 Recipe](examples.md)中选一个开始。每个示例都包含
配置文件、启动命令以及对其输出的说明。
有关延迟测量和传输对比，请参见[实验](experiments.md)。

## 社区

- [贡献指南（英文）](https://embodirun.readthedocs.io/en/latest/contributing/) ——
  开发环境搭建、测试和拉取请求。
- [行为准则（英文）](https://embodirun.readthedocs.io/en/latest/code-of-conduct/) ——
  本项目采用的 Contributor Covenant 2.1 贡献者公约。
- [安全策略](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/SECURITY.md)
  —— 如何非公开地报告漏洞。
- [许可证（英文）](https://embodirun.readthedocs.io/en/latest/license/) ——
  Apache-2.0 与第三方声明。

仓库 README 提供
[英文](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/README.md) 和
[简体中文](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/README.zh-CN.md) 两个版本。
