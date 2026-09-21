# XLeRobot 零食递送

XLeRobot 沿录制路线前往取物台，通过 π0.5/VLA 生成抓取动作，再沿另一条路线返回，
用预先标定的机械臂姿态递出物品。任务还可接入 RPent/Astra，检查相机画面并选择下一阶段。

[观看零食递送演示](demos/xlerobot-snack-delivery.md)，了解
接近、抓取与返回的实际过程。

该任务使用四个组件：

- **EmbodiRun** 负责部署、共享观测、动作校验、
  新鲜度检查、有界执行、停止请求与反馈。
- **XLeRobot owner** 管理电机与相机连接，并通过 HTTP 上报
  观测、控制状态与停止确认。
- **RPent/Astra** 可选地检查观测并选择任务阶段。
  你可以通过同一适配器接口接入其他规划器。
- **VLA 服务** 根据相机图像与状态生成抓取动作。

## 配置文件与使用指南

配置、启动器和任务代码位于
[`examples/xlerobot_snack_delivery`](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/examples/xlerobot_snack_delivery)：

| 需求 | 文档 |
|---|---|
| 场景、组件与 agent/模型划分 | [`README.md`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/xlerobot_snack_delivery/README.md) |
| 硬件清单与预算参考 | [`hardware.md`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/xlerobot_snack_delivery/hardware.md) |
| 配置、标定、路线与监督执行 | [`guide.md`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/xlerobot_snack_delivery/guide.md) |
| 接口与执行反馈 | [`architecture.md`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/xlerobot_snack_delivery/architecture.md) |
| 部署字段 | [`deployment.example.yaml`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/xlerobot_snack_delivery/deployment.example.yaml)、[`config.example.json`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/xlerobot_snack_delivery/config.example.json) |
| 统一配置与启动 | [`example.yaml`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/xlerobot_snack_delivery/example.yaml)、[`examples/run.sh`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/run.sh) |

默认路线文件是测试数据，仅供 `dry-run` 使用。
在真实场景中，先录制带方向的路线片段，也可让 RPent 根据路线图选择本地片段。
路线图负责选路，实际运动则按标定后的路线文件执行。

## 无硬件演练

执行 `uv sync --frozen` 后，通过共享入口演练该任务：

```bash
bash examples/run.sh examples/xlerobot_snack_delivery/example.yaml validate
bash examples/run.sh examples/xlerobot_snack_delivery/example.yaml dry-run
```

示例 README 说明了本地 YAML 配置、服务启动、操作员监督下的执行和输出文件。
常用命令见[复现演示](examples.md)。

对于硬件，请遵循配置指南中的 owner 与安全说明，
保持操作员在场，并在允许运动前校验机器人特定的标定与
急停。为你的工作空间录制路线，并标定机械臂与
相机。
