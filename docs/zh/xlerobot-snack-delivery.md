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
| 让现有 coding Agent 帮助部署 | [`AGENT_GUIDE.md`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/xlerobot_snack_delivery/AGENT_GUIDE.md) |
| 课堂试用与首次使用反馈 | [`FIRST_USE_FEEDBACK.md`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/xlerobot_snack_delivery/FIRST_USE_FEEDBACK.md) |

默认路线文件是测试数据，仅供 `dry-run` 使用。
在真实场景中，先录制带方向的路线片段，也可让 RPent 根据路线图选择本地片段。
路线图负责选路，实际运动则按标定后的路线文件执行。

## 无硬件演练

选择 native 安装或 [Docker 软件路径](installation.md#managed-deployments)。
两者都需要目标 Linux 主机上的 checkout 与首次安装的网络访问。
native XLeRobot setup 使用 Python 3.12 与 uv 0.12.x：

```bash
uv sync --frozen --python 3.12
uv run --frozen embodirun example init xlerobot
CONFIG=examples/local/xlerobot/example.local.yaml
uv run --frozen embodirun example "$CONFIG" validate
uv run --frozen embodirun example "$CONFIG" plan
uv run --frozen embodirun example "$CONFIG" setup --mode software
uv run --frozen embodirun example "$CONFIG" check --mode software --json
uv run --frozen embodirun example "$CONFIG" dry-run
```

此路径无需真实 SDK、标定、相机、路线录制、检查点、model endpoint、RPent 或 GPU。
`init` 遇到已有目录时，复用其提示的配置路径或另选 `--output`；已有工作不会覆盖。
Docker 使用 `xlerobot-software`，镜像已有软件环境，跳过 native `setup`。
它与 native XLeRobot setup 使用同一个 Recipe 依赖项目和锁。

`check` 通过时退出 0、`status: "passed"`，有待处理问题时退出 2。
读取每个 issue 的 `location` 与 `next_action`，修复后重跑同一检查。
终端打印的输出目录包含 `run.json`、`command-0.log` 和
`result/status.json` / `result/events.jsonl`。完成的 dry-run 记录 fixture 任务顺序，
`task_success` 为 `unverified`、`physical_success` 为 null；不启动 owner、推理服务或任务 Agent。

## 准备自己的小车

继续使用生成的本地目录：

| 文件 | 需要填写的本机信息 |
|---|---|
| `deployment.local.yaml` | 模型 endpoint、Control 端口、相机 feature 到 owner 角色的映射 |
| `hardware.local.json` | 已安装 SDK 路径、设备、已验证标定、轮参数、限位 |
| `config.local.json` 和 `routes/` | 录制的有向路线、抓取指令、规划器、标定后的递交姿态 |

执行 `setup --mode hardware` 准备完整 owner 环境，再用
`check --mode hardware --json` 检查本地前提。省略 `--mode` 时，两者默认是 hardware。
双臂 `lerobot.xlerobot.pi05` binding、单位、相机和输入前提见
[支持矩阵](support-matrix.md#xlerobot-recipe-prerequisites)。

静态检查不代表实际就绪。按照配置指南，由在场操作员验证标定、实际相机图像、
模型适配和新鲜停止反馈，再授权运动。推理单独运行，安装 owner 不会提供检查点。
监督下的 `up` / `run` 和关闭方式见局部 README。课堂反馈材料已提供，真实同学反馈仍待收集。
