# 场景与 Recipe

通过统一的 `embodirun example`（兼容 `examples/run.sh`） 启动脚本，运行抓取、导航或移动抓取任务。
每个 Recipe 提供环境准备、本地配置、任务执行、输出文件与停止说明。

| 场景 | 演示 | Recipe |
|---|---|---|
| **VLA** | [SO-101 抓取](demos/so101-grasping.md) | [单臂 + π0.5 / HTTP](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/so101_grasping.md) |
| **VLN** | [MicroDuck 导航](demos/microduck-vln.md) | [MuJoCo + ActiveVLN](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/microduck_vln/README.md) |
| **Agent** | [XLeRobot 零食递送](demos/xlerobot-snack-delivery.md) | [录制路线、RPent/Astra 与 VLA 抓取](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/xlerobot_snack_delivery/README.md) |

[Recipe 索引](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/README.md)
还提供[多臂共享推理](demos/multi-robot-serving.md)、[引擎与传输变体](demos/engine-e2e-contrast.md)
及纯软件示例。其他设备与仿真器的接入情况见[支持矩阵](support-matrix.md)。

部署 YAML 定义设备与服务；示例清单选择任务、执行限制和输出目录。

## 检查配置

在仓库根目录下：

```bash
uv sync --frozen
uv run --frozen embodirun example init so101
CONFIG=examples/local/so101/example.local.yaml
uv run --frozen embodirun example "$CONFIG" validate
uv run --frozen embodirun example "$CONFIG" plan
```

`validate` 在本地检查字段和部署引用；`plan` 打印一次运行
将要执行的命令。

`init` 一次生成独立的本地目录，示例配置与引用的部署文件已联动，
且不会覆盖已有目录。其他模板可选 `xlerobot`、`microduck`、
`embodiinfer-http`、`embodiinfer-wireless` 和 `sglang-http`。
随后填写设备路径、SSH 主机、标定、检查点路径和任务设置。
`examples/local/` 会被 Git 忽略。SO-101 和 XLeRobot 的 `check`
只列出本地缺失前提，不打开设备；它不能证明机器人、相机、模型或停止反馈已就绪。
MicroDuck 的 `check` 还会在目标 Linux 主机执行 GPU/EGL 场景预检。

示例配置中的相对路径以 YAML 所在目录为基准；部署配置中的设备和模型路径则以对应节点为准。
完整字段和命令见 [examples/README.md](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/README.md)。

## 在真实机器人上运行

按照所选示例的标定与环境说明操作后：

```bash
CONFIG=examples/local/so101/example.local.yaml
uv run --frozen embodirun example "$CONFIG" setup
uv run --frozen embodirun example "$CONFIG" check
uv run --frozen embodirun example "$CONFIG" up --allow-hardware
uv run --frozen embodirun example "$CONFIG" run --allow-hardware
uv run --frozen embodirun example "$CONFIG" down
```

SO-101 服务可连续运行多个任务，每次任务之间手动复位场景，全部完成后用 `down` 停止服务。
XLeRobot 的 `up` 在前台运行；另开终端执行任务，结束后使用 `down` 或在服务终端按 Ctrl-C 停止。
启用运动前，确保操作员在场并验证急停。

无需硬件的任务预演：

```bash
uv run --frozen embodirun example examples/local/xlerobot/example.local.yaml dry-run
```

## 在仿真中运行

按照 [MicroDuck 环境搭建](microduck-vln.md) 准备外部场景与检查点，
在 Linux GPU 主机执行：

```bash
uv run --frozen embodirun example init microduck --assets /absolute/asset/root
CONFIG=examples/local/microduck/example.local.yaml
uv run --frozen embodirun example "$CONFIG" setup
uv run --frozen embodirun example "$CONFIG" check
uv run --frozen embodirun example "$CONFIG" run
```

`check` 校验资源以及 GPU/渲染环境。YAML 选择
本地或 Slurm 执行、episode 数、随机种子、动作限制和视频帧率。
运行退出时，推理子进程会停止。

## 查看日志与结果

每次调用都会在 `output_dir` 下新建目录，`run.json` 记录配置、代码版本、输入哈希和运行状态。
SO-101 为每个运行时保存命令日志；MicroDuck 和 XLeRobot 的 `result/` 目录分别保存
每轮任务的视频与指标、递送事件和动作提议。

使用 [推理传输基准测试](inference-transport.md) 进行重复的
延迟测量。
