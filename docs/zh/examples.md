# 复现演示

四个演示均通过 `embodirun example` 启动，并使用带格式版本号的 YAML 配置。
部署配置描述设备和服务，示例配置指定任务、执行限制和输出目录。

| 演示 | 示例目录 | 运行内容 |
|---|---|---|
| [三台机器人，一个服务](demos/multi-robot-serving.md) | [multi_robot_serving](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/examples/multi_robot_serving) | 三个并发的 SO-101 循环，共享 π0.5 批次 |
| [引擎对比](demos/engine-e2e-contrast.md) | [engine_comparison](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/examples/engine_comparison) | EmbodiInfer HTTP/WirelessComm 或 SGLang HTTP |
| [零食递送](demos/xlerobot-snack-delivery.md) | [xlerobot_snack_delivery](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/examples/xlerobot_snack_delivery) | 沿路线移动、VLA 抓取、操作员确认交接 |
| [MicroDuck 导航](demos/microduck-vln.md) | [microduck_vln](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/examples/microduck_vln) | MuJoCo 与 ActiveVLN，本地 GPU 或 Slurm |

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
