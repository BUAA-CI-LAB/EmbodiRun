# MicroDuck 视觉语言导航仿真

MicroDuck 示例在 MuJoCo 中运行视觉语言导航。EmbodiRun 的 HTTP 客户端管理会话并调用 ActiveVLN，
MPC 和 ONNX 行走策略负责执行 R2R 导航动作，同时录制第一、第三人称视频和每轮任务指标。

[观看导航演示](demos/microduck-vln.md)，了解启动顺序、
相机视角和执行日志。

完整说明位于
[示例指南](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/microduck_vln/README.md)。
其中包括环境搭建、外部资源目录和校验和、本地 GPU 与 Slurm 运行方式，
以及自定义任务文件、输出格式和常见问题。

## 运行示例

准备好独立环境、检查点、数据集和场景资源后，运行：

```bash
git submodule update --init third_party/embodiinfer
uv sync --frozen
uv run --frozen embodirun example init microduck --assets /absolute/asset/root
CONFIG=examples/local/microduck/example.local.yaml
uv run --frozen embodirun example "$CONFIG" validate
uv run --frozen embodirun example "$CONFIG" plan
uv run --frozen embodirun example "$CONFIG" setup
uv run --frozen embodirun example "$CONFIG" check --json
uv run --frozen embodirun example "$CONFIG" check
uv run --frozen embodirun example "$CONFIG" run
```

`validate` 和 `plan` 离线运行。`check --json` 只报告本地路径前提，CUDA/EGL 与场景/模型就绪仍未验证。
`check`（不带 `--json`）在本地前提通过后，继续在 Linux GPU 目标上执行完整
资源/CUDA/EGL/ONNX/MPC/视频编码器预检。预检通过不证明学习模型推理或导航成功；
`run` 才启动仿真 episode 和本轮持有的推理子进程。

当前 `setup` 按根目录锁文件安装 Host，再按版本范围安装完整 MicroDuck 可选集成；
后者尚未完整锁定，当前 CLI 没有 MicroDuck 纯软件模式。
Native/Docker 使用同一完整锁以及隔离软件 profile 仍是依赖工作，
还需新的目标主机安装与 GPU/模型验证。Transformers 4.51.3 环境应与 5.x 模型环境分开。
实际执行记录见[独立首次上手](first-use.md)。

Recipe 打印 `Outputs:`，运行目录包含 `run.json`、命令日志与 `result/`；
其中可查看 `preflight.json`、`run.log`、`inference.log`。`run` 退出时清理其子进程。
另一终端沿用同一 checkout 和配置，通过 `uv run --frozen embodirun example "$CONFIG" down`
请求关闭本 Recipe 的前台 launcher。Slurm 提交与 `scancel` 见示例指南。

## 参考场景与声明依赖

此实验性集成使用独立仿真环境。硬件与算法设置来自此前的 A800 场景；
包版本行是当前集成的声明，不是已解析环境或本轮重新验证的安装结果，完整列表见
[包元数据](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/integrations/microduck_vln/pyproject.toml)。

| 设置 | 参考配置 |
| --- | --- |
| 硬件 / 运行时 | Linux，一块 A800 80 GB GPU，四个 CPU 核心，EGL；Python 3.12 |
| 策略 | Qwen2.5-VL-3B ActiveVLN，外部提供的合并 SFT-v3 检查点 |
| 模型包声明，尚未完整锁定 | `torch>=2.5,<2.12`、`torchvision>=0.20,<0.27`、`transformers==4.51.3`、`tokenizers==0.21.4` |
| 仿真包声明，尚未完整锁定 | `mujoco==3.8.1`、`onnxruntime>=1.20,<2`、`casadi==3.7.2` |
| 推理来源 | 由 `third_party/embodiinfer` 固定的 EmbodiInfer 修订版本 |
| 执行 | B=1，eager，bfloat16，SDPA，贪心解码；每个 episode 使用有状态会话 |
| 场景 / 限制 | `val_2`，默认出生点 `(6.5, 13.8, 0)`，最多 60 个基元动作 |
| 成功 | STOP 且最后三个动作端点严格位于 1.0 m 半径内 |

这些范围不会自动与固定推理项目自身的 uv lock/profile 约束合并。
仍需 MicroDuck 专用完整锁和新的目标主机验证，不能由范围推断 CUDA wheel 已可用、
Native/Docker 一致或完整环境已经解析成功。

## 查看评估结果

示例指南提供了评估 40 轮任务（episode）的命令。结果写入 `results.json`，
其中 `status=complete` 表示运行结束；任务成功与否按上表中的 STOP 和距离条件判断。
`spl_euclidean` 按直线距离计算，不使用沿场景可通行路径的测地距离。
