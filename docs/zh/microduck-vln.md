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

先在仓库根目录走软件流程，无需场景资源、推理子模块、Torch 或 GPU：

```bash
uv sync --frozen --python 3.12
uv run --frozen embodirun example init microduck
CONFIG="$PWD/examples/local/microduck/example.local.yaml"
uv run --frozen embodirun example "$CONFIG" validate
uv run --frozen embodirun example "$CONFIG" plan
uv run --frozen embodirun example "$CONFIG" setup --mode software
RECIPE_ENV="$PWD/.venv-microduck"
"$RECIPE_ENV/bin/embodirun" example "$CONFIG" check --mode software --json
```

`init` 保留已有目录；再次试用时使用 `--output /absolute/new/directory`，并沿用新生成的配置路径。
`validate` 和 `plan` 离线运行；`plan` 只预览仿真命令，不执行它。
软件模式的 `check`（带或不带 `--json`）读取配置与已安装包的元数据，
CUDA/EGL、外部资源、依赖导入与模型执行仍未验证。该 Recipe 没有 fixture `dry-run`。

完整仿真需要 Linux GPU 目标，并在清单中填写外部场景根目录、适配的合并 SFT-v3 checkpoint、
episode 与资源清单。`setup` 安装依赖，不下载这些资源。
在仓库根目录执行下面的代码块，新终端也重新设置变量。代码使用默认环境和配置；
自定义路径请沿用之前环境目录与配置文件的绝对路径。
自定义场景解释器通过 `EMBODIRUN_SCENE_PYTHON` 或配置中的 `python` 指定；显式 YAML `python`
优先，应保持它与所选环境一致：

```bash
RECIPE_ENV="$PWD/.venv-microduck"
CONFIG="$PWD/examples/local/microduck/example.local.yaml"
git submodule update --init third_party/embodiinfer
"$RECIPE_ENV/bin/embodirun" example "$CONFIG" setup --mode simulation
"$RECIPE_ENV/bin/embodirun" example "$CONFIG" check --mode simulation --json && \
  "$RECIPE_ENV/bin/embodirun" example "$CONFIG" check --mode simulation && \
  "$RECIPE_ENV/bin/embodirun" example "$CONFIG" run
```

`setup` 与 `check` 默认是 `simulation`；软件流程每次显式使用 `--mode software`。
仿真模式的 `check --json` 读取已安装包元数据、本地资源路径及推理源码入口，
CUDA/EGL 与场景/模型就绪仍未验证。仿真模式的 `check`（不带 `--json`）在本地前提通过后，
继续执行资源/CUDA/EGL/ONNX/MPC/视频编码器预检。预检通过不证明学习模型推理或导航成功；
`run` 才启动仿真 episode 和本轮持有的推理子进程。

Native `setup`、直接 `uv sync` 与 `microduck-software` / `microduck` Docker target
共用 `examples/microduck_vln/pyproject.toml` 和 `uv.lock`。
软件模式选择基础 profile，仿真模式增加 `--extra simulation`。
新目标主机安装、Native/Docker 一致性与 GPU/模型执行分别记录验证结果；
锁解析或软件检查通过不能证明后几项。Docker 命令见示例指南。
Transformers 4.51.3 环境应与 5.x 模型环境分开。
记录实际命令、缓存状态、协助和结果，并与独立人工首次使用分别报告。

Recipe 打印 `Outputs:`，运行目录包含 `run.json`、命令日志与 `result/`；
其中可查看 `preflight.json`、`run.log`、`inference.log`。`run` 退出时清理其子进程。
若要从另一终端请求关闭本 Recipe 的前台 launcher，先进入同一仓库根目录并重新设置路径。
下面使用默认路径；自定义环境或配置应填写之前使用的绝对路径：

```bash
RECIPE_ENV="$PWD/.venv-microduck"
CONFIG="$PWD/examples/local/microduck/example.local.yaml"
"$RECIPE_ENV/bin/embodirun" example "$CONFIG" down
```

Slurm 提交与 `scancel` 见示例指南。

## 参考场景与锁定 profile

此实验性集成使用独立仿真环境。硬件与算法设置来自此前的 A800 场景；
包版本行是 Recipe 锁解析出的 `simulation` 候选版本，该完整 profile 尚未在目标 GPU 上安装或验证。
集成的[包元数据](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/integrations/microduck_vln/pyproject.toml)
仍保留较宽的版本范围。

| 设置 | 参考配置 |
| --- | --- |
| 硬件 / 运行时 | Linux，一块 A800 80 GB GPU，四个 CPU 核心，EGL；Python 3.12 |
| 策略 | Qwen2.5-VL-3B ActiveVLN，外部提供的合并 SFT-v3 检查点 |
| 锁定模型候选 | Linux `torch==2.10.0+cu130`、`torchvision==0.25.0+cu130`、`transformers==4.51.3`、`tokenizers==0.21.4` |
| 锁定仿真候选 | `numpy==1.26.4`、`Pillow==12.3.0`、`mujoco==3.8.1`、`onnxruntime==1.30.0`、`casadi==3.7.2` |
| 推理来源 | 由 `third_party/embodiinfer` 固定的 EmbodiInfer 修订版本 |
| 执行 | B=1，eager，bfloat16，SDPA，贪心解码；每个 episode 使用有状态会话 |
| 场景 / 限制 | `val_2`，默认出生点 `(6.5, 13.8, 0)`，最多 60 个基元动作 |
| 成功 | STOP 且最后三个动作端点严格位于 1.0 m 半径内 |

Torch/torchvision 组合沿用固定推理项目的开发 profile；其约束不会自动作用于独立安装的集成。
锁解析与公开 ARM64 wheel 信息提供了依赖候选，目标主机的驱动、glibc 和 CUDA/EGL 仍需验证。

## 查看评估结果

示例指南提供了评估 40 轮任务（episode）的命令。结果写入 `results.json`，
其中 `status=complete` 表示运行结束；任务成功与否按上表中的 STOP 和距离条件判断。
`spl_euclidean` 按直线距离计算，不使用沿场景可通行路径的测地距离。
