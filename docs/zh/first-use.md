# 独立首次上手

本流程检查新用户或已有编程 Agent 能否实际执行文档中的软件路径。
记录首次试用时选择新的 Linux 任务目录；较小的假设备示例也支持普通电脑。
本页提供执行说明，不代表已完成独立试用。

## 交给操作者的输入

- 仓库和待验证分支/PR head。未合并 PR 应先 checkout 其 head 再安装，默认 `main` 可能没有相应命令。
- 获授权的 Linux 目标及新的任务目录；不复制已有虚拟环境或初始化状态。
- Python 3.12、uv 0.12.x、网络/包下载条件，以及缓存是新的还是复用的；安装失败也保留记录。
  新 Host `init` 还需通过 Git 访问部署仓库及配置的版本，具备有效 TLS 信任和未缓存依赖的下载条件；
  已安装的本地 checkout 不会替代这次源码准备。
- 范围：假设备 Host/Control 工作流与 XLeRobot fixture 演练，均无需 checkpoint、相机、标定或机器人。
- 可选 MicroDuck 软件范围：配置、锁定基础依赖安装与环境元数据检查无需外部资源或 GPU。
  完整场景预检和 episode 另需提供并授权外部资源、checkpoint、Linux CUDA 和 EGL。

给已有编程 Agent 的起始提示：

```text
在指定仓库分支和新的 Linux 任务目录中，按 README 与首次上手指南操作，不复用已有虚拟环境。
安装文档中的依赖；执行假设备 Host 生命周期、describe/observe 调用和 XLeRobot 软件 fixture 演练。
保留实际命令、退出码、输出及求助过程。不要配置或操作真实设备。
MicroDuck 依次执行 init、validate、plan、setup --mode software、check --mode software --json；
仿真预检与 episode 等待操作者提供并授权完整 GPU/资源前提。
分别报告软件、真实模型和真机验证；缺少输入应记录，不要用 mock 替换后称任务已完成。
```

## 执行已有路径

1. 在选定 checkout 中运行 `uv sync --frozen --python 3.12`。安装在 Recipe 启动前失败时，保留标准输出和错误日志。
   `uv sync` 创建 `.venv`，不会把其中的命令加入当前 shell 的 `PATH`。
   Recipe 提示在可用时使用调用环境中 `embodirun` 的绝对路径，连同 checkout 与配置路径一起打印。
   复制完整提示可沿用该环境；没有已安装 CLI 时，提示回退到 `uv run --frozen --project CHECKOUT`。
2. 按[快速开始：无需机器人](quickstart.md#1)依次执行
   `validate` → `init` → `sync --source .` → `up` → `describe` → `observe`。
   该流程启动真实 Control 服务，设备为内存中的模拟关节与假相机。
   保留其中的任务专用 `--root` 和 `--state-dir`，在 `init`/`up` 前确认 8100 空闲，
   Agent 命令也沿用同一 `STATE`。
   `init` 先按 `metadata.deploy-commit` 准备托管源码，之后 `sync --source .` 才覆盖本地 checkout。
   Git 或依赖访问失败时，保留命令、退出码及 stdout/stderr 再重试；初始化成功后才继续 `up`。
   本次本地 `simulated.joints` 试用给 `init` 传入 `UV_PYTHON=3.12`，或已核验的 Python 3.12 解释器绝对路径。
   CLI 的 Python 3.12 启动环境不会自动决定托管 Host 解释器。
   按快速开始检查并记录托管环境实际的 `bin/python --version`。
   快速开始最后的 `down` 延后到下方可选 Agent 调用结束。
3. 如需查询或执行一个有界模拟动作，按 [Agent 工作流](agent-workflow.md)操作。
   保持 caller/session/request ID 一致；提交结果不明确时查询原请求。试用使用 `fake-device`。
   完成后执行 `uv run --frozen embodirun --state-dir "$STATE" --config "$CONFIG" down`，
   确认该服务退出且端点关闭。不做 Agent 动作时，观测后立即停止，无需再启动一次。
4. 按[场景与 Recipe](examples.md)或 [XLeRobot 软件演练](xlerobot-snack-delivery.md)，
   初始化联动本地目录，再执行 `validate`、`plan`、`setup --mode software`、
   `check --mode software --json`、`dry-run`。默认目录已经存在时，给 `init` 指定新的输出目录。
   XLeRobot 的 `plan` 预览的是清单中的 hardware `run` 命令；软件演练使用上面显式的
   software setup/check 与 `dry-run` 命令。
5. 按 [MicroDuck](microduck-vln.md) 依次执行 `init` → `validate` → `plan` →
   `setup --mode software` → `check --mode software --json`（也支持普通软件 `check`）。
   分开记录配置与已安装元数据的结果，以及未验证的 CUDA/EGL、资源和模型执行。
   `plan` 只预览仿真命令；MicroDuck 没有 fixture `dry-run`。
   setup/check 默认是 `simulation`；其 JSON 检查读取已安装包元数据与本地资源/源码路径，
   普通仿真 `check` 执行场景预检。只有提供仿真环境、适配 checkpoint、episode、资源清单与场景后，
   才执行这些检查。预检通过不证明模型推理或导航成功。
6. 记录打印的输出目录、`run.json`、命令/服务日志及关闭结果。
   [输出说明](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/README.md#outputs-and-shutdown)
   区分 launcher 启动前后产生的文件。

## 记录试用结果

记录应附实际终端日志：

| 字段 | 执行后填写 |
|---|---|
| 起始条件 | Checkout 版本/本地改动、目标 OS/架构、调用与托管环境的 Python/uv、新环境/状态及缓存情况 |
| 使用的文档 | 实际阅读的页面、章节与顺序 |
| 命令与结果 | 完整命令、工作目录、退出码、日志/输出位置 |
| 阻力 | 入口不明确、缺失输入、编辑文件、依赖失败与所需帮助 |
| 软件结果 | Host 观测与服务退出证据、XLeRobot fixture 结果、MicroDuck 配置/元数据结果 |
| 模型/真机结果 | 没有资源/设备时填 `not_run` 和原因，不从软件输出推断 |
| 独立程度 | 谁实际执行、是否使用旧环境、得到哪些作者协助 |

只读文档是有用的反馈，但不算独立首次上手。缓存安装不算全新下载，fixture 不算学习模型试验，
软件停止成功不算真机停止证据。
若重试需要维护者提供源码镜像、预下载包或修改配置，保留原始失败并记录源码版本、缓存和人工帮助。
将结果标为有协助的试用，不能算独立首次上手验收或无缓存的完整安装。
