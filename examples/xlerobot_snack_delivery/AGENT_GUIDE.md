# XLeRobot deployment Agent guide

这是普通仓库说明，可交给你现有的 coding Agent 读取。它帮助准备环境、配置和软件演练。
RPent 仍负责运行时的任务规划；Astra 是可选的场景评审；EmbodiRun 负责观测、验证和执行；
XLeRobot owner 负责设备连接与停止。

## 可直接交给 Agent 的起始 prompt

```text
请先读仓库 AGENTS.md、examples/xlerobot_snack_delivery/README.md 和
examples/xlerobot_snack_delivery/AGENT_GUIDE.md，帮助我完成 XLeRobot 上手。
目标 Linux 主机：<SSH 别名或本机，尚未确定则先问我>。
目标：<无设备软件演练，或准备我自己的小车；默认先做软件演练>。
安装方式：<native 或 Docker；未指定时按主机现有环境选择>。
已有 checkout / 本地配置目录：<路径；可从当前工作区推断时直接复用>。
复用 embodirun example CLI，保存每步命令、退出码和输出目录。
根据 check --mode software|hardware --json 的诊断修复可推断的问题，
只向我询问真实设备、标定、相机、路线和模型等无法从文件确认的信息。
本次只做准备与软件演练，不启动 owner/Control、连接设备、执行运动，
不自动标定、不自动确认。完成后报告验证范围、首次阻塞原因和下一步。
```

## 执行步骤

1. **确定主机和阶段。** 先确认在哪台 Linux 主机安装，以及做软件演练还是硬件准备。
   从已有 SSH 配置、checkout 和用户提供的路径推断可确定的信息，保留现有修改。
   在目标主机检查 OS/架构、Python、uv 或 Docker/Compose；Mac 可用于编辑和远程操作，
   不在 Mac 上启动这些服务。软件阶段不需要设备、GPU、SDK、标定、路线录制或模型。
2. **复用本地配置。** 先找已有 `example.local.yaml`，没有时再 `init xlerobot`。
   `init` 遇到已有目录退出 2；按提示继续使用现有文件，或用 `--output` 指定新目录。
   不删除旧目录、不覆盖标定或已编辑配置。命令顺序沿用 [README](README.md)。
3. **准备软件环境。** native 显式执行 `setup --mode software`，再执行
   `check --mode software --json` 和 `dry-run`。Docker 使用 `xlerobot-software`；
   镜像已有环境，无需再跑 `setup`。默认 native 环境是 `.venv-xlerobot-snack`；
   native 配置已指定 `python` 时复用该环境，不另复制一套虚拟环境。
   native 配置迁入 Docker 时，顶层 `python` 优先于镜像的解释器设置，宿主路径不能直接复用。
   先备份 manifest，再仅去掉 Docker 副本的 `python` override，保留其余配置，
   让镜像使用 `/opt/venv/bin/python`；切回 native 时使用原备份。
   Docker 副本留在原配置目录，例如 `example.docker.yaml`，保留 deployment/task 的相对引用；
   后续 Docker 命令将 manifest 参数替换为 `/workspace/xlerobot/example.docker.yaml`。
   也可在新的 Docker 目录 `init` 后迁移 deployment/hardware/task/routes 与路径引用；
   保留原目录和标定文件。所有容器命令保持同一 Compose 项目和同一 UID/GID。
4. **按诊断处理。** `check --json` 只读取本地配置，不连接服务或设备。
   退出 0 / `status: "passed"` 表示这一层检查通过；退出 2 / `needs_attention`
   表示还有待办。逐项读取 `issues` 的 `code`、`location`、`message`、`next_action`。
   能从路径或现有配置推断的就修复并重跑同一模式；缺少真实信息时，说明具体字段和原因再问用户。
   同时读取 `verified`、`unverified`、`next_actions`，不要由静态通过推断设备已就绪。
5. **准备自己的车。** 只有用户选择硬件准备时才运行 `setup --mode hardware` 和
   `check --mode hardware --json`。优先将 `hardware.local.json` 的 `sdk_src` 指向
   诊断中 `environment.sdk_src` 给出的锁定 LeRobot 安装目录；Docker 对应
   `/opt/venv/lib/python3.12/site-packages`。自备 SDK checkout 的版本需单独确认。
   根据 [guide.md](guide.md) 填写设备、标定、相机映射、真实路线、handover 和 model endpoint。
   managed 路径的 Control endpoint 从 deployment 派生；路线相对 task JSON 解析。
   保持 `allow_motion: false`；不代填相机确认、不把 fixture 改成真实路线、
   不设置自动确认，也不执行 `up` 或 `run --allow-hardware`。

`setup` / `check` 省略 `--mode` 时默认是 hardware，因此软件流程每次都显式写 software。
硬件准备结束后，实际标定、相机图像、模型输出、现场急停和新鲜停止反馈仍由操作员核验。

## 安装失败和重试

保留失败命令、模式、退出码和 Python/uv 或 Docker 版本。
`uv sync` / `docker compose build` 在 Recipe CLI 之前失败时，尚无 `Outputs:` 或 `run.json`；
把该命令的终端 stdout/stderr 保存到自己选择的日志文件并记录位置。
Recipe `setup` 已打印 `Outputs:` 时，再保留其中的 `run.json` / `command-0.log`。
先读最早的错误，分清网络下载、Python 版本、依赖安装与本地路径问题，
再使用 CLI 提示重试；复用选定环境和配置目录。

仅当日志明确报 uv HTTP read timeout 时，可用
`UV_HTTP_TIMEOUT=300` 临时重试 native 的同一 setup 命令。
不要为掩盖安装错误改锁文件或换一套未记录的依赖；宿主变量不会自动进入 Docker build。
记录下载缓存是否为空、失败原因和重试次数，缓存重试成功不代表首次空缓存安装已完成。

## 交付结果

报告目标主机、阶段、安装方式、实际命令及退出码、修改的配置字段、诊断状态和输出路径。
软件演练完成时说明 fixture 流程完成、未调用模型/任务 Agent/设备，
`task_success` 仍为 `unverified`、`physical_success` 为 null。
硬件准备完成时列出剩余的操作员核验项；不声称模型兼容、真机运动或递送通过。

如以后经用户授权操作服务，Docker 的 `down` 必须使用与 launcher 相同的 Compose 项目和 UID/GID。
共享 `/tmp` named volume 只让私有 launcher socket 在这些容器间可见，不是依赖缓存；
保留原有权限和清理确认。软件准备阶段仍在 `dry-run` 与报告后结束。

这份说明可由现有 Agent 使用；独立首次使用体验请填写
[FIRST_USE_FEEDBACK.md](FIRST_USE_FEEDBACK.md)，实际反馈需由试用者提供。
