# 安装

使用 [uv](https://docs.astral.sh/uv/) 从源码安装 EmbodiRun。

## 环境要求

- Linux（x86_64 或 ARM64，取决于所选依赖组）。
- Python 3.10 或更新版本。部分组需要更新的 Python：`robot-so101`、
  `sim-libero`、`sim-vlabench` 和 `sim-isaac` 使用 3.12；`sim-habitat` 使用
  3.11。
- [uv](https://docs.astral.sh/uv/) 0.12.x。仓库固定了
  `required-version = ">=0.12.0,<0.13"`。
- `git` 以及对 GitHub 的网络访问。

## 源码安装

```bash
git clone https://github.com/BUAA-CI-LAB/EmbodiRun.git
cd EmbodiRun
uv sync --frozen
```

`uv sync --frozen` 会安装核心包、`host` 依赖组和开发工具。
安装后即可校验配置、通过 Host 管理服务，以及运行测试。

可运行的 Recipe 使用同一源码入口：

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

`init` 在新的 Git 忽略目录中生成已联动的可编辑配置。`setup` 使用
Python 3.12 与 uv 0.12.x。自定义 `--output` 优先使用 `examples/local/`
或仓库外目录，避免生成配置与凭证进入 Docker 构建上下文；`runs/` 和生成的
Control 凭证文件已从 Git、Docker 上下文排除。checkout 的 Host 环境使用根 `uv.lock`，
包含默认开发依赖组。独立 XLeRobot 环境默认为 `.venv-xlerobot-snack`，两种模式都使用
`examples/xlerobot_snack_delivery/` 下的依赖项目和 `uv.lock`：

| 模式 | 安装内容 | 下一步检查 |
|---|---|---|
| `setup --mode software` | 不含 owner 或 PyTorch 的小运行环境 | `check --mode software --json`，再执行 fixture `dry-run` |
| `setup --mode hardware` | 完整 owner extra，含锁定 LeRobot、CPU PyTorch、相机和录制依赖 | `check --mode hardware --json`，再由操作员核验 |

XLeRobot 省略 `--mode` 时，`setup` / `check` 默认是 hardware。硬件检查将安装的 LeRobot 包父目录
报告为 `environment.sdk_src`，可填入 `hardware.local.json` 的 `sdk_src`；自备 SDK checkout
需单独核验版本。安装依赖不会填好标定、设备、路线或提供模型检查点，推理服务单独运行。
MicroDuck 默认使用独立的 `.venv-microduck` 环境。Native setup 与容器目标都使用
`examples/microduck_vln/uv.lock`、Python 3.12 和 uv 0.12.x。
`setup --mode software` 安装小型 CLI/集成环境；`check --mode software --json`
检查环境与配置，不需要资源、GPU 或运行中的场景。`setup --mode simulation` 安装同一锁的
`simulation` extra，加入仿真与推理依赖。MicroDuck 省略 mode 时默认 simulation；
安装依赖不会提供资源，也不能证明 CUDA/EGL 或模型就绪。见 [MicroDuck Recipe](microduck-vln.md)。

在 Ubuntu 24.04 aarch64 上，MicroDuck 软件验证的 native 环境（Python 3.12.3）
与 Docker 环境（Python 3.12.14）安装了相同的 14 个锁定包。两者都使用 Python 3.12，
补丁版本不同；两个入口的普通检查与 JSON 检查都通过。首次无缓存 native 在线 setup
达到外部设置的 180 秒测试上限（受控中断，退出 130），随后使用本轮新下载 wheel 辅助，
正常 frozen setup 通过。空缓存 Docker build 达到外部设置的 600 秒测试上限
（受控中断，退出 130），同一生产 Dockerfile 的缓存续建通过（退出 0）。
这些结果不能证明无缓存首次安装能快速完成，也不是独立用户首次复现验收。
一次全新 native 完整 profile 安装也在完成前达到 600 秒测试上限；包导入与 CUDA/EGL
探针未执行，本轮没有构建完整 Docker 目标，场景和模型仍待验收。

初始 `uv sync` 或 `docker compose build` 在 Recipe CLI 前失败时，尚无 Recipe 的
`Outputs:` 或 `run.json`。将终端 stdout/stderr 保存到自行选择的日志文件，记录完整命令与退出码。
Recipe setup 期间查看 `Outputs:` 目录内的 `command-0.log`。保留首条失败和退出码，
修复原因后重试同一模式。日志明确报 native uv HTTP read timeout 时，可临时执行
`UV_HTTP_TIMEOUT=300 uv run --frozen embodirun example "$CONFIG" setup --mode software`
（按原失败模式重试：XLeRobot 用 `hardware`，完整 MicroDuck 用 `simulation`）。宿主变量不会自动进入 `docker compose build`。
记录下载缓存与重试情况；软件演练耗时不能代表完整 owner 安装耗时。

XLeRobot `check` 是只读本地检查。即使报告 `passed`，真实标定、相机图像、模型输出、
新鲜设备观测和真机停止反馈仍需在目标主机验证，之后才能运动。
见 [Recipe](xlerobot-snack-delivery.md) 与[支持矩阵](support-matrix.md#xlerobot-recipe-prerequisites)。

### 依赖组

按节点上的机器人或仿真器选择依赖组，不同设备的依赖安装在独立环境中。

| 依赖组 | 安装内容 | Python |
|---|---|---|
| `host` | SSH 编排：paramiko、PyYAML、rich | 3.10+ |
| `robot-so101` | Feetech SDK、用于 SO-101 的 OpenCV | 3.12+ |
| `robot-fr3` | 用于 Franka FR3 的 `franky-control`（Linux x86_64） | 3.10+ |
| `robot-arx5` | Pillow、RealSense、OpenCV（厂商电机驱动单独提供） | 3.10+ |
| `robot-go2` | 无额外 Python 依赖（Unitree SDK2 由厂商提供） | 3.10+ |
| `robot-xlerobot-external-owner` | 核心中无额外依赖 | 3.10+ |
| `sim-libero` | LeRobot + LIBERO、Pillow | 3.12+ |
| `sim-habitat` | habitat-sim（从源码构建）、OpenCV、Pillow | 3.11 |
| `sim-isaac` | Isaac Sim（NVIDIA EULA）、OpenCV、Pillow | 3.12 |
| `sim-vlabench` | LeRobot、VLABench、Pillow | 3.12+ |

示例：

```bash
uv sync --frozen --no-dev --group host
uv sync --frozen --no-dev --group robot-so101
UV_PROJECT_ENVIRONMENT=.venv-robot-arx5 uv sync --frozen --no-dev --group robot-arx5
```

ARX5 还需要厂商提供的 `bimanual` 扩展，其编译使用的 Python 版本应与运行环境一致。
在机器人配置中，将 `sdk_path` 设为控制节点上该扩展的绝对导入目录，
并准备厂商原生库、ROS 库和 SocketCAN 接口。

### 安装脚本

也可以通过安装脚本准备环境：

- `scripts/install.sh` 为当前源码安装所选依赖组和可选组件：

  ```bash
  scripts/install.sh --group robot-so101
  scripts/install.sh --group host --extra sglang
  ```

- `requirements/install.sh` 按指定路径和 Python 版本创建独立环境，可选择组件和 PyTorch 版本。运行
  `bash requirements/install.sh --help` 可查看完整选项列表。
  `EMBODIRUN_ENV_ROOT` 和 `EMBODIRUN_PYTORCH_INDEX` 可覆盖其默认值。

## 可选组件

| 组件 | 内容 |
|---|---|
| `sglang` | SGLang diffusion 服务（Linux、Python 3.12+、glibc >= 2.34） |
| `wireless` | WirelessComm 数据平面，固定到其已发布的 tag |

`wireless` 将
[`BUAA-CI-LAB/WirelessComm`](https://github.com/BUAA-CI-LAB/WirelessComm)
固定到 `v0.1.0`。它尚未发布到任何包索引，因此该依赖项是一个 Git
URL：

```bash
uv sync --frozen --extra wireless
```

然后在部署 YAML 中选择该传输
（`models.<id>.transport: wireless`），具体见
[`http_api.md`](http_api.md)。WirelessComm 的 peer ID 及其可选 token
不会加密或认证该链路；请仅在可信网络上使用它。

## 自动部署 {#managed-deployments}

`embodirun init` 会在各节点的部署目录中准备指定版本的 EmbodiRun 和 EmbodiInfer 源码，
再运行 `uv sync --frozen` 安装依赖。

`third_party/embodiinfer` 子模块固定了用于保证可复现性的推理服务器
版本。需要这份源码时，运行：

```bash
git submodule update --init third_party/embodiinfer
```

### 容器 {#containers}

仓库 `Dockerfile` 提供 `host`、`xlerobot-software`、`xlerobot`、`microduck-software`、`microduck` 五个目标。
在目标 Linux 主机构建；构建 `microduck` 前先初始化上述固定子模块：

```bash
docker compose build host
docker compose build xlerobot-software
docker compose build microduck-software
docker compose --profile hardware build xlerobot
docker compose --profile gpu build microduck
```

构建 `xlerobot-software` 后，通过绑定挂载的 `examples/local` 首次运行 XLeRobot 软件演练，
宿主机除 Docker 外无需安装 Recipe 依赖：

```bash
mkdir -p examples/local
docker compose run --rm --user "$(id -u):$(id -g)" xlerobot-software init xlerobot --output /workspace/xlerobot
docker compose run --rm --user "$(id -u):$(id -g)" xlerobot-software /workspace/xlerobot/example.local.yaml validate
docker compose run --rm --user "$(id -u):$(id -g)" xlerobot-software /workspace/xlerobot/example.local.yaml plan
docker compose run --rm --user "$(id -u):$(id -g)" xlerobot-software /workspace/xlerobot/example.local.yaml check --mode software --json
docker compose run --rm --user "$(id -u):$(id -g)" xlerobot-software /workspace/xlerobot/example.local.yaml dry-run
```

容器内的 `/workspace/xlerobot` 对应仓库的 `examples/local/xlerobot`，
使用当前宿主用户创建，便于继续编辑。镜像已有软件环境，跳过 native `setup`。
此 `dry-run` 使用 fixture，不启动 owner、推理服务或 RPent，无需 GPU runtime、设备、SDK、
标定或检查点。XLeRobot 软件/硬件镜像与 native setup 使用同份 Recipe 项目和锁；
通用 `host` 仍使用根项目的 Host 依赖树。

MicroDuck 软件镜像采用相同的离线配置流程，不需要 GPU runtime、仿真依赖、资源或检查点：

```bash
mkdir -p examples/local
docker compose run --rm --user "$(id -u):$(id -g)" microduck-software init microduck --output /workspace/microduck
docker compose run --rm --user "$(id -u):$(id -g)" microduck-software /workspace/microduck/example.local.yaml validate
docker compose run --rm --user "$(id -u):$(id -g)" microduck-software /workspace/microduck/example.local.yaml plan
docker compose run --rm --user "$(id -u):$(id -g)" microduck-software /workspace/microduck/example.local.yaml check --mode software --json
```

镜像构建时已经安装所选环境，容器内跳过 native `setup`。
完整 `microduck` 镜像安装同一锁的 `simulation` extra 和固定版本推理源码。
准备仿真时，通过 `MICRODUCK_ASSETS` 提供外部资源根目录，并用 `--assets /assets`
初始化另一份 Recipe；见 [Recipe 的 Docker 步骤](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/microduck_vln/README.md#docker)。
资源挂载保持只读。软件检查通过不能证明 CUDA/EGL、资源兼容性、模型推理或 episode rollout 已验收。

native manifest 转 Docker 前先备份。顶层 `python` 优先于镜像的
`EMBODIRUN_SCENE_PYTHON`，宿主解释器路径在容器中不可用。
仅去掉 Docker 副本的 `python`，让镜像选择 `/opt/venv/bin/python`，保留其他配置和标定。
副本留在原配置目录，例如 `example.docker.yaml`，保留 deployment/task 的相对引用；
后续 Docker 命令将 manifest 参数替换为 `/workspace/xlerobot/example.docker.yaml`。
切回 native 时使用原备份；也可在独立 Docker 目录重新 init，再迁移 deployment、hardware、
task 与 routes 及其路径引用，保留原始文件。

硬件 profile 需要在本地 Compose 覆盖文件中显式映射设备并只读挂载标定。
锁定 SDK 已装在 `/opt/venv/lib/python3.12/site-packages`，将 `sdk_src` 指向该目录；
自备 SDK 时显式挂载并核验版本。模型推理单独运行。
GPU profile 使用已配置的 NVIDIA 容器运行时（Thor 的 NVIDIA CSV 模式
要求 `runtime: nvidia`）并需要外部资源。默认 Compose 命令只显示
帮助。五个服务通过 Compose 项目内的 `/tmp` named volume 共享私有 launcher Unix socket。
后开的 `docker compose run` 使用与运行中 launcher 相同的 Compose 项目、宿主 UID/GID
和配置，就可通过 Recipe `down` 联系它；从同一 checkout 操作，或每次使用相同的 `-p NAME`。
该 volume 用于 launcher IPC，不是依赖缓存；私有目录/socket 权限与清理确认仍然保留。
也可对原前台容器按 Ctrl-C，或执行 `docker stop --time 300 CONTAINER` 给子进程留出清理时间。
之后核验真机停止反馈，launcher 清理本身不能证明真机已停止。
ARM64 构建和运行需要在目标机器及可选集成上实测，镜像定义本身不构成支持证明。

## PyPI

`embodirun` 尚未发布到 PyPI。请使用上文的源码安装方式。

## 运行测试

```bash
uv run pytest -q
```

CPU 测试无需 GPU、检查点、sglang 或机器人硬件。
缺少运行条件的测试会跳过，并显示原因。
