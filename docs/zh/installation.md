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

## 自动部署

`embodirun init` 会在各节点的部署目录中准备指定版本的 EmbodiRun 和 EmbodiInfer 源码，
再运行 `uv sync --frozen` 安装依赖。

`third_party/embodiinfer` 子模块固定了用于保证可复现性的推理服务器
版本。需要这份源码时，运行：

```bash
git submodule update --init third_party/embodiinfer
```

## PyPI

`embodirun` 尚未发布到 PyPI。请使用上文的源码安装方式。

## 运行测试

```bash
uv run pytest -q
```

CPU 测试无需 GPU、检查点、sglang 或机器人硬件。
缺少运行条件的测试会跳过，并显示原因。
