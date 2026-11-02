# Go2 + ActiveVLN：Orin NX 本体推理

使用这两条开发分支：

- EmbodiRun：`feat/go2-activevln-demo`。
- EmbodiInfer 子模块：`feat/activevln-go2-serving`，以仓库固定的子模块提交为准。

运行入口是 **`embodirun go2 activevln`**，支持 eager 服务的同步导航，以及带执行前缀确认的 overlap；优化的 resident-KV
benchmark 服务是另一个可选部署配置。`configs/activevln/go2-orin-nx-onboard.json` 使用一个 `robot_url` 指向 Orin
本机的 Go2 owner（8080），推理服务独立使用 18767。相机与控制共用一个进程、
HTTP 地址和 `GO2_API_TOKEN`。默认 `mode=overlap`，
可改为 `chunk` 做同步对照。完整行为说明见 [examples](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/README.md)。

## Mac：下载与精度转换

在仓库根目录使用 uv 0.12.x。`scripts/activevln-preparation` 是独立的 Python 3.10+
准备环境，有自己的依赖锁文件，不安装 Torch 或 MicroDuck，也不改变 Host/模型环境。
以下命令均从仓库根目录执行：

```bash
uv sync --project scripts/activevln-preparation --frozen
uv run --project scripts/activevln-preparation --frozen hf auth login
uv run --project scripts/activevln-preparation --frozen \
  python scripts/download_navigation_models.py \
  --repo Arvil/Qwen2.5-VL-3B_rl_r2r_4000 \
  --revision 160987313e3e869705f42400d1b8f28177044518 \
  --output ../models/activevln-r2r --workers 4 --authenticated

# 等下载生成 SHA256SUMS 后再转换；原始文件保持不变。
uv run --project scripts/activevln-preparation --frozen \
  python scripts/prepare_activevln_bf16.py \
  ../models/activevln-r2r ../models/activevln-r2r-bf16
```

该 revision 与 NX benchmark 配置一致。FP32 原始快照约 16.28 GB，BF16 副本约
8.14 GB。下载逐文件核对上游哈希；转换要求配置和索引中的每个分片均通过校验，
使用 nearest-even，并只为完整输出生成独立校验文件。额外的未校验文件不会带入副本。
这不等同于已经在设备上验证模型输出。HF 凭证只保存在标准凭证目录。

## 网线传输

本次确认的 Orin 地址是 `"$GO2_HOST"`，SSH 用户为 `unitree`。在 Mac 执行：

```bash
rsync -av --partial --progress \
  --exclude '.venv' --exclude '__pycache__' --exclude '.pytest_cache' \
  ./ unitree@"$GO2_HOST":/home/unitree/EmbodiRun/
rsync -av --partial --progress ../models/activevln-r2r-bf16/ \
  unitree@"$GO2_HOST":/home/unitree/models/activevln-r2r-bf16/
```

在 Orin 的权重目录执行 `sha256sum -c SHA256SUMS`。Mac 的 `.venv` 不用于 Linux ARM64。

## Orin：运行前提

当前设备检查记录是 L4T R35.3.1、CUDA 11.4、Python 3.8、NX 16GB；分支的
[NX 参考配置](https://github.com/yufoo1/EmbodiInfer/blob/fix/activevln-resident-kv/benchmarks/activevln-benchmark/orin-nx/config.json)
记录的是 **L4T R36.4.7、CUDA 12.6、Python 3.10.20、Torch 2.10.0、Triton 3.4.0**。
部署以设备现有的 JetPack 5.1.1 / CUDA 11.4 为目标。控制环境已经使用 uv 新建在
`/home/unitree/embodirun-envs/control-jp5`，不复用 Conda 或容器。计算环境位于
`/home/unitree/embodirun-envs/compute-jp5`，使用已通过 SHA256 校验的 ARM64 / cp310
预编译 PyTorch 2.2.0、torchvision 0.17.2+c1d70fe wheel。CUDA 11.4 / cuDNN 8.6、
BF16 GPU 运算和一次真实相机推理已验证；源码编译已停止。该 eager 路径不使用
参考开分配置。普通 CUDA 13.0 锁文件也不是 Jetson 安装方案；本次没有升级系统或刷机。

控制环境使用 Python 3.10.20、独立编译的 CycloneDDS 0.10.5，以及在设备编译的
`pyrealsense2==2.55.1+go2.jp5`。已检查的 PyPI ARM64 RealSense wheel 虽标记为
manylinux2014，实际依赖较新 glibc，不能在本机 Ubuntu 20.04 直接导入。
该环境的 `python` 包装器会加载私有原生库路径，可同时用于 owner 和 supervisor：

```bash
# 在 Mac 的仓库环境执行；Wi-Fi 地址由 DHCP 分配，需按当前地址设置。
GO2_HOST=192.168.8.6
CONTROL_PY=/home/unitree/embodirun-envs/control-jp5/python
embodirun go2 edge --host "$GO2_HOST" --username unitree --ask-password status \
  --supervisor-python "$CONTROL_PY"
```

已验证 CLI 的 `install/start/status/stop`、真实 SDK 状态读取和 D435I 的 RGB/RGB-D
观测读取。该次静态检查的 owner 未启用 `--operator-ready`，模型输出未发送给机器人执行；后续运动见真机演示。

用厂商 Python 读取相机序列号、深度单位和可用 profile，生成一份 owner 配置。
此步骤不启动视频流或运动；已有文件不会覆盖。多相机时加 `--serial`：

```bash
cd /home/unitree/EmbodiRun
/usr/bin/python3 scripts/inspect_go2_onboard.py --camera --write-owner-config go2-owner.local.json
```

检查生成配置中的 `control.interface`（当前为 `eth0`）和 `cyclonedds_lib_dir`。
`camera` 使用实测序列号、深度单位和 640x480@15 profile，不能直接猜深度单位。
`server` 管理唯一地址和凭证。若只检查控制，可将 `camera` 设为 `null`。

统一硬件进程支持厂商 Python 3.8，与 Host、模型环境独立。在仓库中直接运行：

```bash
export PYTHONPATH="$PWD/integrations/go2_owner/src:$HOME/unitree_sdk2_python"
/usr/bin/python3 -m embodirun_go2_owner --config go2-owner.local.json --validate
/usr/bin/python3 -m embodirun_go2_owner --config go2-owner.local.json
```

也可在已有 SDK 环境安装 `integrations/go2_owner` 后使用 `embodirun-go2-owner`。
owner 包不安装 Host、Torch 或模型依赖；相机的 numpy/Pillow/pyrealsense2 使用厂商兼容版本。
默认没有运动权限，环境变量不能自动启用权限。

`/v1/state`、`/v1/actions`、`/v1/stop` 与相机接口共用 8080。
ActiveVLN 读取 `/observation/rgb.json`，仅要求新鲜 RGB；其他策略仍可读取严格匹配的
RGB-D `/observation.json`。`/health` 返回各组件状态；`/health/camera` 检查 RGB 并附带
深度就绪信息。相机请求独立调度，关闭 owner 时先停止运动，再等待相机退出。

远程部署入口已从模型绑定移到 **`embodirun go2 edge`**，由 Mac 上传上述配置：

```bash
embodirun go2 edge --host "$GO2_HOST" --username unitree --ask-password install
embodirun go2 edge --host "$GO2_HOST" --username unitree --ask-password start \
  --config /path/to/go2-owner.local.json --python /usr/bin/python3 \
  --sdk-pythonpath /home/unitree/unitree_sdk2_python --supervisor-python /path/to/host/python3
embodirun go2 edge --host "$GO2_HOST" --username unitree --ask-password status \
  --supervisor-python /path/to/host/python3
embodirun go2 edge --host "$GO2_HOST" --username unitree --ask-password stop \
  --supervisor-python /path/to/host/python3
```

`--supervisor-python` 是远端 Python 3.10+，`--python` 是拥有 SDK 的远端 Python 3.8+；
两者可以不同。远端只有一个 `go2` 服务。`install` 仅上传 owner 和独立 supervisor，
不会上传整套 Host。旧 `go2 activevln edge` 和 `embodirun-go2-streamvln` 命令保留转发，
旧的 `--service camera/control` 和两个 agent 模块入口已移除，请先停止旧进程再迁移。
绑定非回环地址时必须提供至少 32 字符的 `GO2_API_TOKEN`；相机与控制使用同一凭证。
本体推理推荐保留回环绑定。PID 状态不代表硬件已就绪，随后还需运行 `activevln check`。

模型环境确定后，用该环境的 Python 执行 `scripts/inspect_go2_onboard.py --torch`
检查版本和 CUDA。

## 本机已验证的 eager 推理服务

在 Orin 的仓库目录中，使用独立计算环境启动：

```bash
/home/unitree/embodirun-envs/compute-jp5/run embodiinfer-http-serve \
  --policy activevln --checkpoint /home/unitree/models/activevln-r2r-bf16 \
  --adapter-config configs/activevln/infer-jp5-eager.json \
  --device cuda:0 --dtype auto --no-cuda-graph --max-batch 1 \
  --host 127.0.0.1 --port 18767 --max-sessions 1 --max-images 1
```

另一终端使用控制环境执行 `embodirun go2 activevln check/run`，配置选择
`configs/activevln/go2-orin-nx-jp5-eager.json`。此配置为同步 chunk 模式，推理请求
超时 300 秒；执行时的计划时效限制仍为 5 秒。模型服务绑定本机，无 token 时
不要设置 `ACTIVEVLN_BENCHMARK_TOKEN`。

2026-10-29 已通过 CLI 完成一次 D435I RGB → ActiveVLN → 动作建议。首次请求
往返约 7.72 秒，建议连续右转 15° 三次，`executed_actions=0`。这不是持续性能或
运动闭环验证；首帧时延超过默认执行时效，不能据此直接启用运动。

## 优化的 resident-KV 参考服务

使用 [Infer 的 NX 启动器](https://github.com/yufoo1/EmbodiInfer/blob/fix/activevln-resident-kv/benchmarks/activevln-benchmark/orin-nx/README.md)，
从 BF16 开始。该启动器需要其固定的 R2R warmup 数据和 256 集选择 manifest，
不仅是模型权重；请另行准备这些资源，JP5 eager 配置无需它们。Orin 不需要启动 Habitat。

所需参数为：

```bash
# INFER_PY：已验证的 NX 推理 Python；WARMUP_R2R / MANIFEST：固定的预热资产。
"$INFER_PY" third_party/embodiinfer/benchmarks/activevln-benchmark/orin-nx/run.py bf16 \
  --dataset r2r --checkpoint /home/unitree/models/activevln-r2r-bf16 \
  --data-root "$WARMUP_R2R" --manifest "$MANIFEST" \
  --output "$MODEL_RUN" --port 18767
```

`MODEL_RUN` 必须是新的输出目录；等其 `service-ready.json` 生成后再连接。
启动器校验预热资产和 manifest 哈希。INT8 另需匹配的校准文件，不在本次默认路径内。

## 检查和运行

Host 使用单独的 Python 3.10+ 环境安装 EmbodiRun。服务均启动后，在 Orin 执行：

```bash
CONFIG=configs/activevln/go2-orin-nx-onboard.json
export ACTIVEVLN_BENCHMARK_TOKEN="$(cat "$MODEL_RUN/service-token")"
embodirun go2 activevln plan --config "$CONFIG"
embodirun go2 activevln check --config "$CONFIG"
embodirun go2 activevln run --config "$CONFIG" \
  --instruction "Go to the door and stop."
```

`plan` 完全离线；`check` 只读服务；上面的 `run` 只推理一次，不写运动命令。
真实运行前，操作员确认场地和停止手段，停止 owner 后，添加 `--operator-ready` 重新启动，再给 `run` 添加 `--execute`。
单次 `run` 在尝试运动后结束时仍会停止并撤销运动许可；相机继续可读。
需要连续输入指令时，使用下面的 `interactive` 模式。急停和运动故障后的重新授权仍需操作员重启 owner。

## 常驻交互和长程任务

模型服务原本就会保持权重加载；新增的 `interactive` 让终端也持续接收指令。
同一条指令内保留逐轮导航历史，每条新指令新建推理 session；结束时释放上一条的
session 和历史，不重载权重。需要同时更新 EmbodiRun CLI 和 `integrations/go2_owner`；
如果 owner 不支持任务级停止，CLI 会在运动前报错要求更新。

启动模型与 owner 服务后，在控制环境执行：

```bash
source /home/unitree/embodirun-envs/control-jp5/environment.sh
embodirun go2 activevln interactive \
  --config configs/activevln/go2-orin-nx-jp5-eager.json \
  --max-chunks 100 --max-runtime-s 600 --execute
```

使用已部署的目录时，也可以把 `--config` 换成
本地验证记录（保留在实验归档中）。

```text
activevln> Move to the open area to the right of the yellow pillar.
……持续进行多轮观测、推理、执行……
Task ended: stop (... chunks). Enter the next instruction.
activevln> Go to the door and stop.
activevln> /quit
```

每轮模型最多给出三个动作。首测文件中的 `max_chunks=1` 把整条指令限制成了一轮；
上述参数覆盖配置，允许每条指令最多 100 轮、600 秒。模型返回 STOP 会提前结束；
达到上限不代表已经到达目标。时间上限限制继续运动，正在进行的推理和清理可能延迟
回到输入提示。模型自身的上下文长度限制仍然存在。单轮 9 秒、速度、心跳和 5 秒计划
时效等限制保持有效。

- 任务执行中按 **Ctrl+C**：停止当前任务的运动，等待正在进行的推理清理，再接收下一条指令。
- 输入提示处按 Ctrl+C、EOF 或输入 `/quit`：退出交互终端，模型和 owner 服务继续运行。
- 不加 `--execute`：每条输入只做一次推理，不写运动命令。
- 普通文本命令在任务之间读取；正在执行时用 Ctrl+C 中止。

正常结束和 Ctrl+C 使用携带 action_id 的任务级停止，只释放自己的 stream lease。
确认停止且状态正常后，保留已经开启的运动许可；不能重新授权，也不能停止其他控制器的任务。
`/v1/stop`、运动中的 SIGTERM 和运动后的任务故障仍走撤销许可的停止路径。
操作失败会退出 CLI，不自动重试或重新授权。

[真机演示](demos/go2-activevln.md)已记录 chunk 与 overlap 的实际运动。

真机视频展示了有限的执行片段，不代表导航成功率验证。
overlap 使用运动中的图像和已承诺动作前缀；视频保留实际轮间停顿，深度不是避障控制器。
