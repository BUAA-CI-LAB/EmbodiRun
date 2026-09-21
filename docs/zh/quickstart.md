# 快速开始

先用模拟设备走通启动和观测流程，再配置真实机器人。

## 0. 安装

```bash
git clone https://github.com/BUAA-CI-LAB/EmbodiRun.git
cd EmbodiRun
uv sync --frozen
```

依赖组和可选组件见[安装指南](installation.md)。

## 1. 运行模拟设备

`examples/shared-device-fake.yaml` 使用模拟关节和虚拟相机启动 Control 服务，
无需硬件或模型。你可以通过实际的 Host JSON API 查看设备和观测。

```bash
CONFIG=examples/shared-device-fake.yaml

uv run embodirun --config "$CONFIG" validate
uv run embodirun --config "$CONFIG" init
uv run embodirun --config "$CONFIG" sync --source .
uv run embodirun --config "$CONFIG" up

uv run embodirun --config "$CONFIG" describe \
  --runtime fake-device --caller-id example-agent --session-id example-session --json
uv run embodirun --config "$CONFIG" observe \
  --runtime fake-device --caller-id example-agent --session-id example-session --json
uv run embodirun --config "$CONFIG" down
```

`sync --source .` 会将当前工作目录中的源码同步到已准备好的本地部署中。
`describe` 报告能力，`observe` 返回一条共享观测，
`media` 按观测 ID 获取帧数据。使用 `execute` 提交一个动作。
录制、取消等完整操作见
[`examples/shared-device-fake.md`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/shared-device-fake.md)。

## 2. 配置真实机器人

复制示例，填入设备、节点和模型信息，再校验配置：

```bash
cp configs/pi05/bi-so101-embodiinfer.yaml my-deployment.yaml
$EDITOR my-deployment.yaml

uv run embodirun --config my-deployment.yaml validate
```

`validate` 只检查引用、端口、绑定和必填字段，不连接节点。
各字段的含义见[部署配置](configuration.md)。

## 3. 准备环境并启动服务

```bash
uv run embodirun --config my-deployment.yaml probe   # 连通性与工具
uv run embodirun --config my-deployment.yaml init    # 环境与源码
uv run embodirun --config my-deployment.yaml up       # 启动服务
```

`init` 会在本地记录成功状态；如果配置自 `init` 以来发生
变化，`up` 会拒绝启动。启动 Control 会加载静态
配置，并检查模型健康和相机，但不会连接或移动
机械臂。

## 4. 运行任务

该命令使用从上面复制的配置中的 `bi-so101-pi05` 运行时。
它会移动双臂。在运行它之前，请先完成[双臂配置](pi05-bi-so101.md)
和[操作员安全检查](safety.md)。

```bash
uv run embodirun --config my-deployment.yaml run \
  --runtime bi-so101-pi05 \
  --prompt "Pick up the cube and put it into the bowl." \
  --chunk-steps 10 \
  --max-steps 1
```

| 参数 | 含义 |
|---|---|
| `--runtime` | 要使用的已配置运行时 ID。 |
| `--prompt` | 覆盖任务指令；仿真器也可以提供指令。 |
| `--task`、`--seed` | 仿真器任务与 episode 种子。 |
| `--chunk-steps` | 每个动作块中实际执行的步数，不能超过绑定上限。 |
| `--max-steps` | 推理与动作执行的最大轮数（默认 1）。 |
| `--control-hz` | 动作回放频率（默认 5）。 |
| `--request-timeout` | 单次推理请求的超时时间（默认 60 s）。 |

π0.5 每次返回一个动作块（chunk），不返回任务完成信号。运行在达到设定的动作块数量后结束，
也可能因错误或取消提前结束。若请求的动作块长度超过绑定上限，运行时会在连接机器人前报错。

## 5. 停止服务

```bash
uv run embodirun --config my-deployment.yaml down
```

`down` 会停止经过身份校验的进程，并且在
配置发生变化后仍然可用。`--target control` 或 `--target model` 可以将其限制为一种
服务类型。

## 常见问题

- **`uv` 版本错误** —— 安装 uv 0.12.x；仓库已固定该版本。
- **配置校验失败** —— 检查必填字段和引用，
  并替换 `REPLACE_*` 占位符。使用 `probe` 检查节点连通性。
- **`up` 拒绝运行** —— 先运行 `init`，或者在 YAML
  发生变化后重新运行 `init`。
- **服务不健康** —— 检查 `up` 打印的每个服务的日志路径，
  查看是否有文件缺失、依赖错误或设备不可用。
- **`sync` 要求先停止 Control** —— 正在运行的进程已经导入了
  它的模块。

关于人工控制和软件急停，见 [`control.md`](control.md) 和
[`safety.md`](safety.md)。
