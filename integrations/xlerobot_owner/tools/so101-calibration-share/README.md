# SO101 标定包 · Mac 入口 / 四臂依次标定

**同一个入口运行四次，每只臂保存自己的 `arm1.json`～`arm4.json`。** 四只臂无需同时动作。主入口调用已安装的 LeRobot 完整标定流程；我们之前使用的双臂范围复核原脚本也放在 `original_range_review/`，供参考。

## 先确定 USB 接在哪里

| 接法 | `mode` | IP 填什么 | LeRobot 安装在哪 |
|---|---|---|---|
| 四只臂的 USB 直接接 Mac，或逐只插到 Mac | `local` | 不用填，`host`、`user` 会被忽略 | Mac |
| USB 接 Linux / AGX，Mac 远程启动 | `ssh` | Linux / AGX 的 IP；机械臂本身没有这里要填的 IP | Linux / AGX |

SSH 模式下，Mac 只需要 Python 3 和系统自带的 `ssh`；标定设备上的环境需要 Python 3.10+、LeRobot 及 Feetech 支持。已有可用环境直接复用。本包不包含 SDK、模型、旧标定数据或登录密钥。

## 三步运行

**1. 解压，在终端进入这个文件夹，复制配置。**

```bash
cp config.example.json config.json
```

**2. 用文本编辑器修改 `config.json`。** 示例中的地址和路径是占位示例，要替换成对方设备的配置。

| 字段 | 怎么填 |
|---|---|
| `mode` | `local` 或 `ssh` |
| `host` | SSH 目标 IP，例如 `192.168.1.100` |
| `user` | 目标 Linux 的登录用户名，例如 `robot` |
| `ssh_port` | 通常保留 `22` |
| `identity_file` | 密码登录留空；用密钥则填 Mac 上自己的私钥绝对路径。不要把私钥打包发出 |
| `python` | **执行标定那台机器**上已安装 LeRobot 的 Python 可执行文件绝对路径 |
| `sdk_src` | 通常留空。需要指定源码版 SDK 时，填执行机器上的 `lerobot/src` 绝对路径 |
| `calibration_dir` | 执行机器上保存标定结果的绝对路径；不要写 `~` |
| `arms.arm1.port` 等四项 | 每只臂实际对应的 USB 串口，替换所有 `REPLACE_ARM*` |
| `arms.arm1.type` 等四项 | 从臂用 `so101_follower`；主臂用 `so101_leader`。若是两主两从，按实际配置分别填写 |
| `arms.arm1.id` 等四项 | 默认保留 `arm1`～`arm4` 即可，四个名称不能相同 |

**Mac 直连时的关键修改示例：**

```json
"mode": "local",
"python": "/Users/yourname/lerobot-env/bin/python",
"calibration_dir": "/Users/yourname/so101-calibrations"
```

串口填写类似 `/dev/tty.usbmodemXXXX` 的实际设备名。这里的 `yourname`、`XXXX` 也要替换。其余字段保留完整 JSON 结构，不能只粘贴这三行作为整个配置。

查串口可在**插着 USB 的机器**上，用对应 LeRobot 环境的 `lerobot-find-port`，按提示拔插一只臂确认端口。Linux 也可以查看 `ls -l /dev/serial/by-id/`，优先使用这个稳定路径。给线和机械臂贴上 `arm1`～`arm4` 标签，避免认错。

同一个 USB 转接器轮流接四只臂时，四项 `port` 可以相同，但 `id` 要不同。四只臂同时接入时，应分别使用各自的总线/USB 接口；不要把具有重复电机 ID 的四只臂并到一条串行总线。

**3. 先预览，然后逐只执行。**

```bash
bash calibrate.command --arm arm1 --dry-run
bash calibrate.command --arm arm1
bash calibrate.command --arm arm2
bash calibrate.command --arm arm3
bash calibrate.command --arm arm4
```

每次等上一只标定结束再运行下一条。`--dry-run` 只显示配置和命令，不连接 SSH，也不打开串口。也可以运行 `bash calibrate.command`，在终端选择一只臂。SSH 模式按正常提示确认主机指纹、输入自己的登录密码。

## 标定时做什么、结果在哪里

先退出占用**当前臂**串口的遥操作、录制和控制程序，放稳或托稳机械臂。完整标定会修改零位偏移和范围等校准参数；跟随终端提示摆到中间位置，再逐个活动关节记录范围，不强拧机械限位。

如果提示已有同名标定，直接按 Enter 会复用旧结果；需要重新标定时输入 `c` 再回车。重做前备份原 JSON。首次装配且电机 ID / 波特率尚未设置的机械臂，需要先按官方说明完成电机配置，本入口不代替该步骤。

最终文件位于配置的 `calibration_dir` 下：

```text
arm1.json
arm2.json
arm3.json
arm4.json
```

SSH 模式文件在远端，不会自动下载。可在 Mac 上按实际地址执行：

```bash
scp -r robot@192.168.1.100:/home/robot/so101-calibrations ./calibrations-from-robot
```

后续遥操作或采集应使用对应的机械臂类型、同一个 `id` 和 `calibration_dir`。这里的 `arm1` 是标定文件标识，不会把各关节内部的电机 ID 改为 `arm1`。

## 环境与包内文件

完整标定入口按 [LeRobot v0.4.4 SO101 官方说明](https://huggingface.co/docs/lerobot/v0.4.4/so101)和该版本的 `lerobot.scripts.lerobot_calibrate` 接口整理。没有 SDK 时，先按[官方环境安装说明](https://huggingface.co/docs/lerobot/v0.4.4/installation)安装，并启用 Feetech 支持；装好后把环境的 Python 路径填入配置。使用其他版本或定制 SDK 时，具体提示和行为以该 SDK 为准。

| 文件 | 用途 |
|---|---|
| `calibrate.command` | Mac 启动入口 |
| `launch.py` | 读取四臂配置，每次仅启动选中的一只；支持 local / SSH |
| `config.example.json` | 配置模板，不含真实登录信息 |
| `original_range_review/` | 我们之前的双臂范围复核原脚本及说明 |
| `tests/test_launcher.py` | 离线命令构造测试，可用 `python3 -m pytest -q tests/test_launcher.py` 运行 |

本次检查覆盖配置、四臂选择、主从参数、SSH 引号和预览模式；没有连接收件人的四只机械臂做实机验证。若报 `No module named lerobot`，优先核对 `python` 是否填了安装 SDK 的环境；若串口忙，先退出占用当前端口的程序。
