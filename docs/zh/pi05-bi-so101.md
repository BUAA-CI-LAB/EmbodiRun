# 用 π0.5 控制双臂 SO-101

此配置将两台 SO-101 从臂组合成一个双臂机器人，由同一策略协同控制。
[多机器人演示](demos/multi-robot-serving.md)则为每台机械臂使用独立客户端和控制循环。

## 准备硬件与模型

准备两台已标定的从臂，分别设置串口和标定 ID，配好前视、腕部相机和一台 CUDA 主机。
模型检查点需针对该双臂配置训练，状态、动作维度及三路相机输入均应匹配。

## 配置与校验

```bash
cp configs/pi05/bi-so101-embodiinfer.yaml my-deployment.yaml
# 编辑部署版本、设备、标定和检查点路径。
uv run embodirun --config my-deployment.yaml validate
uv run embodirun --config my-deployment.yaml probe
```

示例运行时为 `bi-so101-pi05`。请按照[快速开始](quickstart.md)
完成准备工作、启动、有界执行和关闭。在连接或移动机械臂之前，
请阅读[安全](safety.md)。

## 动作格式与故障处理

`lerobot.bi_so101` 组合了两个现有的已标定 SO-101 适配器。每个机械臂
都有自己的串口和标定标识；状态与动作的顺序为
左侧六个值，然后是右侧六个值，其中夹爪采用原生的
`range_0_100` 约定。绑定按名称匹配动作特征，并
在首次总线写入之前拒绝格式错误或过大的命令。

使用 [configs/pi05/bi-so101-embodiinfer.yaml](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/pi05/bi-so101-embodiinfer.yaml) 作为
校验模板。请替换串口路径、相机路径、标定 ID 和
推理检查点。`validate` 和 `build_plan` 不会接触设备。
双臂和单臂适配器均由 Control 统一管理硬件资源。双臂运行时需独占两个串口，
因此应先停止占用这些机械臂的其他遥操作或控制服务。

每条总线上的命令串行执行。若第二条命令失败，运行时会尝试让双臂保持当前测得的位置，
并保留原始错误。断连时的力矩处理、单步运动限制、被动连接和显式准备流程
均沿用单臂适配器。双臂实现已通过自动化软件测试，实机验证尚未完成。
