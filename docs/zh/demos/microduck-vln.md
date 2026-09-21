# MicroDuck 语言导航演示

给 MicroDuck 一条导航指令，让它在 MuJoCo 场景中寻找目标。
ActiveVLN 根据相机画面生成导航动作，EmbodiRun 客户端连接推理服务与行走控制器。
视频同步展示相机视角和执行日志。

<video controls muted playsinline preload="metadata" width="540"
       poster="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/microduck_simulation_one_click_demo.jpg">
  <source src="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/microduck_simulation_one_click_demo.mp4" type="video/mp4">
  您的浏览器不支持 video 标签。
</video>

*29 秒的启动与导航执行演示。第一人称和
第三人称视图显示在终端日志上方。*

## 看懂演示画面

演示指令是 **“去客厅，在垃圾桶附近停下。”**
画面将每个模型决策与机器人的运动对应起来：

- **第一人称视图：** MicroDuck 相机拍摄的场景。
- **第三人称视图：** 机器人在仿真中转身行走。
- **执行日志：** 连续的导航动作、推理时间，以及到目标的
  距离。

ActiveVLN 通过推理服务生成导航动作。EmbodiRun 客户端为每轮任务维护会话，
MicroDuck 则用 MPC 和 ONNX 行走策略执行这些动作。

## 启动仿真

准备好独立环境、模型检查点和场景资源，填写本地配置后运行：

```bash
CONFIG=examples/microduck_vln/example.local.yaml
bash examples/run.sh "$CONFIG" validate
bash examples/run.sh "$CONFIG" check
bash examples/run.sh "$CONFIG" run
```

启动脚本支持本地 GPU，也支持通过 Slurm 分配资源。运行结果包括 `result/` 下的
第一、第三人称视频和每轮任务指标，以及 `run.json` 和日志。
从 `example.yaml` 复制本地配置，设置资源路径、Python 解释器、任务轮数、随机种子和计算资源。

环境要求、参考配置和完整运行说明见 [MicroDuck 仿真指南](../microduck-vln.md)。
