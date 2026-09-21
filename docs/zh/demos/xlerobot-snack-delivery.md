# XLeRobot 零食递送演示

“帮我拿一包薯片。”Agent 安排路线和任务阶段，VLA 策略完成抓取，
XLeRobot 带着零食返回。EmbodiRun 串联观测、推理和运动执行，完成整个递送过程。

<video controls muted playsinline preload="metadata" width="720"
       poster="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/xlerobot_snack_delivery/xlerobot_snack_delivery_overview.jpg">
  <source src="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/xlerobot_snack_delivery/xlerobot_snack_delivery.mp4" type="video/mp4">
  你的浏览器不支持 video 标签。
</video>

*67 秒剪辑，展示下指令、接近、抓取和返回的过程。*

## 从指令到递送

Agent 负责任务的协调，VLA 策略负责抓取。两者都使用
EmbodiRun 与机器人交互：

1. **到达餐桌。** Agent 选择路线并请求底盘运动。
2. **拿起零食。** 相机观测送入 VLA 服务；EmbodiRun
   通过机器人硬件服务（owner）校验动作，并按设定范围执行。
3. **把它带回来。** 任务从抓取切换到返回路线，
   同时夹爪始终握住包装袋。

视频在实景画面上叠加任务阶段和路线，展示底盘移动与机械臂抓取之间的切换。

## 配置并运行任务

[零食配送示例](../xlerobot-snack-delivery.md) 连接了录制好的
路线、可选的 RPent/Astra 观测检查、π0.5/VLA 服务，以及
由操作员确认的交接流程。EmbodiInfer 提供模型服务；EmbodiRun 管理
观测、动作校验、执行和反馈。XLeRobot owner
管理电机和相机连接。

请先阅读示例的设置指南，配置 owner 和推理
服务、录制路线，并在你的机器人上标定抓取与交接。
验证急停功能，并在操作员监督下运行。

[示例包](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/examples/xlerobot_snack_delivery)
包含示例配置、任务参数和预置路线。运行 `uv sync --frozen` 后，
即可在不连接硬件的情况下预演任务：

```bash
bash examples/run.sh examples/xlerobot_snack_delivery/example.yaml dry-run
```

对于已标定的部署，请按照该包的 README 创建
`example.local.yaml`，然后使用 `up --allow-hardware`，并在第二个终端中
运行 `run --allow-hardware`。递送事件、决策和动作提议保存在
`result/` 下。通用命令说明见[复现演示](../examples.md)。

- [示例与设置](../xlerobot-snack-delivery.md)
- [Agent 执行工作流](../agent-workflow.md)
- [硬件安全](../safety.md)
