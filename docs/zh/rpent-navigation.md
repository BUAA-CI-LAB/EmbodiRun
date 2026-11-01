# RPent + ActiveVLN 导航

RPent 根据图像和执行反馈维护任务进度、选择子目标、判断是否需要重规划或结束。
ActiveVLN 负责执行局部导航技能；EmbodiRun 负责观测、HTTP 会话和有界动作执行。

| 工具 | 行为 |
| --- | --- |
| `observe` | 读取新图像和已消耗的执行预算 |
| `navigate` | 执行 1–8 个完整 ActiveVLN chunk，返回图像和执行反馈 |
| `update_plan` | 保存本集已完成目标、剩余目标和观测记录 |
| `finish` | 结束长任务；规划器的成功判断不直接计为评测成功 |

相同子目标继续使用原模型会话；切换目标、明确重启或局部 STOP 后再次执行时，
重置 ActiveVLN 的历史。RPent 保留本集任务进度。局部 STOP 不结束整个任务。
Go2 使用已有相机、控制客户端和同步 chunk 执行器，在等待 RPent 时已经释放本次运动租约。

安装和 Go2 命令见[导航入口](https://github.com/yufoo1/EmbodiRun/tree/feat/activevln-go2-adaptation/agents/rpent/navigation)。
未加 `--execute` 时只观测和规划；运动要求已有 owner 处于 armed 状态并支持按租约释放。
这个新增规划层尚未进行 Go2 真机验证。

仿真对比使用相同 ActiveVLN 权重、精度、初始状态和 500 步预算。
RPent 只获得指令、RGB、动作记录、碰撞数量和已消耗步数；目标坐标、参考路径和评分距离只用于评测。
每集记忆独立，使用最终 STOP 且测地距离小于 3 米计算 SR，并报告 SPL 和额外规划开销。
R2R 的未见场景导航结果不等同于通用真机长任务能力。

数据和复现方式见[仿真对比](https://github.com/yufoo1/EmbodiRun/tree/feat/activevln-go2-adaptation/benchmarks/rpent-navigation)。
