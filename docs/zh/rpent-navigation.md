# RPent + ActiveVLN 导航

RPent 根据图像和执行反馈维护任务进度、选择子目标、判断是否需要重规划或结束。
ActiveVLN 负责执行局部导航技能；EmbodiRun 负责观测、HTTP 会话和有界动作执行。

| 工具 | 行为 |
| --- | --- |
| `observe` | 读取新图像和已消耗的执行预算 |
| `navigate` | 本地视觉闭环执行到事件或定期检查点，可在同一次调用中保存计划 |
| `navigate_task` | 任务已初始化时提供的事件工具；由模型选择，原样执行当前完整任务，省去重复生成指令 |
| `finish` | 可同时保存计划并结束任务；规划器判断不直接计为评测成功 |

默认 `event` 配置在局部 STOP、连续碰撞、画面停滞/重复、预算耗尽或至多 32 个 chunk
后交回 RPent。ActiveVLN 每个 chunk 前仍读取新图像。视觉事件只提示可能卡住或循环，
不证明定位或语义进度；Go2 后端没有碰撞遥测，因此使用视觉事件。
`navigate` 和 `finish` 的 `plan={completed,remaining,observations}` 参数保存本集进度，
无需单独调用更新计划工具。仅上层反馈图使用 JPEG quality 80，保持原分辨率，
远端执行器在 SSH 回传前压缩；ActiveVLN 输入保持不变。
首帧直接随任务传入，省去开场的 `observe` 工具往返，采集耗时仍计入整集耗时。
`--profile legacy` 保留原四工具、1–8 chunk 接口用于对比。

相同子目标继续使用原模型会话；切换目标、明确重启或局部 STOP 后再次执行时，
重置 ActiveVLN 的历史。RPent 保留本集任务进度。局部 STOP 不结束整个任务。
Go2 使用已有相机、控制客户端和同步 chunk 执行器，在等待 RPent 时已经释放本次运动租约。

本地 skill 在每次模型调用前后检查取消状态和本集时限。已完成的推理仍计入调用预算，
但迟到的动作不会交给执行后端。共享模型规划器保留工具实际返回的反馈，
同时拒绝把取消或超时后返回的 finish 当作正常完成。
共享模型工具在调用线程内执行时，退出信号先记录本集取消请求，让执行在下一检查边界
退出后再等待清理，避免线程等待自己的工具结束。后台 MCP 工具仍在取消边界结束后清理。

安装和 Go2 命令见[导航入口](https://github.com/yufoo1/EmbodiRun/tree/feat/rpent-activevln-navigation/agents/rpent/navigation)。
未加 `--execute` 时只观测和规划；运动要求已有 owner 处于 armed 状态并支持按租约释放。
这个新增规划层尚未进行 Go2 真机验证。

仿真对比使用相同 ActiveVLN 权重、精度、初始状态和 500 步预算。
RPent 只获得指令、RGB、动作记录、碰撞数量和已消耗步数；目标坐标、参考路径和评分距离只用于评测。
每集记忆独立，使用最终 STOP 且测地距离小于 3 米计算 SR，并报告 SPL 和额外规划开销。
初始记忆把完整原任务登记为一个尚未完成的目标，后续语义进度由模型决定。
R2R 的未见场景导航结果不等同于通用真机长任务能力。

数据和复现方式见[仿真对比](https://github.com/yufoo1/EmbodiRun/tree/feat/rpent-activevln-navigation/benchmarks/rpent-navigation)。

实验性的 `shared-model` 评测后端让同一个驻留模型负责导航和 RPent 工具决策。
它核对模型实例身份，在有界上下文中保留原始任务与本集记忆，并按实际 RPent schema
校验模型生成的参数，无需配置第二个规划模型。Go2 CLI 现在默认使用
`--planner-backend shared-model`，在创建硬件客户端前核对导航服务的模型实例，
不回退到其他模型。使用 event 配置，不设置 `--planner-model`。未加 `--execute`
时只暴露 `observe`/`finish`，并拒绝把未执行任务声明为成功；执行模式暴露
`navigate_task`、`navigate`、`advance_goal` 和 `finish`。原云端后端需要显式指定
`--planner-backend codex --planner-model MODEL`。单模型服务的精度、延迟及 Go2
真机行为仍在开发验证中；入口仅通过模拟客户端的 CPU 检查，不代表真机验证。
规划图像需满足服务的 307200 像素预算，入口不缩放图像。统计分别记录客户端请求耗时
`request_elapsed_s`、服务端处理耗时 `service_elapsed_s` 及其有效样本数
`service_timing_samples`。服务端耗时包含排队和预处理，两者均不是纯 GPU 执行时间。
可用 `--shared-plan-tools` 注册 `advance_goal`，并将其加入 `--shared-tools`。
模型判断当前目标已完成并提供证据后，该工具把首个待办目标移入完成记录，保留历史、
顺序和重复访问目标，然后执行下一个目标；最后一个目标完成时请求最终 STOP。
`instruction` 可指定下一段路线，显式重规划仍通过 `navigate` 的替换计划完成。
该工具只更新记录，不验证视觉完成情况。`finish(success)` 在所提交计划或当前记忆中
仍有待办目标时，会在请求 STOP 前被拒绝。模型必须显式更新计划或推进已完成的目标，
省略计划参数不能跳过已有待办。
可选的 `--shared-source-tools` 注册 `plan_task` 和 `continue_goal`，还需将它们加入
`--shared-tools`。上下文为原任务的空白分隔词编号，模型自主选择有序词区间；
harness 原样提取对应子目标并保留完成历史。`continue_goal` 原样继续当前子目标，
保留导航历史。工具不会代替模型拆解任务、判断完成或修补非法区间；其规划能力及
闭环效果仍需单独验证。
实验服务要求和评测参数见
[导航集成说明](https://github.com/yufoo1/EmbodiRun/blob/feat/rpent-activevln-navigation/agents/rpent/navigation/README.md#experimental-shared-model-planner)。
