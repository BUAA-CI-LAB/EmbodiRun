# ActiveVLN 驱动 Go2 真机

EmbodiRun 将 Go2 相机与硬件 owner 接到 EmbodiInfer 的 ActiveVLN HTTP 服务。
同一条指令可以只执行一轮、连续执行多轮，或让下一轮推理与当前运动重叠。

<video controls muted playsinline preload="metadata" width="720"
       poster="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/activevln_go2/go2_activevln_3x.jpg">
  <source src="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/activevln_go2/go2_activevln_3x.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

[3× 版本（9 秒）](https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/activevln_go2/go2_activevln_3x.mp4) · [部署指南](../go2-activevln.md)

演示以 3 倍速播放，时长 9 秒，包含三种模式：

| 模式 | CLI 参数 | 行为 |
|---|---|---|
| Chunk 1 | `--mode chunk --max-chunks 1` | 观测、推理并执行一轮 |
| Chunk 100 | `--mode chunk --max-chunks 100` | 同步循环，直到 STOP 或任务上限 |
| RTC + Chunk 100 | `--mode overlap --max-chunks 100 --overlap-lead-s 1.5` | 当前动作块执行时开始下一轮推理 |

这里的 1 和 100 是**每条指令最多执行的 chunk 数**，不是单次推理输出的动作数。
每次模型响应最多包含三个动作。视频中的参数与日志经过重新排版，
不是实时终端或逐帧同步的遥测。

## 录制配置与范围

视频录于 2026-10-29，使用 Go2、机载 Orin NX、D435I RGB，以及
ActiveVLN `Arvil/Qwen2.5-VL-3B_rl_r2r_4000`
（revision `160987313e3e869705f42400d1b8f28177044518`）。
JP5 eager 服务在 JetPack 5.1.1 / CUDA 11.4 上加载 BF16 转换权重。
部署指南区分了此配置与较新平台上的优化 benchmark 环境。

RTC 片段保留第一次回合的转向、前进、自然停顿及下一轮前进，
排除了后一次回合中人工将头转回去的镜头。
静态时序图对应该次日志中的 **1.493 秒重叠**和 **1.861 秒轮间等待**。
这展示了执行模式，不是成功率的严格对照实验，也不代表普遍加速比例。
三个片段来自不同任务或回合，不是配对测试。

## 运行三种模式

按部署指南准备好 owner 与推理服务后：

```bash
CONFIG=configs/activevln/go2-orin-nx-jp5-eager.json
embodirun go2 activevln plan --config "$CONFIG"
embodirun go2 activevln check --config "$CONFIG"
embodirun go2 activevln interactive --config "$CONFIG" --mode chunk --max-chunks 1 --execute
embodirun go2 activevln interactive --config "$CONFIG" --mode chunk --max-chunks 100 --execute
embodirun go2 activevln interactive --config "$CONFIG" --mode overlap --max-chunks 100 \
  --overlap-lead-s 1.5 --execute
```

每次运行一种模式。owner 需由操作者显式启用运动；省略 `--execute` 时只推理、不运动。
overlap 要求推理服务支持已执行动作前缀的确认协议。
STOP、运行时限或 chunk 上限均可结束任务；达到上限不表示任务成功。

[复现演示](../examples.md) · [支持矩阵](../support-matrix.md)
