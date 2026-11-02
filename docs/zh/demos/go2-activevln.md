# Go2 真机演示

EmbodiRun 在 Go2 上执行 EmbodiInfer 输出的 ActiveVLN 动作，使用机载 Orin NX
和 D435I RGB 相机。录像展示三种执行模式。

<video controls muted playsinline preload="metadata" width="720"
       poster="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/activevln_go2/go2_activevln_3x.jpg">
  <source src="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/activevln_go2/go2_activevln_3x.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

[观看视频 — 3 倍速，9 秒](https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/activevln_go2/go2_activevln_3x.mp4)

| 模式 | 录像中的行为 |
|---|---|
| Chunk 1 | 观测、推理并执行一个动作块。 |
| Chunk 100 | 同步循环执行动作块，直到 STOP 或任务上限。 |
| RTC + Chunk 100 | 当前动作块执行时开始下一轮推理。 |

1 和 100 表示每条指令最多执行的动作块数，每次推理最多返回三个动作。
录像拍摄于 2026-10-29，使用 ActiveVLN Qwen2.5-VL-3B、BF16，
以及 JetPack 5.1.1 / CUDA 11.4。

RTC 片段保留自主转向和随后的前进，已排除后续人工将头转回去的镜头。
三个片段来自不同回合，命令与时序标注依据日志重新排版，用于展示执行模式，
不构成严格的速度对照实验。

实现与部署说明见尚待整理合入的
[Go2 适配 PR](https://github.com/BUAA-CI-LAB/EmbodiRun/pull/13)。
