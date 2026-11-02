# 在 EmbodiRun 中运行 ActiveVLN 三引擎

**EmbodiRun** 负责同一套 Habitat 观测和动作执行循环，EmbodiInfer、SGLang、vLLM
分别提供推理。原速视频仅显示总耗时和实时距目标的测地距离。

<video controls muted playsinline preload="metadata" width="720"
       poster="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/activevln_engines/activevln_engines_1x.jpg">
  <source src="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/activevln_engines/activevln_engines_1x.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## 录制任务

三组都运行 R2R validation-unseen 的 episode 17，场景为 `2azQ1b91cZZ`：
穿过台球室，到达对面能看到花园的窗户。它们采用相同初始状态、检查点
（`Arvil/Qwen2.5-VL-3B_rl_r2r_4000`，revision
`160987313e3e869705f42400d1b8f28177044518`）、RTX 4090、BF16、batch 1、
贪心解码、640×480 RGB、25 cm 前进步长和 15 度转向。

| 推理引擎 | 任务耗时 | 最终距目标 | 原子动作步数 / 推理次数 |
|---|---:|---:|---:|
| EmbodiInfer | 1.979 s | 0.673 m | 49 / 11 |
| SGLang | 2.736 s | 0.673 m | 49 / 11 |
| vLLM | 2.682 s | 0.673 m | 49 / 11 |

三组均在距离目标 3 m 的成功阈值内发出 STOP。这是 2026-11-02 录制的单个任务，
不是整个导航数据集的统计结果。

计时从创建 session 后开始，到实际 STOP 结束，包含 HTTP、推理和仿真；
不包含服务启动、场景加载、一次预热任务和离线编码。
画面使用真实单调时钟采集时间。成片 4.9 秒，其中片头 0.65 秒、片尾 1.5 秒；
保持 1× 原速，没有慢放或生成中间运动帧。

SGLang 首次运行停在 2.458 m。重启服务后三次录制均停在 0.673 m；
视频使用事先指定的**第一次**复跑，不是最快的一次。
原始结果保留在实验归档中，跨服务启动的输出变化原因尚未确定。
各引擎使用各自调优配置；相同精度和检查点不意味着内核相同。

## 接入与复现

新增的 [vLLM 服务包](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/integrations/vllm_activevln)
提供 `provider: vllm` 动作接口，[Habitat 示例](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/examples/activevln_engines)
给出部署配置。EmbodiInfer 与 SGLang 已有接入继续保留。
SGLang 已封装的 provider 面向 π0.5；本视频的 ActiveVLN 使用实验中的原生引擎 wrapper
与通用动作 HTTP 协议。

录像来自当时实际比较使用的实验服务。此次整理的可复用 vLLM 包通过 CPU 协议及失败路径测试，
尚未重新进行该封装入口的 GPU 验证，其默认参数也不承诺复现表中耗时。
封装验证与录像结果分别说明。
