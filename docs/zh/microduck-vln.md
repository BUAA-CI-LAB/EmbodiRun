# MicroDuck 视觉语言导航仿真

MicroDuck 示例在 MuJoCo 中运行视觉语言导航。EmbodiRun 的 HTTP 客户端管理会话并调用 ActiveVLN，
MPC 和 ONNX 行走策略负责执行 R2R 导航动作，同时录制第一、第三人称视频和每轮任务指标。

[观看导航演示](demos/microduck-vln.md)，了解启动顺序、
相机视角和执行日志。

完整说明位于
[示例指南](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/microduck_vln/README.md)。
其中包括环境搭建、外部资源目录和校验和、本地 GPU 与 Slurm 运行方式，
以及自定义任务文件、输出格式和常见问题。

## 运行示例

准备好独立环境、检查点、数据集和场景资源后，运行：

```bash
git submodule update --init third_party/embodiinfer
cp examples/microduck_vln/example.yaml examples/microduck_vln/example.local.yaml
# 编辑 YAML，填入资产路径、Python 环境和资源选择。
bash examples/run.sh examples/microduck_vln/example.local.yaml check
bash examples/run.sh examples/microduck_vln/example.local.yaml run
```

该示例在仿真中运行，使用本地准备好的检查点、数据集
和场景资源。示例指南也提供了通过 `sbatch` 提交任务的方法。

## 参考配置

此实验性集成使用独立的仿真环境，参考配置如下。

| 设置 | 参考配置 |
| --- | --- |
| 硬件 / 运行时 | Linux，一块 A800 80 GB GPU，四个 CPU 核心，EGL；Python 3.12 |
| 策略 | Qwen2.5-VL-3B ActiveVLN，外部提供的合并 SFT-v3 检查点 |
| 模型环境 | Torch 2.11 / torchvision 0.26，Transformers 4.51.3，tokenizers 0.21.4 |
| 仿真环境 | MuJoCo 3.8.1，ONNX Runtime 1.30，CasADi 3.7.2 |
| 推理来源 | 由 `third_party/embodiinfer` 固定的 EmbodiInfer 修订版本 |
| 执行 | B=1，eager，bfloat16，SDPA，贪心解码；每个 episode 使用有状态会话 |
| 场景 / 限制 | `val_2`，默认出生点 `(6.5, 13.8, 0)`，最多 60 个基元动作 |
| 成功 | STOP 且最后三个动作端点严格位于 1.0 m 半径内 |

## 查看评估结果

示例指南提供了评估 40 轮任务（episode）的命令。结果写入 `results.json`，
其中 `status=complete` 表示运行结束；任务成功与否按上表中的 STOP 和距离条件判断。
`spl_euclidean` 按直线距离计算，不使用沿场景可通行路径的测地距离。
