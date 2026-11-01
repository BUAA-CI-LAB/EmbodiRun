# RPent / ActiveVLN 闭环对比

R2R val-unseen 固定前 128 集，RTX 4090 单卡同时推理和渲染，BF16、batch=1。
两组使用同一 ActiveVLN R2R checkpoint、初始姿态、640×480 RGB、随机种子 42 和 500 个原子动作上限。
前进步长 0.25 m，转向步长 15°。成功要求最终 STOP 且目标测地距离严格小于 3 m。

正式 128 集对比运行完成后在此记录 SR、SPL、模型调用数、规划工具调用数和总耗时。
第 129 集仅用于连通性验证，不纳入正式统计。

RPent 使用 `gpt-6-astra`、`low` reasoning、每集最多 80 次工具调用和 3600 秒。
每集记忆隔离；只读取指令、RGB、动作记录、碰撞数和已消耗步数。
子目标 STOP 消耗一步，但只有最终 STOP 结束整集。两组顺序按集交替。
模型历史在同一子目标内保留，切换子目标或局部 STOP 后重新执行时重置。

模型：`Arvil/Qwen2.5-VL-3B_rl_r2r_4000@160987313e3e869705f42400d1b8f28177044518`。
推理实现：EmbodiInfer `d53107d`；RPent：`7f403130`。
推理配置开启 resident KV、GPU preprocess、CUDA Graph、fused vision/GQA/packed prefill；
使用该闭环 profile 的 linear decode，两组配置完全相同。

## 复现

安装[规划环境](../../agents/rpent/navigation/README.md)，并启动上述冻结配置的
ActiveVLN HTTP 服务。准备 R2R val-unseen manifest、MP3D 场景和 Habitat-Sim 0.2.4 环境。
将本仓库相同版本放到 GPU 主机，设置 `PYTHONPATH` 包含仓库根目录及 `src`。

GPU 主机执行器（由评测器通过 stdin/stdout 控制，无需另起监听端口）：

```bash
PYTHONPATH=.:src CUDA_VISIBLE_DEVICES=0 NO_PROXY=localhost,127.0.0.1 \
python benchmarks/rpent-navigation/executor.py \
  --endpoint http://127.0.0.1:28391 --token-file /path/to/service-token \
  --habitat-python /path/to/habitat/bin/python --output /path/to/executor-logs
```

把启动该命令的 argv 数组保存到仓库外的 `executor-command.json`，可以是本机 argv
或 `ssh -C ...` 加远端命令。规划端运行：

```bash
PYTHONPATH=. NO_PROXY=localhost,127.0.0.1 \
.venv-rpent/bin/python benchmarks/rpent-navigation/evaluate.py \
  --manifest /path/to/selected-r2r-navigation-256.json \
  --scenes /path/on/gpu-host/to/scene_datasets \
  --simulator-command /path/to/executor-command.json --remote-executor \
  --count 128 --planner-model gpt-6-astra --output /path/to/new-comparison
```

同一输出目录可跳过已完成的集；配置或源码校验不一致时拒绝续跑。
异常中止的集保留原始记录，需明确创建新运行。`protocol.json`、逐集轨迹、规划记录和
`summary.json` 仅保存在实验目录，不提交仓库。
