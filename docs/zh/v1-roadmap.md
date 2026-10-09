# V1 开源方案与交付清单

这页是 V1 的公开范围、复现入口和证据清单。缺少实测数据的项目保留为“待补充”，不会用估算值替代。

## 项目定位

EmbodiRun 把机器人控制与模型计算解耦：控制可以在机器人侧运行，推理可以放在边缘 GPU、共享 GPU 服务或云端。部署可以使用独立后端、多个设备共享后端，或按设备混合使用多个后端；一控多和多控多都由部署配置表达。EmbodiInfer 负责模型执行、批处理和服务协议，EmbodiRun 负责设备、任务、动作执行和生命周期。

这套拆分适合经济型机器人和算力复用场景：硬件不需要为每台设备配一块高端 GPU，用户可以先用 fixture 或仿真验证流程，再替换为真实设备。

## 三个主场景

| 场景 | 主 Demo | Recipe 扩展 | 当前状态 |
|---|---|---|---|
| VLA | SO-101 抓取 | ARX5、Franka、松灵 | SO-101 已有配置；其余按支持矩阵标注 |
| VLN | MicroDuck 导航 | Go2 等移动平台 | MicroDuck 为实验性示例 |
| Agent | 小车 + SO-101 + RPent/Astra | 其他移动操作组合 | 需要补齐采集到部署的完整材料 |

首页只保留主 Demo；其他设备、模型和后端通过 `examples/` 与配置 Recipe 提供。

## Recipe 使用前提

每个真机 Recipe 都要明确以下输入：Python/uv 或 Docker 版本、机器人 SDK、标定文件、相机角色映射、控制限幅、模型 checkpoint、checkpoint binding、动作单位、路线或任务文件、模型 endpoint、急停和人工接管方式。`check` 只验证声明和文件，不代表设备连接、标定、模型输出或任务成功。

## 证据清单

| 证据 | 记录位置 | 状态 |
|---|---|---|
| 硬件预算 | 本页或对应 Recipe 的 `hardware.md` | 🟨 待补充 |
| 首次部署耗时、步骤数 | `benchmarks/deployment-readiness.md` | 🟨 待补充 |
| CPU/GPU/显存/内存占用 | 对应 benchmark 原始数据 | 🟨 待补充 |
| EmbodiRun 与手工部署对照 | `benchmarks/deployment-readiness.md` | 🟨 待补充 |
| 推理、部署和多机扩展曲线 | `benchmarks/` 原始数据与脚本 | 部分已有 |
| WirelessComm 延迟、吞吐、失败率、恢复时间 | `benchmarks/inference-transport/` | 部分已有 |

没有原始数据和生成脚本的图表不能作为最终性能结论。

## 生态边界

EmbodiRun 位于应用任务与底层驱动之间。机器人 SDK、ROS 2、相机驱动和急停仍由设备侧负责；EmbodiRun 通过适配器、观测契约和动作绑定接入它们。ROS 2 的正式桥接仍是待补充项，当前文档只描述边界，不宣称已有桥接实现。

## 后续贡献入口

- [ ] `good first issue`：新增一个 fixture 或配置校验
- [ ] `good first issue`：补充一个中英文 Recipe 页面
- [ ] `good first issue`：为现有 benchmark 添加原始数据和绘图脚本
- [ ] `help wanted`：补齐部署成本对照和多机扩展数据
- [ ] `help wanted`：实现硬件信息采集、模型/设备候选检索和算子优化报告

提交代码时请同时更新英文页面、中文页面、支持矩阵和 Recipe 前提；缺少实测结果时保留“待补充”标记。
