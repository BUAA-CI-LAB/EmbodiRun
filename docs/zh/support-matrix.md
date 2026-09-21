# 支持的平台与路线图

按仿真器、机器人或模型查找当前支持的组合，再选择对应部署配置。
计划接入的设备和模型列在下方路线图中。

**✓ 已通过软件测试** — 已实现，并有自动化接口测试。<br>
**◐ 实验性支持** — 集成代码可用，需按平台配置。<br>
**○ 计划支持** — 尚未实现。

## 当前支持

<div class="grid cards support-grid" markdown>

-   ### :material-monitor: 仿真器 {#simulators}

    ---

    **✓ LIBERO**<br>
    π0.5 操作任务，支持 EmbodiInfer 和 SGLang 后端。

    **◐ VLABench**<br>
    π0.5 操作任务。

    **◐ Habitat · Isaac Sim**<br>
    使用 StreamVLN 进行导航。

    [浏览仿真器配置](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/configs/simulation)

-   ### :material-robot: 机器人 {#robots}

    ---

    **✓ SO-101** — 单臂；[真机演示](demos/multi-robot-serving.md)。<br>
    **✓ Bi-SO-101** — 双臂协同控制。<br>
    **✓ Franka FR3** — 适配器与 π0.5 绑定。

    **◐ ARX5** — DM0.5 绑定。<br>
    **◐ Unitree Go2** — StreamVLN 导航。<br>
    **◐ XLeRobot** — 通过独立硬件服务接入。

    [查看机器人适配器](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/src/embodirun/robots)

-   ### :material-brain: 模型 {#models}

    ---

    **✓ π0.5**<br>
    SO-101、Bi-SO-101、FR3 以及仿真器绑定。

    **◐ DM0.5 · StreamVLN**<br>
    ARX5 操作与导航集成。

    **◐ LightNav-0**<br>
    外部 XLeRobot 绑定；模型服务单独提供。

    [EmbodiInfer 模型目录](https://embodiinfer.readthedocs.io/zh-cn/latest/models/)

</div>

## 部署配置 {#deployment-recipes}

=== "仿真器"

    | 环境 | 策略 / 后端 | 配置入口 |
    |---|---|---|
    | LIBERO | π0.5 / EmbodiInfer | [配置](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/simulation/libero-pi05-embodiinfer.yaml) |
    | LIBERO | π0.5 / SGLang | [配置](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/simulation/libero-pi05-sglang.yaml) |
    | VLABench | π0.5 / EmbodiInfer | [配置](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/simulation/vlabench-pi05-embodiinfer.yaml) |
    | Habitat | StreamVLN / EmbodiInfer | [配置](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/simulation/habitat-streamvln-embodiinfer.yaml) |
    | Isaac Sim | StreamVLN / EmbodiInfer | [配置](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/simulation/isaac-streamvln-embodiinfer.yaml) |

    在启动闭环之前，准备仿真器资产、模型检查点和环境依赖。
    Isaac Sim 使用 NVIDIA 的 Isaac Sim EULA。
    依赖组见[安装](installation.md)。

=== "机器人"

    | 机器人 | 策略 / 接口 | 配置入口 |
    |---|---|---|
    | SO-101 | π0.5，独立的单臂运行时 | [共享服务配置](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/http-wireless-inference/http.yaml) |
    | Bi-SO-101 | π0.5，12 维双臂策略 | [双臂指南](pi05-bi-so101.md) |
    | Franka FR3 | π0.5 绑定 | [适配器](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/src/embodirun/robots/franka/fr3) |
    | ARX5 | DM0.5 绑定 | [绑定](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/src/embodirun/bindings/arx/x5/dm05) |
    | Unitree Go2 | StreamVLN 导航 | [机器人集成](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/src/embodirun/robots/unitree/go2) |
    | XLeRobot | 独立硬件服务 | [集成包](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/integrations/xlerobot_owner/README.md) |
    | SO-101（有线遥操作） | 多主臂 UDP 分发，从臂侧 episode 采集 | [集成包](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/integrations/so101_wired_teleop/README.md) |

    SO-101 部署需要已标定的机械臂、相机映射，以及针对所选绑定
    训练的检查点。FR3 需要其机器人 SDK；ARX5 需要厂商电机
    驱动。源码链接是自定义集成的起点。

    在操作物理硬件之前，请遵循[安全](safety.md)和[手动控制](control.md)。
    [LightNav-0](lightnav0_xlerobot.md) 记录了独立的实验性
    XLeRobot 路点绑定。

=== "模型与传输"

    **EmbodiInfer** 提供第一方的 π0.5、DM0.5 和 StreamVLN
    推理服务。其[服务指南](https://embodiinfer.readthedocs.io/zh-cn/latest/serving/)
    涵盖检查点和模型特定的输入映射。

    **SGLang** 有单独安装的 π0.5 集成。**外部
    服务**通过带 `lifecycle: external` 的 provider 连接；
    参见[配置](configuration.md)和
    [外部服务示例](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/examples/external.yaml)。

    **HTTP** 是标准传输。**WirelessComm** 是实验性的、
    单独安装的选项；参见
    [WirelessComm 配置](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/configs/http-wireless-inference/wireless.yaml)
    和[传输测量](inference-transport.md)。

## 路线图 {#roadmap}

<div class="grid cards support-grid" markdown>

-   :material-robot: __机器人__

    ---

    ○ **AgileX PiperX**

    - [ ] 机器人连接与观测适配器
    - [ ] 策略到机器人的动作绑定
    - [ ] 标定与部署示例

-   :material-brain: __模型__

    ---

    ○ **SmolVLA · OpenVLA**

    - [ ] EmbodiInfer 推理适配器
    - [ ] 服务与观测映射
    - [ ] EmbodiRun 部署绑定

    这里的 OpenVLA 指基础模型；OpenVLA-OFT 已经有
    EmbodiInfer 推理适配器。

-   :material-monitor: __仿真器__

    ---

    ○ **更多仿真器**

    - [ ] 选择下一批环境
    - [ ] 观测与动作适配器
    - [ ] 示例部署与闭环检查

</div>

## Agent 与扩展

- **MicroDuck VLN：** 一个实验性的 [MuJoCo 导航示例](microduck-vln.md)，
  带独立集成包、HTTP 推理和视频录制。
- **XLeRobot 零食配送：** 一个实验性的[任务示例](xlerobot-snack-delivery.md)，
  结合录制的底盘路线、π0.5 机械臂动作提议和操作员确认的交接。
- **有线 SO-101 遥操作：** 一个实验性的、单独安装的
  [多主臂集成](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/integrations/so101_wired_teleop/README.md)，
  带 UDP 分发和从臂侧录制。
- **自定义规划器：** [Python 客户端](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/agents/CLIENT.md)
  提供观测读取、动作提议、执行、状态查询和取消接口。
- **RPent：** 一个实验性的 [Agent 集成](rpent-integration.md)。
- **Astra + π0.5：** 一个[协作评审循环](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/agents/astra_pi05/README.md)，
  支持替换评审模块，示例默认使用模拟实现。
- **相机与传输实验：** [相机采集](camera-only-experiments.md)
  和[传输实验](transport-experiments.md)涵盖可选的
  GStreamer、共享内存、Zenoh 和 NIXL 路径。

演示通过[统一启动脚本](examples.md)准备环境、启动服务和执行任务。
服务使用的节点与 GPU 在部署 YAML 中指定。
