# 用 LightNav-0 驱动 XLeRobot 底盘

这个实验性绑定将模型预测的局部路点转换为 XLeRobot 底盘命令。
推理由外部服务完成，相机读取和运动执行通过机器人侧 HTTP 服务完成。
目前使用独立命令行入口，尚未接入 Host 配置和统一控制仲裁。

## 准备依赖和推理服务

该绑定需要实现下方 LightNav-0 契约的外部推理服务。
你需要自行提供该服务：`lightnav0` 尚未
收录在 EmbodiInfer 的公开模型目录中。

```bash
uv sync --frozen --no-dev --group binding-lightnav0
```

这仅安装 NumPy 和 Pillow。设备依赖由外部机器人服务提供。
当推理服务使用 Bearer 认证时，将 `--inference-token-env` 设置为
某个环境变量的名称。

该绑定发送一张名为 `observation.images.rgb` 的无损 PNG、导航
指令和采集时间。它不向模型发送仿真器位姿、地图、目标坐标
或特权导航提示。`lightnav0.waypoints.v1` 响应
必须包含十个有限的 `[forward_m, left_m, ccw_rad]` 累积局部路点
以及一个显式的布尔值 `stop`。现有 HTTP 会话/reset/step API 会被复用。

## 连接机器人服务

先启动 XLeRobot HTTP 服务，提供控制权认证、轮式底盘反馈，以及带时间戳的最新相机帧。
本仓库提供访问该服务的客户端，机器人侧服务需另行准备。
客户端先检查推理服务是否就绪，再申请机器人控制租约。

```bash
.venv/bin/python -m embodirun.bindings.xlerobot.lightnav0.cli \
  --robot-url http://robot-host:8080 --robot-token-env XLEROBOT_TOKEN \
  --inference-url http://inference-host:8050 \
  --inference-token-env INFERENCE_TOKEN \
  --instruction 'go to the door' --camera front --authorize-motion
```

`--authorize-motion` 表示允许底盘运动，省略时内置接口会拒绝运动命令。
若使用 `--robot-factory module:function`，该函数需返回已连接且已授权的机器人。
图像或反馈过期、控制权丢失、请求超时或模型返回 stop 时，均会进入停止和资源清理流程。

## 验证进展

CPU 测试覆盖局部坐标系变换、保持曲率的限速、
过期观测、授权、清理和 HTTP 客户端校验。
本机 HTTP 冒烟测试使用模拟模型。真实检查点的 HTTP
评估、导航成功率与真机验证仍未完成。
