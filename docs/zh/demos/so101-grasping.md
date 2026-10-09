# VLA：SO-101 抓取

给 SO-101 机械臂一条指令：**“拿起方块并放入碗中。”**
π0.5 根据前视、腕部相机图像和关节状态预测动作。
EmbodiRun 采集观测、请求动作块、校验动作，再通过机器人适配器执行。

<video controls muted playsinline preload="metadata" width="720"
       poster="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/multi_robot_serving.jpg">
  <source src="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/multi_robot_serving.mp4" type="video/mp4">
  你的浏览器不支持内嵌视频。
</video>

*三段 SO-101 方块入碗任务录像并排展示。*

## 从观测到动作

1. 采集前视、腕部图像，以及六维关节与夹爪状态。
2. 将观测和指令发送给 π0.5 服务。
3. 将返回的动作块映射到机器人，检查限位，以 20 Hz 回放 50 步动作。
4. 采集下一轮观测，重复执行。

Recipe 使用一台机械臂和 EmbodiInfer HTTP 服务。
模型检查点需要匹配机器人的相机角色、特征名称和动作单位。

## 运行 Recipe

按 [SO-101 抓取 Recipe](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/so101_grasping.md)
配置机器人、相机、标定、检查点和 SSH 主机。

在仓库根目录预览默认配置：

```bash
uv sync --frozen
bash examples/run.sh examples/engine_comparison/embodiinfer-http.yaml validate
bash examples/run.sh examples/engine_comparison/embodiinfer-http.yaml plan
```

这两个命令在本地检查配置并显示启动计划。环境准备、服务启动、执行和停止命令见 Recipe。
启用真机运动前请阅读[安全指南](../safety.md)。

## 扩展配置

- [多台 SO-101 共享模型服务](multi-robot-serving.md)。
- [推理引擎与传输对比](engine-e2e-contrast.md)。
- [配置 Bi-SO-101 双臂策略](../pi05-bi-so101.md)。
- [查看机器人与仿真器支持情况](../support-matrix.md)。
