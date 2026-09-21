# Python API

本页列出可从 `embodirun` 直接导入的推理客户端、数据类型和机器人接口，
供自定义仿真器、硬件适配器和测试框架集成使用。
日常部署与操作可直接使用命令行和配置文件，见[部署配置](configuration.md)和[手动控制](control.md)。

## 推理客户端

::: embodirun
    options:
      members:
        - PolicyClient
        - SglangHttpClient
        - EmbodiInferHttpClient
        - EmbodiInferWirelessClient

## 推理数据类型

::: embodirun
    options:
      members:
        - PolicyObservation
        - PolicyResult
        - PolicyAction
        - ImagePayload
        - Session

## 机器人适配器

::: embodirun
    options:
      members:
        - RobotAdapter
        - RobotObservation
        - RobotAction
