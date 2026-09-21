# Python API

This reference covers inference clients, data types, and robot interfaces
exported from the `embodirun` package root.

A deployment normally drives the runtime through its command-line entry points and its
configuration files, covered in [Configuration](configuration.md) and
[Control](control.md). This page is for callers that embed the runtime in Python — a
custom simulator, a hardware adapter, or a test harness.

## Inference clients

::: embodirun
    options:
      members:
        - PolicyClient
        - SglangHttpClient
        - VvlaHttpClient
        - VvlaWirelessClient

## Inference contracts

::: embodirun
    options:
      members:
        - PolicyObservation
        - PolicyResult
        - PolicyAction
        - ImagePayload
        - Session

## Robot adapters

::: embodirun
    options:
      members:
        - RobotAdapter
        - RobotObservation
        - RobotAction
