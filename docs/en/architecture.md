# Architecture

EmbodiRun deploys inference and control services, sends observations to a model,
and maps the returned actions to a robot or simulator. Models run in separate
inference services such as EmbodiInfer or SGLang.

## Process boundaries

```mermaid
flowchart LR
  subgraph operator["Operator machine"]
    host["Host / CLI<br/>(embodirun)"]
  end
  subgraph compute["Compute node"]
    inference["Inference service<br/>(EmbodiInfer / SGLang)"]
  end
  subgraph robotnode["Robot node"]
    control["Control service"]
    hardware["robot / sensors"]
  end
  host -- "SSH: plan, prepare, start, stop" --> inference
  host -- "HTTP through an SSH tunnel" --> control
  control -- "HTTP or WirelessComm" --> inference
  control --> hardware
```

- **Host** runs only on the operator machine. It resolves configuration into a
  plan, prepares environments and sources over SSH, starts and stops
  identity-checked processes, and forwards tasks over a tunneled HTTP
  connection.
- **Control** runs beside the robot and owns the hardware. It owns observation
  sharing, execution arbitration, manual input, recording, and the loopback
  task API.
- **Inference** is a separate service. EmbodiInfer is the first-party engine;
  SGLang is supported through a provider. External services are never started
  or stopped by Host.

The nodes can be the same machine. The boundaries are process and environment
boundaries, not physical ones.

## Runtime domains

| Domain | Responsibility |
|---|---|
| `deployment` | Configuration, planning, state, environments, sources, SSH and process lifecycle. |
| `application` | Jobs, proposals, authentication, execution coordination, and the default model loop. |
| `devices` | Connection ownership, shared observations, action arbitration, teleoperation, recording. |
| `model_services` | Versioned inference contracts, HTTP/WirelessComm clients, provider registry. |
| `client` | Dependency-free public Agent client for the Control API. |
| `robots`, `bindings`, `simulators` | Hardware adapters, policy-to-robot mappings, simulator adapters. |
| `services` | CLI and HTTP process entrypoints. |

## Deployment plan

`build_plan(config)` turns a YAML file into nodes, environment profiles,
services, and runtimes. It validates references, detects port conflicts, and
generates the adapter configuration each binding needs. The same plan drives
`init`, `up`, `down`, and `sync`, which is why `up` can detect a configuration
change since initialization.

## Inference contract

Control talks to inference over a versioned, model-neutral API
([`http_api.md`](http_api.md)). A session carries a stable `session_id`,
monotonic `step_id`, and unique `request_id`. Repeated request IDs are
idempotent while their responses remain cached. Requests include a task
instruction; checkpoint loading, model-specific prompt construction, and
tokenization belong to the inference service. The same schemas are carried by HTTP and WirelessComm; image bytes
are separate payload segments rather than base64.

## Execution path

```mermaid
flowchart TB
  observation["Observation<br/>Control reads the robot and mapped sensors"]
  request["Inference request<br/>the model service returns a policy action chunk"]
  mapping["Action mapping and validation<br/>the binding maps policy rows to RobotAction and checks limits"]
  execution["Bounded execution<br/>the arbiter plays the chunk, or manual input takes over"]
  feedback["Execution feedback<br/>job state, timing, and the next observation"]
  observation --> request --> mapping --> execution --> feedback
  feedback -. "next chunk" .-> observation
```

An execution receipt reports which actions ran and how they ended. The
application evaluates task success from subsequent observations or task metrics.

## Extending EmbodiRun

- **A new robot or simulator** implements the adapter interface and registers
  its type; a runtime then refers to it by `type`.
- **A new policy** adds a binding that maps observations and action chunks.
- **A new inference backend** registers a provider and ships a client; the
  deployment and execution path is unchanged.
- **A new agent** uses the public `client` and keeps its own planning loop.
