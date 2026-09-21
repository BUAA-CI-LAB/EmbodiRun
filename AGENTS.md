# EmbodiRun

## Repository purpose

EmbodiRun is the deployment and execution runtime for embodied AI. It connects
robot and simulator observations, inference services, session and step
coordination, action validation, and robot-specific motion execution.

It owns robot and simulator observations, HTTP/WirelessComm inference clients,
session and step coordination, action validation, robot-specific motion
execution, and deployment routing. It does not own model checkpoints, model
inference, prompt construction, model frameworks, or model-output parsing.

Do not import runtime code from `third_party/embodiinfer` into `src/embodirun`.
Communication with EmbodiInfer must go through the versioned HTTP or
WirelessComm API.

## Project structure

- `src/embodirun/client`: public Agent HTTP client and shared transport semantics
- `src/embodirun/deployment`: configuration, planning, state, environment/source preparation, and SSH/process lifecycle
- `src/embodirun/application`: jobs, authentication, execution coordination, policy proposals, and default runtime loops
- `src/embodirun/devices`: connection ownership, shared observations, execution arbitration, and recording
- `src/embodirun/model_services`: inference contracts, protocol clients, providers, and launch descriptors
- `src/embodirun/services`: CLI/HTTP process entrypoints and compatibility imports for the former layout
- `src/embodirun/robots`: robot interfaces and adapters
- `src/embodirun/bindings`: policy-to-robot bindings
- `src/embodirun/simulators`: simulator adapters
- `tests`: unit, boundary, and integration tests
- `agents`: optional upper-layer algorithm integrations and software examples, outside the installed core package
- `integrations`: separately installed model-service and hardware-owner packages;
  each package owns its optional SDK dependencies and process entrypoint
- `third_party/embodiinfer`: optional pinned source for the first-party inference engine
- `configs`: example deployment configurations
- `examples`: runnable demo manifests, the shared launcher, and scene implementations
- `benchmarks`: measurement programs and their experiment-specific supporting tools
- `docs/en`, `docs/zh`: parallel English and Simplified Chinese documentation;
  shared assets and the stylesheet stay under `docs/`

## Development commands

Install the package for development:

    uv sync --frozen

Run the full test suite:

    uv run pytest

Run a specific test file:

    uv run pytest tests/<test_file>.py

## Repository conventions

- License: Apache-2.0. See `LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES.md`.
- Support Python 3.10 and newer.
- Keep robot-specific dependencies optional.
- Declare dependencies in `pyproject.toml`.
- Preserve the separation between generic robot adapters and policy-specific
  bindings.
- Keep `src/embodirun/services` compatibility imports working; new code targets
  the canonical domains (`client`, `deployment`, `application`, `devices`,
  `model_services`).
- Do not modify `third_party/embodiinfer` unless the task explicitly targets the
  pinned upstream code.
- Do not perform operations against physical robots unless the user explicitly
  requests them.
