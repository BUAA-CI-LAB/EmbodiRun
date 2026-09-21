# Quick Start

Start a simulated device, inspect its observations, then configure a robot deployment.

## 0. Install

```bash
git clone https://github.com/BUAA-CI-LAB/EmbodiRun.git
cd EmbodiRun
uv sync --frozen
```

See [`installation.md`](installation.md) for capability groups and extras.

## 1. No robot required

`examples/shared-device-fake.yaml` starts a robot-only Control service with
simulated joints and a fake camera. It opens no hardware and exposes the real
Host JSON API.

```bash
CONFIG=examples/shared-device-fake.yaml

uv run embodirun --config "$CONFIG" validate
uv run embodirun --config "$CONFIG" init
uv run embodirun --config "$CONFIG" sync --source .
uv run embodirun --config "$CONFIG" up

uv run embodirun --config "$CONFIG" describe \
  --runtime fake-device --caller-id example-agent --session-id example-session --json
uv run embodirun --config "$CONFIG" observe \
  --runtime fake-device --caller-id example-agent --session-id example-session --json
uv run embodirun --config "$CONFIG" down
```

`sync --source .` overlays this checkout on the prepared local deployment.
`describe` reports capabilities, `observe` returns one shared observation, and
`media` fetches frame data for an observation ID. Use `execute` to submit an action.
The full self-contained walkthrough, including recording and cancellation, is
in [`examples/shared-device-fake.md`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/shared-device-fake.md).

## 2. Validate a real configuration

Copy an example and replace every placeholder before touching hardware:

```bash
cp configs/pi05/bi-so101-vvla.yaml my-deployment.yaml
$EDITOR my-deployment.yaml

uv run embodirun --config my-deployment.yaml validate
```

`validate` is static: it checks references, ports, bindings, and required
fields without contacting any node. See [`configuration.md`](configuration.md).

## 3. Prepare and start

```bash
uv run embodirun --config my-deployment.yaml probe   # connectivity + tools
uv run embodirun --config my-deployment.yaml init    # environments + sources
uv run embodirun --config my-deployment.yaml up       # start services
```

`init` records successful state locally; `up` refuses to start if the
configuration changed since `init`. Starting Control loads static
configuration and checks model health and cameras, but does not connect or move
the arm.

## 4. Run a task

This command uses the `bi-so101-pi05` runtime from the configuration copied
above. It can move both arms. Complete the [dual-arm setup](pi05-bi-so101.md)
and [operator safety checks](safety.md) before running it.

```bash
uv run embodirun --config my-deployment.yaml run \
  --runtime bi-so101-pi05 \
  --prompt "Pick up the cube and put it into the bowl." \
  --chunk-steps 10 \
  --max-steps 1
```

| Flag | Meaning |
|---|---|
| `--runtime` | Configured runtime ID to use. |
| `--prompt` | Instruction override. Simulators may supply their own. |
| `--task`, `--seed` | Simulator task and episode seed. |
| `--chunk-steps` | Actions executed from each inference chunk (capped by the binding). |
| `--max-steps` | Maximum number of inference/action chunks (default 1). |
| `--control-hz` | Action playback rate (default 5). |
| `--request-timeout` | Timeout for one inference request (default 60 s). |

A π0.5 response has no task-complete signal. The chunk bound limits the run;
errors or cancellation can end it earlier. A requested chunk length above the binding maximum is
rejected before connecting to the robot.

## 5. Stop

```bash
uv run embodirun --config my-deployment.yaml down
```

`down` stops identity-checked processes and remains available after the
configuration changes. `--target control` or `--target model` limits it to one
service kind.

## Troubleshooting

- **`uv` version error** — install uv 0.12.x; the repository pins it.
- **Configuration validation fails** — check required fields and references,
  and replace `REPLACE_*` placeholders. Use `probe` to check node connectivity.
- **`up` refuses to run** — run `init` first, or re-run `init` after the YAML
  changed.
- **A service is not healthy** — check the per-service log path printed by
  `up` for missing files, dependency errors, or unavailable devices.
- **`sync` requires Control stopped** — a running process has already imported
  its modules.

For manual control and the software stop, see [`control.md`](control.md) and
[`safety.md`](safety.md).
