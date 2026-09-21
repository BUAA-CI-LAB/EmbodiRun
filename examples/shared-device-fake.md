# Local simulated device example

For **MicroDuck navigation in MuJoCo with ActiveVLN inference**, see the
[MicroDuck VLN recipe](microduck_vln/README.md). It includes external asset
preparation, GPU/Slurm launch, episode recording and evaluation commands.

For **real VVLA inference with recorded images and simulated execution**, see
[the shared-device inference walkthrough](shared-device-inference.md).

`shared-device-fake.yaml` starts a robot-only Control service with no model or
inference endpoint. The `simulated.joints` adapter keeps five SO-style joint
values and a bounded gripper in memory, while the `fake` camera emits a static
PNG with host monotonic read timestamps. Both report `simulated: true` and
`hardware_access: false`; their observations are software examples, not proof
of physical robot behavior.

For a self-contained walkthrough that starts the real local
`ControlHttpServer`, use the repository helper.  It creates a temporary Host
state directory, runs the actual JSON CLI for describe/observe/execute/inspect,
reads the generated PNG, and exercises recording plus cancellation.  It
accepts only this fake YAML and exits without opening any physical resource:

```sh
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 uv --no-config run --no-project \
  --offline --python 3.12 --with 'uv>=0.12,<0.13' \
  --with 'PyYAML>=6,<7' --with 'rich>=13,<15' --with 'paramiko>=3.4,<5' \
  python examples/run_shared_device_fake.py
```

The helper's echoed commands include the temporary `--state-dir`; copy them
only while that process is serving.  The directory is an ephemeral demo
fixture and is not a production deployment state store.

Alternatively, validate and start it with the normal Host lifecycle commands:

```text
embodirun --config examples/shared-device-fake.yaml validate
embodirun --config examples/shared-device-fake.yaml init
# When running from this checkout, overlay the checkout before starting the
# service so the simulated adapters and Host API are available on the node.
embodirun --config examples/shared-device-fake.yaml sync --source .
embodirun --config examples/shared-device-fake.yaml up
```

Then use the Host JSON boundary to inspect the running local service. Stable
caller and session IDs keep ownership and observation scope consistent across
separate CLI invocations:

```sh
CONFIG=examples/shared-device-fake.yaml
embodirun --config "$CONFIG" describe --runtime fake-device \
  --caller-id example-agent --session-id example-session --json
embodirun --config "$CONFIG" observe --runtime fake-device \
  --caller-id example-agent --session-id example-session --json
```

When the service publishes media, pass the returned `observation_id` to the
JSON media command (frame data is explicitly opt-in):

```sh
OBSERVATION_ID=observation-id-from-json
embodirun --config "$CONFIG" media --runtime fake-device \
  --caller-id example-agent --session-id example-session \
  --observation-id "$OBSERVATION_ID" --include-data --json
```

Recorder commands report `unsupported` when no recorder is configured; they do
not create a second local recorder. Use `recording-status` for a read-only
check, and use `recording-start`, `recording-stop`, or `recording-get` only
when the service description says recording is available.

The legacy Host `run` command rejects this runtime with an explicit
no-inference message. The Control `execute` command is the bounded direct
action path for this fake robot; attach a model and binding in a separate
runtime when model inference tasks are required.

For the full bounded action, inspect, cancellation, and timeout guidance, see
[Agent execution workflow](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/docs/en/agent-workflow.md).
It keeps the fake action workflow software-only and requires inspecting the
original request ID after an uncertain transport result.
