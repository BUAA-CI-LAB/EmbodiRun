# MicroDuck VLN integration

This optional package supplies the MuJoCo/MPC execution and streaming video
adapter used by the [MicroDuck VLN example](../../examples/microduck_vln/README.md).
It also provides a separate ActiveVLN service entrypoint for inference snapshots
that expose `EngineCore` and the versioned HTTP server but do not yet expose an
ActiveVLN `build_serving_adapter` factory.

| Module | Responsibility |
| --- | --- |
| `runtime.py` | Existing external MicroDuck backend and MPC, bounded primitive execution, EGL rendering, MP4 encoding |
| `protocol.py` | Validate decoded R2R action rows before movement; no model text parsing |
| `assets.py` | Check supplied simulation/model/source inventories without copying assets |
| `process.py` | Launch one authenticated loopback service, check readiness and capabilities, terminate the owned child |
| `service.py` | Optional inference process: raw PNG to EmbodiInfer observation, recurrent EngineCore call, decoded rows to HTTP result |

The upper-level episode loop and success metrics live in
`examples/microduck_vln/run_demo.py`. EmbodiRun's installed core and default
dependency lock remain unchanged. The example uses
`embodirun.model_services.backends.vvla.http.VvlaHttpClient` directly;
it is a standalone simulation example, not a registered Host/Control device.

The service uses `vvla.policy.*.v1` sessions/steps/reset/close. Its adapter-specific
action space is `activevln.r2r.discrete.v1`, with one `discrete_chunk` action:

```json
{"type":"discrete_chunk","values":{"rows":[[1,25],[0,0],[-1,0]],"valid":true,"raw_text":"...","stop_reason":"eos"}}
```

Rows are already parsed by the upstream ActiveVLN policy. The executor checks
their shape, finite values and supported magnitudes; it never converts invalid
output into a STOP. The first STOP discards subsequent rows. Model prompts,
tokenization, decoding and recurrent memory remain in the inference process.

Install instructions, external asset requirements, protocol limitations and
validation commands are in the linked recipe. This package follows the
repository's [Apache-2.0 license](../../LICENSE) and [NOTICE](../../NOTICE); external checkpoints, scenes, robot
meshes and SDK sources retain their own licenses and are not bundled here.

The base package installs only the EmbodiRun client and protocol dependency.
Install `simulation` for MuJoCo/MPC/video execution, `service` for the
ActiveVLN inference process, or `full` for both profiles. The `test` extra
contains the CPU test dependencies.

The distribution is named `embodirun-microduck`, the Python package is
`embodirun_microduck`, and its service entrypoint is `embodirun-microduck-serve`.
EmbodiInfer is the upstream distribution name; its public Python namespace and
wire schema remain `vvla`. Those identifiers are intentionally retained.
