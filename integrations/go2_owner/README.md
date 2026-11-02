# Go2 hardware owner

One process owns SDK2 control, camera capture, the shared HTTP endpoint and shutdown.
Motor leases and SDK calls keep their own workers; camera response/encoding work
cannot hold the motor lifecycle lock. Shutdown stops motion before joining capture.

This package supports Python 3.8+ independently of the Host (3.10+) and inference.
It has no base dependencies. Use the vendor-compatible Unitree SDK2, CycloneDDS,
numpy, Pillow and pyrealsense2 in the robot environment. The optional `camera`
extra is for environments where compatible wheels are available; installing the
owner alone never replaces vendor SDKs.

From the repository root, without installing the Host:

```bash
PYTHONPATH=integrations/go2_owner/src python3 -m embodirun_go2_owner \
  --config configs/go2/owner-dry-run.json
```

The dry-run config has no camera and cannot control hardware. `--validate` checks
configuration without opening devices or loading SDKs. A real Orin config can be
created by `scripts/inspect_go2_onboard.py --camera --write-owner-config owner.local.json`.
Check the generated DDS interface and library directory before use. The script
reads the camera serial/depth scale and refuses to overwrite existing files.

Configuration has three sections:

- `server`: one `host`, `port`, and `token_env` (default `GO2_API_TOKEN`).
- `control`: `mode`, DDS `interface`, `cyclonedds_lib_dir`, optional `state_topic`
  and `rpc_timeout_s`.
- `camera`: `null`, or capture options (`backend`, `serial`, `profile`,
  `depth_scale`, `depth_calibrated`, capture timeouts and encoding settings).

Motion requires explicit `--operator-ready`. Configuration and old readiness
variables cannot arm it. Stop disarms until an intentional process restart.
`POST /v1/actions` with `{"action":"release_move","action_id":"..."}`
releases only the named current (or most recently finished) stream lease. After
a confirmed stop with fresh, healthy state it preserves existing readiness for
the next interactive task. It never re-arms or cancels a different lease. Failed
leases, including heartbeat and SDK stop failures, revoke readiness. Clients can
check the `stream_lease_release` capability in `/v1/state` before using this mode.
The existing bounds, fresh-state/tilt checks, 2-second heartbeat watchdog and
10-second motion lease limit remain enforced. Non-loopback listeners require a
32-character-or-longer bearer token shared by every route.

| Route | Purpose |
| --- | --- |
| `GET /health` | Component state and configured capabilities |
| `GET /v1/state` | Robot state and active motion lease |
| `POST /v1/actions` | Bounded actions and lease heartbeats |
| `POST /v1/stop` | Stop and revoke motion readiness |
| `GET /observation/rgb.json` | Fresh RGB with monotonic frame age; no depth requirement |
| `GET /observation.json` | Fresh, registered, sequence-matched RGB-D |
| `GET /health/camera` | RGB health and depth readiness details |
| `GET /frame.jpg`, `/depth.json`, `/depth.png` | Diagnostic camera data |

ActiveVLN uses the RGB endpoint; RGB-D consumers retain strict registration and
freshness checks. Neither path is an obstacle avoidance controller.

Deployment is `embodirun go2 edge` (`deployment/go2`), independent of policy.
It uploads this package plus the shared standalone supervisor, and manages one
service named `go2`. The supervisor requires Python 3.10+; pass
`--supervisor-python` separately from the vendor owner's `--python` when needed.
Stop old camera/control processes before migration. The old two-agent module
entrypoints and `--service camera/control` switches have been removed.

Host clients live in `robots/unitree/go2`; shared contracts in `robots/mobile_base`;
navigation loops in `application/navigation`. There are no Host/model imports here.

For on-device SDK, camera and motion setup, see the [Orin deployment guide](../../docs/en/go2-activevln.md).
