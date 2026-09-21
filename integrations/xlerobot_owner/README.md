# EmbodiRun XLeRobot owner integration

This optional package is the namespace migration of the recovered
`embodied_runtime.teleop` owner. It preserves the legacy hardware/server
implementation, including one serial owner, read-only startup, explicit arm,
deadman input, feedback checks, `stop_unconfirmed` handoff, leader collection,
shared camera capture, browser assets, and crash-tolerant recording/export.

The default dependency is the HTTP/web owner boundary (`aiohttp`). Hardware
SDK, HID, camera, LeRobot dataset export, browser video, and certificate
support are separate extras:

```bash
# Run this from the repository root in the selected uv environment.  The
# integration depends on the local EmbodiRun checkout; installing only from
# this directory would make uv search for an unpublished distribution.
uv pip install -e . -e 'integrations/xlerobot_owner[hardware,hid,camera,recording,video,web]'
```

The portable owner test suite needs only the HTTP owner, NumPy/Pillow recorder
helpers, and the async pytest plugin; hardware SDKs and video/WebRTC remain
optional.  From the repository root, install the local packages and test
extra with:

```bash
uv pip install -e . -e 'integrations/xlerobot_owner[test]'
PYTHONPATH=src:.:integrations/xlerobot_owner/src \
  python -m pytest tests/xlerobot_owner -q
```

Tests use targeted `importorskip` checks at optional runtime boundaries;
HTTP, recorder, and fake-hardware tests should run with the `test` extra, while
WebRTC/`aiortc` remains optional.

The old module is renamed mechanically:

```bash
python -m embodirun_xlerobot_owner serve --mode demo \
  --token-file /path/to/token
```

`prepare`, `serve`, `validate`, and `export` retain their old command names and
wire semantics. The old `embodied_runtime.teleop` package remains a historical
source during migration; do not run both owners against the same serial bus.
Lab-specific AGX launchers are retained under `tools/archive/` for audit and
reference only; the portable owner entrypoint is the explicit module command above.
The AGX-only `nudge_held_joint_and_refresh_snapshot.py` repair script remains
archived and is not a normal owner service.
