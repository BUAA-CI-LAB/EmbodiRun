# Go2 / ActiveVLN on Orin NX

Use this Go2 integration with the pinned EmbodiInfer ActiveVLN serving adapter.
The entrypoint is `embodirun go2 activevln`. The [physical demo](demos/go2-activevln.md)
shows synchronous chunk execution and execution-prefix-aware overlap on Go2.

Start with `configs/activevln/go2-orin-nx-jp5-eager.json` for the recorded JP5
eager service. `go2-orin-nx-onboard.json` describes the separate optimized
service profile. Both use loopback inference (18767) and one hardware owner
(8080). `plan` is offline, `check` is read-only, and `run` without `--execute`
performs one inference without motion.

The checkpoint revision remains `160987313e3e869705f42400d1b8f28177044518`.
The standalone download and BF16 conversion scripts preserve source weights
and write checksums. Preparation uses its own Python 3.10+ project and lock at
`scripts/activevln-preparation`; it installs neither Torch nor MicroDuck and
does not change the Host or inference environments. With uv 0.12.x, run these
commands from the repository root:

```bash
uv sync --project scripts/activevln-preparation --frozen
uv run --project scripts/activevln-preparation --frozen hf auth login
uv run --project scripts/activevln-preparation --frozen \
  python scripts/download_navigation_models.py \
  --repo Arvil/Qwen2.5-VL-3B_rl_r2r_4000 \
  --revision 160987313e3e869705f42400d1b8f28177044518 \
  --output ../models/activevln-r2r --workers 4 --authenticated
uv run --project scripts/activevln-preparation --frozen \
  python scripts/prepare_activevln_bf16.py \
  ../models/activevln-r2r ../models/activevln-r2r-bf16
```

Conversion requires verified checksums for the config and every indexed shard.
It excludes unverified files, preserves integer tensors and records nearest-even
BF16 conversion. A final `SHA256SUMS` is written only after complete conversion.

The optimized benchmark NX launcher also requires its pinned recorded R2R warmup assets and
256-episode selection manifest. Prepare these assets separately; the JP5 eager profile does not use them. INT8 needs
an additional matching calibration file. The reference NX environment is L4T
R36.4.7 / CUDA 12.6, while the inspected robot uses R35.3.1 / CUDA 11.4; deployment targets the existing JetPack 5.1.1 system. The fresh compute environment
at `/home/unitree/embodirun-envs/compute-jp5` now uses checksummed ARM64/cp310
PyTorch 2.2.0 and torchvision 0.17.2+c1d70fe wheels. CUDA 11.4 / cuDNN 8.6, BF16
GPU operations and camera inference have passed. Source compilation
was stopped. This eager path does not enable the optimized benchmark profile.

## Unified hardware owner

`integrations/go2_owner` is an independently installed Python 3.8+ package. It
owns SDK2 control and camera capture with one lifecycle, HTTP endpoint and token.
Host clients remain in `robots/unitree/go2`; shared mobile-base contracts live in
`robots/mobile_base`; execution loops live in `application/navigation`. Neither
hardware nor client code imports the model service.

The deployed control environment is a fresh uv environment at
`/home/unitree/embodirun-envs/control-jp5`, using Python 3.10.20, privately built
CycloneDDS 0.10.5 and device-built `pyrealsense2==2.55.1+go2.jp5`. The inspected
PyPI ARM64 RealSense wheels require newer glibc despite their manylinux2014 tags.
The environment's `python` wrapper selects its private native libraries and can
serve as both `EDGE_PYTHON` and `HOST_PYTHON`; no Conda/container environment is reused.
CLI install/start/status/stop, live SDK state and D435I RGB/RGB-D observations
have been checked on the Orin with motion readiness disabled. Camera-to-model CLI inference and the physical execution modes shown in the linked demo have been recorded.

Generate configuration on the Orin with its vendor Python, without streaming:

```bash
/usr/bin/python3 scripts/inspect_go2_onboard.py --camera --write-owner-config go2-owner.local.json
export PYTHONPATH="$PWD/integrations/go2_owner/src:$HOME/unitree_sdk2_python"
/usr/bin/python3 -m embodirun_go2_owner --config go2-owner.local.json --validate
/usr/bin/python3 -m embodirun_go2_owner --config go2-owner.local.json
```

Check the generated DDS interface/library paths. The file uses measured camera
serial and depth scale; multiple cameras require `--serial`. Existing files are
never overwritten. Motion readiness is an explicit `--operator-ready` startup
flag; it is never inherited from environment or configuration.

ActiveVLN uses fresh RGB at `/observation/rgb.json` without requiring depth.
`/observation.json` retains matched RGB-D validation. All camera and control routes
share one credential. Capture/HTTP workers do not hold the motor control lock;
shutdown stops motion before joining capture workers.

Remote deployment is model-independent:

```bash
embodirun go2 edge --host "$GO2_HOST" --username "$GO2_USER" --ask-password install
embodirun go2 edge --host "$GO2_HOST" --username "$GO2_USER" --ask-password start \
  --config /path/to/go2-owner.local.json --python "$EDGE_PYTHON" \
  --sdk-pythonpath /home/unitree/unitree_sdk2_python --supervisor-python "$HOST_PYTHON"
```

`EDGE_PYTHON` is the remote vendor SDK Python (3.8+); `HOST_PYTHON` is the remote
supervisor Python (3.10+). `status` and `stop` take the same `--supervisor-python`.
Install uploads only the owner and standalone supervisor. One service ID, `go2`,
replaces independent camera/control services. A running PID does not imply camera
or DDS readiness; use `activevln check` after starting the inference service.

The old `go2 activevln edge` and `embodirun-go2-streamvln` commands forward to this
CLI, but old `--service camera/control` flags and the two agent module entrypoints
are removed. Stop old processes before migrating. A non-loopback bind requires a
32-character-or-longer `GO2_API_TOKEN`. Navigation's `robot_url` supplies both
camera/control endpoints; explicit legacy URLs must agree when `robot_url` is set.
Single-shot `run` still disarms at the end of attempted motion. Use `interactive`
below to accept successive tasks without restarting a healthy owner. Emergency
stop and motion faults still require an intentional restart with operator readiness.
The model service remains separate.

## Repeated instructions and longer tasks

The inference service already keeps model weights loaded. The `interactive` CLI
keeps the terminal open, creates a fresh recurrent session for each instruction,
and closes that session before accepting the next instruction. Steps within one
instruction retain navigation history. This requires updated EmbodiRun **and**
`integrations/go2_owner`; the CLI rejects an old owner before enabling motion.

With the model and owner services running, use the control environment:

```bash
source /home/unitree/embodirun-envs/control-jp5/environment.sh
embodirun go2 activevln interactive \
  --config configs/activevln/go2-orin-nx-jp5-eager.json \
  --max-chunks 100 --max-runtime-s 600 --execute
```

```text
activevln> Move to the open area to the right of the yellow pillar.
... observation / inference / execution repeats ...
Task ended: stop (... chunks). Enter the next instruction.
activevln> Go to the door and stop.
activevln> /quit
```

Each model chunk contains at most three actions. `max_chunks=1` in the initial
first-motion configuration limited the whole instruction to one chunk; the CLI
overrides above allow up to 100 chunks or 600 seconds per instruction. Model STOP
ends the task earlier; reaching a bound does not mean the destination was reached.
Runtime bounds prevent further motion; an in-flight inference/cleanup may delay
returning to the prompt. Model context limits still apply. The 9-second chunk,
speed, heartbeat, and 5-second plan-freshness bounds remain enforced.

Ctrl+C **during a task** stops its lease, drains pending inference, clears that
task's model history and returns to the prompt. Ctrl+C at the prompt, EOF, or
`/quit` exits the CLI. Services remain running. Commands are read between tasks;
use Ctrl+C to interrupt an executing task. Omitting `--execute` makes each entered
instruction perform one inference with no motion writes.

Normal completion and Ctrl+C use an authenticated, action-ID-scoped lease release.
It stops only this task's stream and preserves already-enabled readiness after a
confirmed healthy stop. It cannot re-arm, cancel someone else's action, or hide a
failed lease. `/v1/stop`, SIGTERM during motion, and task faults after motion still
use disarming stop. Operational failures exit the CLI without retrying or re-arming.
The linked demo records physical chunk and overlap execution; it does not
establish navigation reliability.

## Verified JP5 eager service

From the repository directory on the Orin:

```bash
/home/unitree/embodirun-envs/compute-jp5/run embodiinfer-http-serve \
  --policy activevln --checkpoint /home/unitree/models/activevln-r2r-bf16 \
  --adapter-config configs/activevln/infer-jp5-eager.json \
  --device cuda:0 --dtype auto --no-cuda-graph --max-batch 1 \
  --host 127.0.0.1 --port 18767 --max-sessions 1 --max-images 1
```

Use the control environment for `embodirun go2 activevln check/run` with
`configs/activevln/go2-orin-nx-jp5-eager.json`. This synchronous chunk profile
allows a 300-second inference request timeout while retaining the five-second
execution freshness limit. Leave `ACTIVEVLN_BENCHMARK_TOKEN` unset for the
local server above, which has no token configured.

The 2026-10-29 stationary D435I RGB test completed one CLI inference with about
7.72 seconds first-request round-trip latency and three proposed 15-degree right
turns. `executed_actions=0`; no motion was enabled. This does not establish
steady-state performance or closed-loop navigation. Its first-request latency
exceeds the default execution freshness limit.
