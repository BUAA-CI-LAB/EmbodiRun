# Installation

Install EmbodiRun from source with [uv](https://docs.astral.sh/uv/).

## Prerequisites

- Linux (x86_64 or ARM64, depending on the capability group).
- Python 3.10 or newer. Some groups require newer Python: `robot-so101`,
  `sim-libero`, `sim-vlabench`, and `sim-isaac` use 3.12; `sim-habitat` uses
  3.11.
- [uv](https://docs.astral.sh/uv/) 0.12.x. The repository pins
  `required-version = ">=0.12.0,<0.13"`.
- `git` and network access to GitHub.

## Source install

```bash
git clone https://github.com/BUAA-CI-LAB/EmbodiRun.git
cd EmbodiRun
uv sync --frozen
```

`uv sync --frozen` installs the core package, the `host` group, and the
development tools. It is enough for `embodirun validate`, the Host lifecycle,
and the test suite.

For runnable recipes, use the same source checkout as the entrypoint:

```bash
uv sync --frozen --python 3.12
uv run --frozen embodirun example init xlerobot
CONFIG=examples/local/xlerobot/example.local.yaml
uv run --frozen embodirun example "$CONFIG" validate
uv run --frozen embodirun example "$CONFIG" plan
uv run --frozen embodirun example "$CONFIG" setup --mode software
uv run --frozen embodirun example "$CONFIG" check --mode software --json
uv run --frozen embodirun example "$CONFIG" dry-run
```

`init` creates linked, editable files in a new ignored directory. `setup`
uses Python 3.12 and uv 0.12.x. For custom `--output`, prefer
`examples/local/` or a directory outside the checkout
so generated settings and credentials stay outside the Docker build context.
`runs/` and generated Control credential files are excluded from Git and Docker.
The checkout Host environment uses the root `uv.lock` and includes the default
development group. The separate XLeRobot environment defaults to
`.venv-xlerobot-snack` and uses the dependency project and `uv.lock` under
`examples/xlerobot_snack_delivery/` for both modes:

| Mode | Installs | Next check |
|---|---|---|
| `setup --mode software` | Small runtime environment without owner or PyTorch | `check --mode software --json`, then fixture `dry-run` |
| `setup --mode hardware` | Full owner extra, including locked LeRobot, CPU PyTorch, cameras, and recording | `check --mode hardware --json`, then operator verification |

XLeRobot `setup` / `check` default to hardware mode when `--mode` is omitted. Hardware
check reports the installed LeRobot parent directory as `environment.sdk_src`;
use it for `hardware.local.json`'s `sdk_src`. A custom SDK checkout needs its own
version verification. Installing dependencies does not fill calibration,
devices, routes, or a model checkpoint. Inference runs in a separate service.
MicroDuck defaults to a separate `.venv-microduck` environment. Native setup
and its container targets use `examples/microduck_vln/uv.lock` with Python 3.12
and uv 0.12.x. `setup --mode software` installs the small CLI/integration profile;
`check --mode software --json` checks that environment and configuration without
assets, a GPU, or a running scene. `setup --mode simulation` adds the locked
`simulation` extra, including simulator and inference packages. MicroDuck
defaults to `simulation` when mode is omitted; installing it does not supply
assets or prove CUDA/EGL or model readiness. See the [MicroDuck Recipe](microduck-vln.md).

MicroDuck software validation on Ubuntu 24.04 aarch64 installed the same 14
locked packages natively (Python 3.12.3) and in Docker (Python 3.12.14).
Both use Python 3.12, with different patch versions; both entrypoints passed
plain and JSON checks. The initial uncached native online setup reached an
external 180-second test limit (interrupted, exit 130); normal frozen setup
then passed using freshly downloaded wheel assistance. The empty-cache
Docker build reached an external 600-second test limit (interrupted, exit 130);
the same production Dockerfile passed on a warm retry (exit 0). These results
do not establish fast uncached installation or independent first use.
One fresh full-profile native install also reached the 600-second test limit
before completion: imports and CUDA/EGL probes were not run, and the full
Docker target was not built in this validation round. Scene/model acceptance
remains open.

The Recipe software CI jobs install XLeRobot and MicroDuck from frozen locks
without restored uv caches, build their production software targets in fresh
Docker builders, run both entrypoints as the host UID with container networking
disabled, and compare native/container package inventories. XLeRobot also runs
its fixture rehearsal through both entrypoints. Printed hints preserve the
selected scene Python and work outside the checkout with space-containing paths.
These checks cover software onboarding; full simulation, native LeRobot cost
comparisons and independent human first use remain follow-up acceptance work.

If the initial `uv sync` or `docker compose build` fails before the Recipe CLI
starts, no Recipe `Outputs:` or `run.json` exists yet. Save the terminal
stdout/stderr to your chosen log file, with the exact command and exit code.
During Recipe setup, read `command-0.log` under the printed `Outputs:` directory. Keep
the first failure and exit code, then retry the same mode after correcting the
cause. For an explicit native uv HTTP read timeout, a retry can use
`UV_HTTP_TIMEOUT=300 uv run --frozen embodirun example "$CONFIG" setup --mode software`
(repeat the failed mode: `hardware` for XLeRobot or `simulation` for full MicroDuck). This host variable is
not automatically passed to `docker compose build`. Record download cache use
and retries; software rehearsal time does not measure a full owner installation.

XLeRobot `check` is a read-only local check. A `passed` report still leaves
actual calibration, camera images, model output, fresh device observations,
and physical stop feedback for verification on the target host before motion.
See the [recipe](xlerobot-snack-delivery.md) and [support matrix](support-matrix.md#xlerobot-recipe-prerequisites).

### Capability groups

Choose the dependency group for the robot or simulator on each node. Use
separate environments when running different device stacks.

| Group | Adds | Python |
|---|---|---|
| `host` | SSH orchestration: paramiko, PyYAML, rich | 3.10+ |
| `robot-so101` | Feetech SDK, OpenCV for SO-101 | 3.12+ |
| `robot-fr3` | `franky-control` for Franka FR3 (Linux x86_64) | 3.10+ |
| `robot-arx5` | Pillow, RealSense, OpenCV (vendor motor driver separate) | 3.10+ |
| `robot-go2` | no extra Python dependency (Unitree SDK2 is vendor-supplied) | 3.10+ |
| `robot-xlerobot-external-owner` | no extra dependency in core | 3.10+ |
| `sim-libero` | LeRobot + LIBERO, Pillow | 3.12+ |
| `sim-habitat` | habitat-sim (built from source), OpenCV, Pillow | 3.11 |
| `sim-isaac` | Isaac Sim (NVIDIA EULA), OpenCV, Pillow | 3.12 |
| `sim-vlabench` | LeRobot, VLABench, Pillow | 3.12+ |

Examples:

```bash
uv sync --frozen --no-dev --group host
uv sync --frozen --no-dev --group robot-so101
UV_PROJECT_ENVIRONMENT=.venv-robot-arx5 uv sync --frozen --no-dev --group robot-arx5
```

`robot-arx5` needs the vendor `bimanual` extension built for the Python in its
environment; set the ARX5 robot option `sdk_path` to its absolute import
directory on the control node. ARX5 also needs the vendor native/ROS libraries
and a configured SocketCAN interface.

### Installer scripts

Two helpers cover the common cases:

- `scripts/install.sh` syncs this checkout with a chosen group and extra:

  ```bash
  scripts/install.sh --group robot-so101
  scripts/install.sh --group host --extra sglang
  ```

- `requirements/install.sh` installs a standalone environment at a chosen path
  and Python version, with optional component extras and a selectable PyTorch
  version. Run `bash requirements/install.sh --help` for the full option list.
  `EMBODIRUN_ENV_ROOT` and `EMBODIRUN_PYTORCH_INDEX` override its defaults.

## Extras

| Extra | Contents |
|---|---|
| `sglang` | SGLang diffusion serving (Linux, Python 3.12+, glibc >= 2.34) |
| `wireless` | WirelessComm data plane, pinned to its released tag |

The extra pins
[`BUAA-CI-LAB/WirelessComm`](https://github.com/BUAA-CI-LAB/WirelessComm)
to `v0.1.0`. It is not on a package index yet, so the requirement is a Git
URL:

```bash
uv sync --frozen --extra wireless
```

Then select the transport in the deployment YAML
(`models.<id>.transport: wireless`) as described in
[`http_api.md`](http_api.md). WirelessComm peer IDs and its optional token do
not encrypt or authenticate the link; use it only on a trusted network.

## Managed deployments {#managed-deployments}

`embodirun init` checks out the pinned EmbodiRun and EmbodiInfer revisions in
each node's deployment directory and runs `uv sync --frozen` there.

The `third_party/embodiinfer` submodule pins the inference server revision used
for reproducibility. Initialize it only if you want the pinned tree:

```bash
git submodule update --init third_party/embodiinfer
```

### Containers {#containers}

Install Docker Engine, the Compose plugin and Buildx with BuildKit support.
The Dockerfile's cache mounts require BuildKit; the legacy builder cannot build it.
Check `docker buildx version` and `docker compose version` before building.

The recipe container targets are `host`, `xlerobot-software`, `xlerobot`,
`microduck-software`, and `microduck` in the
repository `Dockerfile`. Build them on the target Linux host, with the pinned
submodule initialized before building `microduck`:

```bash
docker compose build host
docker compose build xlerobot-software
docker compose build microduck-software
docker compose --profile hardware build xlerobot
docker compose --profile gpu build microduck
```

After building `xlerobot-software`, its first software-only XLeRobot run uses the bind
mounted `examples/local` directory without installing anything on the host
beyond Docker:

```bash
mkdir -p examples/local
docker compose run --rm --user "$(id -u):$(id -g)" xlerobot-software init xlerobot --output /workspace/xlerobot
docker compose run --rm --user "$(id -u):$(id -g)" xlerobot-software /workspace/xlerobot/example.local.yaml validate
docker compose run --rm --user "$(id -u):$(id -g)" xlerobot-software /workspace/xlerobot/example.local.yaml plan
docker compose run --rm --user "$(id -u):$(id -g)" xlerobot-software /workspace/xlerobot/example.local.yaml check --mode software --json
docker compose run --rm --user "$(id -u):$(id -g)" xlerobot-software /workspace/xlerobot/example.local.yaml dry-run
```

`/workspace/xlerobot` inside the container is
`examples/local/xlerobot` in this checkout. The same host user owns the new
files and can edit them. The image already has the software environment, so
skip native `setup`. This dry-run uses fixtures and starts no robot owner,
inference service, or RPent. It needs no GPU runtime, devices, SDK, calibration,
or checkpoint. XLeRobot software/hardware images and native setup use the same
recipe project and lock; generic `host` remains on the root Host dependency tree.

MicroDuck's software image follows the same offline configuration flow, without
GPU runtime, simulator packages, assets, or checkpoints:

```bash
mkdir -p examples/local
docker compose run --rm --user "$(id -u):$(id -g)" microduck-software init microduck --output /workspace/microduck
docker compose run --rm --user "$(id -u):$(id -g)" microduck-software /workspace/microduck/example.local.yaml validate
docker compose run --rm --user "$(id -u):$(id -g)" microduck-software /workspace/microduck/example.local.yaml plan
docker compose run --rm --user "$(id -u):$(id -g)" microduck-software /workspace/microduck/example.local.yaml check --mode software --json
```

The image already installed the selected profile during its build; skip native
`setup` inside the container. The full `microduck` image adds the same lock's
`simulation` extra and the pinned inference source. To prepare simulation,
provide the external asset root through `MICRODUCK_ASSETS` and initialize a
separate recipe with `--assets /assets`; see the [Recipe's Docker steps](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/microduck_vln/README.md#docker).
The asset mount stays read-only. A successful software check does not verify
CUDA/EGL, external asset compatibility, model inference, or an episode rollout.

When reusing a native manifest in Docker, back it up first. Its top-level
`python` overrides the image's `EMBODIRUN_SCENE_PYTHON`, so a host interpreter
path will not work in the container. Remove only `python` from the Docker copy
to select `/opt/venv/bin/python`, preserving other settings and calibration.
Keep the copy beside the original as, for example, `example.docker.yaml`, so
relative deployment/task references remain valid. In the Docker commands above,
use `/workspace/xlerobot/example.docker.yaml` as the manifest argument.
Use the native backup to return to native execution. Another option is to
initialize a separate Docker directory and migrate your deployment, hardware,
task, and routes with their path references, keeping the originals.

The hardware profile requires explicit device mappings and read-only
calibration mounts in a local Compose override. Its locked SDK is installed
under `/opt/venv/lib/python3.12/site-packages`; set `sdk_src` to that directory,
or explicitly mount and verify a custom SDK. Model inference stays separate.
The GPU profile uses a configured
NVIDIA container runtime (`runtime: nvidia`, required by Thor's NVIDIA CSV mode)
and external assets. The Compose defaults only show
help. The five services share a Compose-project named volume at `/tmp` for
private launcher Unix sockets. A later `docker compose run` can use Recipe
`down` when it uses the same Compose project, host UID/GID, and configuration
as the active launcher. Use the same checkout or the same explicit `-p NAME`
for these commands. The volume is launcher IPC storage, not a dependency cache;
private directory/socket permissions and cleanup confirmation still apply.
You can also stop the original foreground container with Ctrl-C or
`docker stop --time 300 CONTAINER`, giving owned children time to clean up.
Then confirm physical stop feedback; launcher cleanup alone does not prove it.
ARM64 build and runtime support must be verified on each target and optional
integration; an image definition alone does not establish it.

## PyPI

`embodirun` is not published on PyPI yet. Use the source installation above.

## Verify the checkout

```bash
uv run pytest -q
```

The CPU suite passes without a GPU, checkpoints, sglang, or robot hardware.
Tests that need those are skipped with an explicit reason.
