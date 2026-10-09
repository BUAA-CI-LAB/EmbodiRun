# MicroDuck VLN in MuJoCo

Run ActiveVLN instructions against a simulated MicroDuck and save complete
first/third-person videos and per-episode JSON. One launcher owns the simulation
loop and a separate inference process on the same GPU node, optionally inside
Slurm. This is an optional example using EmbodiRun's existing HTTP client; it
does not require or register a Host/Control device.

```text
examples/run.sh → run_demo.py
  ├─ embodirun.model_services → authenticated loopback HTTP → EmbodiInfer
  │                                                          └─ ActiveVLN / EngineCore
  └─ decoded R2R actions → MPC → walking ONNX → MicroDuck / MuJoCo → EGL / MP4
```

The simulator and episode logic stay in EmbodiRun. Prompt construction, model
loading, text parsing and recurrent memory stay in the inference process.
Each episode opens a fresh session and closes it in `finally`. Startup and
requests have deadlines; shutdown terminates only the child owned by this run.
The service binds an OS-assigned loopback port with a per-run credential.

## Prerequisites

- Offline configuration checks need Linux, Bash, Python 3.12 and uv 0.12.x.
  The software profile needs no GPU, simulator, inference source, assets or checkpoint.
- Simulation additionally needs a CUDA GPU and working NVIDIA/EGL drivers. The previous simulation
  profile used one A800 80 GB GPU and four CPU cores; no physical robot is involved.
- A dedicated Python 3.12 environment for the native Recipe; the Docker images
  include their own environment.
- For simulation, EmbodiInfer at this repository's pinned `third_party/embodiinfer` submodule revision.
- For simulation, a separately provisioned MicroDuck asset project and SFT-v3 merged checkpoint.
  Assets, checkpoints, datasets and recorded media are not included here.

Before simulation or building the full image, initialize the public inference submodule from the repository root:

```bash
git submodule update --init third_party/embodiinfer
```

The upstream distribution is named **embodiinfer**, but its Python namespace
is still **embodiinfer**, and its versioned wire schemas remain `embodiinfer.policy.*.v1`.
Keep those identifiers when using the service. The optional integration package
here is `embodirun-microduck`, importing as `embodirun_microduck`.

## Prepare external assets

Set `parameters.project_root` in the example YAML to the directory containing these paths:

| Path | Contents |
| --- | --- |
| `src/sim/vln_mujoco/` | Existing backend and MPC, plus `assets/scenes/procthor-10k-val/val_2_ceiling.xml` and all referenced meshes/textures |
| `src/robots_assets/mjcf_assets/robot_allcollisions.xml` | MicroDuck MJCF and its referenced assets |
| `src/robots_assets/alpha_walking.onnx` | Walking policy |
| `lightnav_sft_v3/merged/` | Qwen2.5-VL-3B SFT-v3 config, processor/tokenizer, shard index and all weight shards (about 7.1 GB) |
| `data/demo_microduck_vln.jsonl` | Two externally supplied fixed-spawn demonstration episodes |
| `data/train/eval_val2_40_valid.jsonl` | External 40-episode evaluation file |
| `src/val_2.json`, `src/rlinf_integration_lightnav_env.py` | Additional files in the reference asset inventory; the training adapter is only hashed |

The reference scene is `val_2`; the two demonstration episodes start at
`(6.5, 13.8, 0)` and target bins in the living room and bathroom. Custom episodes
must use coordinates in the same scene. Obtain external inputs and their usage
terms from their maintainers; this recipe does not download or redistribute them.

[`assets.md5.json`](assets.md5.json) pins relative paths, sizes and MD5 hashes for
the reference assets and the public inference source. MD5 detects accidental
changes, not authenticity. Model and simulator assets stay in place; launch
creates no copies and never modifies the training adapter.

## Install and configure

The Recipe has its own [`pyproject.toml`](pyproject.toml) and
[`uv.lock`](uv.lock), using Python 3.12 and uv 0.12.x. Native `setup` and Docker
use this same lock. From the EmbodiRun root, prepare the Host CLI, create linked
local files and install the software profile:

```bash
uv sync --frozen --python 3.12
uv run --frozen embodirun example init microduck
CONFIG=examples/local/microduck/example.local.yaml
uv run --frozen embodirun example "$CONFIG" validate
uv run --frozen embodirun example "$CONFIG" plan
uv run --frozen embodirun example "$CONFIG" setup --mode software
uv run --frozen embodirun example "$CONFIG" check --mode software --json
```

The Host CLI uses the root lock; `setup` installs the separate MicroDuck environment
at `.venv-microduck`. Software mode installs Host's CLI dependencies and the base
MicroDuck integration, without PyTorch or MuJoCo. `check --mode software`, with or
without `--json`, checks Python 3.12, package presence and configuration, and
reports installed package versions. Version selection comes from frozen setup;
this metadata check does not compare every package against `uv.lock`.
It does not start a launcher, simulator or inference
process. Scene inputs and their compatibility remain in `unverified`; missing
scene paths do not fail this software check. This flow does not run a simulation rehearsal.

The software flow was verified on Ubuntu 24.04 aarch64: native Python 3.12.3
and container Python 3.12.14 installed the same 14 locked packages, and
`embodirun` / `run.sh` returned matching software reports. Both use Python 3.12;
their patch versions differ. Native verification used freshly downloaded wheel
assistance after the first uncached online setup reached an external 180-second
test limit (exit 130). The container flow passed with the production Dockerfile
on a warm retry (exit 0), after its new-builder, empty-cache build reached an
external 600-second test limit (exit 130). These results do not establish fast
uncached installation or independent first use.

The separate Recipe software workflow runs for relevant source, dependency or
container changes, and supports manual dispatch. It checks the software scope with the production native installer
and Dockerfile. It runs `init`, `validate`, `plan` and software checks through
both entrypoints, copies printed commands outside the checkout without inherited
Recipe variables, checks caller-owned files and compares locked package inventories.
To repeat the check after software setup, select the installed Recipe Python:

```bash
"$RECIPE_ENV/bin/python" scripts/check_recipe_software.py --recipe microduck
```

This checks software onboarding. Full simulation/model execution, native LeRobot
deployment comparisons and independent human first use remain follow-up work.

`init` copies the reference inventory; `--assets /absolute/asset/root` fills
project, checkpoint and episode paths but does not check that those external
inputs exist. Without that option, edit the generated placeholders before
simulation. If the local recipe directory already exists, reuse it or choose
a new directory with `init --output`; existing edits are preserved.

From the checkout root on the Linux GPU host, add the full simulator and
inference dependencies from the same lock before running the scene. Set the
manifest path again in a new terminal; if you chose a custom directory,
replace it with the original path:

```bash
CONFIG=examples/local/microduck/example.local.yaml
uv run --frozen embodirun example "$CONFIG" setup --mode simulation
```

MicroDuck `setup` and `check` default to `simulation` when mode is omitted;
choose `software` explicitly for offline checks. Both setup modes call
`uv sync --frozen --no-default-groups` on this Recipe project, with
`--extra simulation` only for the full profile. The `simulation` extra includes
`embodirun-microduck[full]`. Its resolved
profile includes NumPy 1.26.4, Pillow 12.3.0, MuJoCo 3.8.1, ONNX Runtime 1.30.0,
CasADi 3.7.2 and Transformers 4.51.3. On Linux, PyTorch 2.10.0 and torchvision
0.25.0 use the official CUDA 13.0 wheel index. Those two versions align with
the pinned EmbodiInfer development/runtime lock; they are a candidate profile
for this Recipe, not proof of GPU compatibility or a restriction on every
EmbodiInfer backend. A fresh native full-profile installation did not finish
within an external 600-second validation limit (interrupted, exit 130);
package imports and CUDA/EGL probes were not run. The full Docker target uses
this lock, but its build was not executed in this validation round. Full
target-host installation, scene and model execution remain unverified.

The launcher defaults to the repository-root `.venv-microduck/bin/python`.
If you install into another directory, set `EMBODIRUN_SCENE_PYTHON` to its
absolute `bin/python` path, or set `python` in your local example YAML. An
explicit YAML `python` takes precedence; keep it consistent with the selected
environment. The commands below use the Recipe's locked CLI environment. Edit
`example.local.yaml` if your asset layout differs. The inference source
defaults to the pinned submodule; use `parameters.inference_root` for another
compatible checkout. It is added only to the inference child's import path.

The model profile requires `transformers==4.51.3`; keep it separate from
Transformers 5.x and native LeRobot profiles. Provision packages and weights
before running on an offline cluster. The sync commands install packages, but
do not download weights or external assets.

For `examples/run.sh`, select the already-installed Host CLI with
`EMBODIRUN_EXAMPLE_PYTHON="$PWD/.venv/bin/python"`; it accepts the same manifest,
command and mode as `embodirun example`. For example:

```bash
CONFIG=examples/local/microduck/example.local.yaml
EMBODIRUN_EXAMPLE_PYTHON="$PWD/.venv/bin/python" \
  examples/run.sh "$CONFIG" check --mode software --json
```

Independent first use and full inference remain separate acceptance checks;
neither dependency installation nor a Docker build proves them.

## Docker

Docker Engine, Compose and Buildx with BuildKit are required; check
`docker buildx version` and `docker compose version` before building.

Build and use the software image on the target Linux host. It installs the same
software lock during the image build; no native Recipe environment is needed:

```bash
docker compose build microduck-software
mkdir -p examples/local
docker compose run --rm --user "$(id -u):$(id -g)" microduck-software init microduck --output /workspace/microduck
docker compose run --rm --user "$(id -u):$(id -g)" microduck-software /workspace/microduck/example.local.yaml validate
docker compose run --rm --user "$(id -u):$(id -g)" microduck-software /workspace/microduck/example.local.yaml plan
docker compose run --rm --user "$(id -u):$(id -g)" microduck-software /workspace/microduck/example.local.yaml check --mode software --json
```

`examples/local/microduck` stays editable by your host UID/GID. Skip `setup`
inside the image: dependencies already live at `/opt/venv`. To use the shell
entrypoint for the same check, replace the image entrypoint explicitly:

```bash
docker compose run --rm --user "$(id -u):$(id -g)" --entrypoint /bin/bash microduck-software /opt/embodirun/examples/run.sh /workspace/microduck/example.local.yaml check --mode software --json
```

The full image adds `--extra simulation` from the same lock and the pinned
EmbodiInfer source. After initializing that submodule and provisioning the
external assets, configure the NVIDIA runtime and prepare a separate local
simulation recipe:

```bash
git submodule update --init third_party/embodiinfer
docker compose --profile gpu build microduck
export MICRODUCK_ASSETS=/absolute/asset/root
mkdir -p examples/local
docker compose --profile gpu run --rm --user "$(id -u):$(id -g)" microduck init microduck --assets /assets --output /workspace/microduck-simulation
docker compose --profile gpu run --rm --user "$(id -u):$(id -g)" microduck /workspace/microduck-simulation/example.local.yaml check --mode simulation --json
```

Set `MICRODUCK_ASSETS` again in a new terminal. Assets mount at `/assets:ro`;
they are not copied into the image. For Docker simulation, use the same Compose
service and container manifest path with `check --mode simulation` or `run`;
the next section's commands use a native environment. To reuse a native manifest, preserve the
original and remove only its top-level `python` from a Docker copy beside it;
an explicit host Python path would override the image's `/opt/venv/bin/python`.
See [installation](../../docs/en/installation.md#containers) for shared
`/tmp` launcher IPC storage and owned shutdown. Default Compose commands show
help. Software checks do not establish CUDA/EGL support, asset compatibility,
model inference or episode success.

The container software flow was also exercised as a non-root host UID/GID.
Its generated files stayed in `/workspace` with that ownership, both entrypoints
passed plain and JSON checks, and the printed check command worked from another
directory with a manifest path containing spaces. An unavailable environment
returned exit 2 with a repair hint. The task's containers, launcher volume and
network were removed after verification; this was software evidence only.

## Run

Run from the checkout root on the Linux GPU host. The assignments below use
the default environment and manifest; if you chose custom paths, substitute
their original absolute paths and select the scene interpreter as described
above. Set these variables again in a new terminal:

```bash
RECIPE_ENV="$PWD/.venv-microduck"
CONFIG="$PWD/examples/local/microduck/example.local.yaml"
"$RECIPE_ENV/bin/embodirun" example "$CONFIG" validate
"$RECIPE_ENV/bin/embodirun" example "$CONFIG" plan
"$RECIPE_ENV/bin/embodirun" example "$CONFIG" check --mode simulation --json
"$RECIPE_ENV/bin/embodirun" example "$CONFIG" check --mode simulation
"$RECIPE_ENV/bin/embodirun" example "$CONFIG" run
```

`validate` and `plan` are offline. `check --mode simulation --json` reports local
environment and path
prerequisites; it does not check CUDA/EGL or model inference. `check` without
`--json` continues into the asset/CUDA/EGL/ONNX/MPC/encoder preflight after the
local checks pass. The full preflight exercises the scene and walking policy
but does not run the learned VLN model. `run` starts simulation and its
inference child; inspect the episode results for task success separately.
All settings come from the YAML. Outputs are under
`examples/local/microduck/runs/<timestamp>-run/result/`.

Set `parameters.slurm: "off"` on a GPU host, or `auto` to request one GPU
outside an existing allocation. `cpus`, `time`, and optional `partition`
configure that request. To keep a run alive after SSH disconnects, set
`slurm: "off"` in the YAML and submit the same entrypoint:

```bash
sbatch --gres=gpu:1 --cpus-per-task=4 --time=01:00:00 \
  --wrap="\"$RECIPE_ENV/bin/embodirun\" example \"$CONFIG\" run"
```

Add your cluster's partition option if needed. Run `sbatch` from the repository
root with the absolute `RECIPE_ENV` and `CONFIG` defined above. The
wrapped command uses absolute CLI and manifest paths. Cancel with
`scancel JOB_ID`; termination saves available diagnostics and stops the
inference child.

## Episodes and asset integrity

Set `parameters.episodes` to the episode JSONL file and `num_episodes` to the
number to run. `episode_offset`, `max_steps`, `seed`, and `fps` control the
selection, action cap, deterministic seed, and recording rate. Decoding is greedy.

An episode record has this shape (positions in metres, yaw in radians):

```json
{"id":"example","instruction":"Walk to the bin.","start":[6.5,13.8,0.0],"goal":[8.517,9.358],"max_steps":60}
```

The smaller of the manifest and episode action caps applies. For the reference
evaluation, select `data/train/eval_val2_40_valid.jsonl` and set
`num_episodes: 40`. Episode IDs are metadata, not filenames.

When deliberately changing an asset, checkpoint, or inference source, first
check compatibility. `init` already created `assets.local.json` from the
reference inventory, so retain it for the reference inputs. For a changed
asset set, point `parameters.manifest` at a **new, absent**
`assets.revised.local.json`, then provision its inventory:

```bash
"$RECIPE_ENV/bin/embodirun" example "$CONFIG" provision
"$RECIPE_ENV/bin/embodirun" example "$CONFIG" check
```

Provisioning does not allocate a GPU and refuses to overwrite an existing
inventory. Subsequent runs verify the selected inventory; they do not repair it.

## Execution, metrics and outputs

The decoded chunk is `[3, 2]`: action ID and magnitude. Forward supports
25/50/75 cm; left/right support 15/30/45 degrees. Padding `-1` is skipped. STOP
`0` ends the chunk and episode, discarding later rows. Invalid output ends that
episode without executing the invalid chunk. The 60-step limit counts primitive
actions including STOP, not inference turns.

The existing CasADi MPC runs at 10 Hz. Commands become body-frame endpoints
transformed once into world coordinates; speed is capped at 0.30 m/s and yaw
rate at 1.5 rad/s. Each primitive is bounded to 1,200 physics steps (6 seconds),
with up to 150 extra turn-braking steps. Actual endpoint errors and physical path
length are recorded; the robot is not teleported to predicted targets.

Success requires model STOP and the last three action endpoints strictly within
1.0 m of the goal. `sr` is the successful-episode fraction; `spl_euclidean` uses
straight-line start-to-goal distance and is **not standard geodesic SPL**.

Every invocation has a fresh output directory. The top level contains the
shared launcher's `run.json` and command log; `result/` contains:

| File | Contents |
| --- | --- |
| `results.json`, `episode_NNN.json` | Status, configuration, fingerprints, model text, session/request IDs, actions, timing, distances and termination |
| `episode_NNN.mp4` | 960×400 H.264, two views, instruction/action/distance HUD and actual final outcome |
| `run.log`, `inference.log` | Deployment and model diagnostics |
| `preflight.json`, `preflight_*.jpg/mp4` | Environment, rendering and encoder checks |

Videos follow simulation time, omitting inference pauses, with title/outcome
cards. Encoding streams frames without image dumps and has a 90 MB per-video
guard. HTTP timing starts after local PNG encoding. Technical failures exit
nonzero and preserve diagnostics; inspect `success`/`termination` separately.

## Troubleshooting and tests

For inventory failures, check paths and intended versions before generating a
replacement inventory. For CUDA/EGL failures, check GPU allocation and drivers.
For service startup failures or timeouts, read `inference.log` and verify the
checkpoint, source revision and 4.51.3 environment. ORT uses one intra/inter-op
thread to respect Slurm CPU affinity. Turn undershoot and max-step outcomes are
reported behavior; inspect endpoint errors when experimenting with the controller.

From the repository root, the normal development environment runs the CPU tests:

```bash
uv run --frozen pytest -q tests/test_microduck_vln.py
uv run --frozen ruff check src tests examples/microduck_vln integrations/microduck_vln
uv run --frozen ruff format --check src tests examples/microduck_vln integrations/microduck_vln
bash -n examples/run.sh
```

The HTTP adapter tests skip when Torch/EmbodiInfer are absent. To exercise them
with the dedicated simulation environment, without a GPU or checkpoint, also
prepare pytest as a development tool in that environment; see the integration's
[`test` extra](../../integrations/microduck_vln/README.md). The Recipe's runtime
lock does not include pytest. That extra's installation has not been verified
as a Recipe deployment step; the commands below are developer checks, not an
additional locked installation profile:

```bash
PYTHONPATH="src:third_party/embodiinfer" "$RECIPE_ENV/bin/python" \
  -m pytest -q tests/test_microduck_vln.py tests/test_embodiinfer_http.py
```

CPU checks cover action bounds, STOP, success conditions, failure cleanup,
checksums, subprocess deadlines and, when dependencies are present, the real
HTTP server/client with a fake model core. The current support status and
validation scope are documented in [the integration guide](../../docs/en/microduck-vln.md).
This contribution uses the repository's Apache-2.0 license; external assets and
SDK distributions retain their own terms.
