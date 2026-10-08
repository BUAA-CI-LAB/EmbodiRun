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

- Linux, Bash, a CUDA GPU and working NVIDIA/EGL drivers. The previous simulation
  profile used one A800 80 GB GPU and four CPU cores; no physical robot is involved.
- A dedicated Python environment. Python 3.12 is the reference profile; optional
  package metadata supports Python 3.10+ where its dependencies are available.
- EmbodiInfer at this repository's pinned `third_party/embodiinfer` submodule revision.
- A separately provisioned MicroDuck asset project and SFT-v3 merged checkpoint.
  Assets, checkpoints, datasets and recorded media are not included here.

Initialize the public inference submodule from the repository root:

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

From the EmbodiRun root, create a linked local recipe with your absolute
external asset root, then install its dedicated environment:

```bash
uv sync --frozen
uv run --frozen embodirun example init microduck --assets /absolute/asset/root
CONFIG=examples/local/microduck/example.local.yaml
uv run --frozen embodirun example "$CONFIG" setup
```

`init` copies the reference inventory and fills project, checkpoint and episode
paths. `setup` installs Host from `uv.lock` and the optional integration from
its declared version ranges in `.venv-microduck`; that integration is not fully
locked by the Host lock. Edit `example.local.yaml` if your asset layout differs.
The inference source defaults to the pinned submodule; use
`parameters.inference_root` for another compatible checkout. It is added only
to the inference child's import path.

The model profile requires `transformers==4.51.3`; keep it separate from
Transformers 5.x profiles. Provision packages and weights before running on an
offline cluster. `setup` installs packages, but does not download weights or
external assets.

The current Recipe has no `--mode software` option. A minimal software profile,
an isolated full simulation profile, and one complete lock shared by native and
Docker installation remain reproducibility work. The current Docker target
uses the existing installer with version ranges; building it is not evidence
of a matching fresh native install or working GPU/model. Fresh target-host
installation, independent first use and full inference remain separate checks.

## Run

On the Linux GPU host:

```bash
CONFIG=examples/local/microduck/example.local.yaml
uv run --frozen embodirun example "$CONFIG" validate
uv run --frozen embodirun example "$CONFIG" plan
uv run --frozen embodirun example "$CONFIG" check --json
uv run --frozen embodirun example "$CONFIG" check
uv run --frozen embodirun example "$CONFIG" run
```

`validate` and `plan` are offline. `check --json` only reports local path
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
  --wrap='uv run --frozen embodirun example examples/local/microduck/example.local.yaml run'
```

Add your cluster's partition option if needed. Run `sbatch` from the repository
root. Cancel with `scancel JOB_ID`; termination saves available diagnostics and
stops the inference child.

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
uv run --frozen embodirun example "$CONFIG" provision
uv run --frozen embodirun example "$CONFIG" check
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
uv run pytest -q tests/test_microduck_vln.py
uv run ruff check src tests examples/microduck_vln integrations/microduck_vln
uv run ruff format --check src tests examples/microduck_vln integrations/microduck_vln
bash -n examples/run.sh
```

The HTTP adapter tests skip when Torch/EmbodiInfer are absent. To exercise them
with the dedicated optional environment, without a GPU or checkpoint:

```bash
PYTHONPATH="src:third_party/embodiinfer" .venv-microduck/bin/python \
  -m pytest -q tests/test_microduck_vln.py tests/test_embodiinfer_http.py
```

CPU checks cover action bounds, STOP, success conditions, failure cleanup,
checksums, subprocess deadlines and, when dependencies are present, the real
HTTP server/client with a fake model core. The current support status and
validation scope are documented in [the integration guide](../../docs/en/microduck-vln.md).
This contribution uses the repository's Apache-2.0 license; external assets and
SDK distributions retain their own terms.
