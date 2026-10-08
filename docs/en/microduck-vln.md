# MicroDuck VLN in MuJoCo

The optional MicroDuck example connects EmbodiRun's session-oriented HTTP client
to ActiveVLN inference, executes decoded R2R actions through MPC and an ONNX
walking policy, and records first/third-person videos plus episode metrics.

[Watch the navigation demo](demos/microduck-vln.md) for the launch sequence,
camera views, and execution log.

The complete instructions live in the
[example guide](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/microduck_vln/README.md).
It documents environment setup, the external asset layout, checksum provisioning,
local GPU and Slurm modes, custom episode files, output schemas and troubleshooting.

## Run the example

Install the small locked environment first and prepare the linked configuration
from the repository root:

```bash
RECIPE_ENV="$PWD/.venv-microduck"
UV_PROJECT_ENVIRONMENT="$RECIPE_ENV" uv sync \
  --project examples/microduck_vln --frozen --python 3.12 --no-default-groups
"$RECIPE_ENV/bin/embodirun" example init microduck --assets /absolute/asset/root
CONFIG=examples/local/microduck/example.local.yaml
"$RECIPE_ENV/bin/embodirun" example "$CONFIG" validate
"$RECIPE_ENV/bin/embodirun" example "$CONFIG" plan
"$RECIPE_ENV/bin/embodirun" example "$CONFIG" check --json
```

The base profile supports configuration and help without Torch, MuJoCo or model
assets. Missing external paths are reported by `check --json`. To run the full
example, supply the assets and use a Linux GPU target. Run the following from
the checkout root, including in a new terminal. The assignments use the default
environment and manifest; for custom paths, replace them with the same absolute
environment directory and manifest used during preparation. Select a custom
scene environment with `EMBODIRUN_SCENE_PYTHON` or the manifest's `python` field;
an explicit YAML `python` takes precedence, so keep it consistent with that
environment:

```bash
RECIPE_ENV="$PWD/.venv-microduck"
CONFIG="$PWD/examples/local/microduck/example.local.yaml"
git submodule update --init third_party/embodiinfer
UV_PROJECT_ENVIRONMENT="$RECIPE_ENV" uv sync \
  --project examples/microduck_vln --frozen --python 3.12 --no-default-groups \
  --extra simulation
"$RECIPE_ENV/bin/embodirun" example "$CONFIG" check
"$RECIPE_ENV/bin/embodirun" example "$CONFIG" run
```

`validate` and `plan` are offline. `check --json` reports local path prerequisites
and leaves CUDA/EGL and scene/model readiness unverified. `check` without
`--json` continues into the full asset/CUDA/EGL/ONNX/MPC/video-encoder preflight
on the Linux GPU target after the local prerequisites pass. A passed preflight
does not prove learned-model inference or navigation success; `run` starts those
simulation episodes and its owned inference child.

The direct install uses `examples/microduck_vln/pyproject.toml` and `uv.lock` for
both profiles. The existing `setup` and Docker target still use the earlier
installer with integration version ranges; they have not been connected to
this lock. The CLI still has no MicroDuck `--mode software` option. Fresh target
installation, native/Docker parity and GPU/model validation remain pending.
Keep this Transformers 4.51.3 environment separate from Transformers 5.x model profiles.
See [independent first use](first-use.md) for the actual-execution record.

The Recipe prints `Outputs:`; under its run directory, inspect `run.json`,
command logs and `result/` files, including `preflight.json`, `run.log` and
`inference.log`. `run` owns cleanup on exit. To request shutdown of this Recipe's
foreground launcher from another terminal, enter the same checkout root and
set the paths again. These are the defaults; if you used a custom environment
or manifest, substitute their original absolute paths:

```bash
RECIPE_ENV="$PWD/.venv-microduck"
CONFIG="$PWD/examples/local/microduck/example.local.yaml"
"$RECIPE_ENV/bin/embodirun" example "$CONFIG" down
```

Slurm submission and `scancel` are described in the source guide.

## Reference scenario and locked profile

MicroDuck is an experimental integration with a separate simulation environment.
The hardware and algorithm settings describe the earlier A800 scenario. Package
rows show the Recipe lock's resolved `simulation` candidates; this full profile
has not yet been installed or validated on the target GPU. The integration's
[package metadata](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/integrations/microduck_vln/pyproject.toml)
retains its broader version ranges.

| Setting | Reference configuration |
| --- | --- |
| Hardware / runtime | Linux, one A800 80 GB GPU, four CPU cores, EGL; Python 3.12 |
| Policy | Qwen2.5-VL-3B ActiveVLN, externally supplied merged SFT-v3 checkpoint |
| Locked model candidates | Linux `torch==2.10.0+cu130`, `torchvision==0.25.0+cu130`, `transformers==4.51.3`, `tokenizers==0.21.4` |
| Locked simulation candidates | `numpy==1.26.4`, `Pillow==12.3.0`, `mujoco==3.8.1`, `onnxruntime==1.30.0`, `casadi==3.7.2` |
| Inference source | EmbodiInfer revision pinned by `third_party/embodiinfer` |
| Execution | B=1, eager, bfloat16, SDPA, greedy decoding; recurrent session per episode |
| Scene / limits | `val_2`, default spawn `(6.5, 13.8, 0)`, at most 60 primitive actions |
| Success | STOP and the last three action endpoints strictly inside a 1.0 m radius |

The Torch/torchvision pair follows the pinned inference project's development
profile, whose constraints do not automatically apply to a separately installed
integration. Lock resolution and advertised ARM64 wheels establish package
candidates; the target's driver, glibc and CUDA/EGL still need validation.

## Read the results

Use the example guide's 40-episode command for an evaluation. Each run writes
`results.json`: `status=complete` means the run finished, while the success
metric uses the STOP and distance criteria above. `spl_euclidean` uses
straight-line distance rather than a geodesic path length.
