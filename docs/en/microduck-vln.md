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

After preparing the dedicated optional environment and external assets:

```bash
git submodule update --init third_party/embodiinfer
uv sync --frozen
uv run --frozen embodirun example init microduck --assets /absolute/asset/root
CONFIG=examples/local/microduck/example.local.yaml
uv run --frozen embodirun example "$CONFIG" validate
uv run --frozen embodirun example "$CONFIG" plan
uv run --frozen embodirun example "$CONFIG" setup
uv run --frozen embodirun example "$CONFIG" check --json
uv run --frozen embodirun example "$CONFIG" check
uv run --frozen embodirun example "$CONFIG" run
```

`validate` and `plan` are offline. `check --json` reports local path prerequisites
and leaves CUDA/EGL and scene/model readiness unverified. `check` without
`--json` continues into the full asset/CUDA/EGL/ONNX/MPC/video-encoder preflight
on the Linux GPU target after the local prerequisites pass. A passed preflight
does not prove learned-model inference or navigation success; `run` starts those
simulation episodes and its owned inference child.

Current `setup` installs Host from the root lock and the full optional MicroDuck
integration from declared version ranges. The latter is not fully locked, and
the current CLI has no MicroDuck software-only mode. Native/Docker full-lock
parity and an isolated software profile remain dependency work, with fresh
target-host installation and GPU/model validation still required. Keep this
Transformers 4.51.3 environment separate from Transformers 5.x model profiles.
See [independent first use](first-use.md) for the actual-execution record.

The Recipe prints `Outputs:`; under its run directory, inspect `run.json`,
command logs and `result/` files, including `preflight.json`, `run.log` and
`inference.log`. `run` owns cleanup on exit. From another terminal with the same
checkout/configuration, `uv run --frozen embodirun example "$CONFIG" down`
requests shutdown of this Recipe's foreground launcher. Slurm submission and
`scancel` are described in the source guide.

## Reference scenario and declared dependencies

MicroDuck is an experimental integration with a separate simulation environment.
The hardware and algorithm settings describe the earlier A800 scenario. Package
rows show the current integration's declarations, not a resolved environment
or newly validated installation. See the
[package metadata](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/integrations/microduck_vln/pyproject.toml).

| Setting | Reference configuration |
| --- | --- |
| Hardware / runtime | Linux, one A800 80 GB GPU, four CPU cores, EGL; Python 3.12 |
| Policy | Qwen2.5-VL-3B ActiveVLN, externally supplied merged SFT-v3 checkpoint |
| Declared model packages, not fully locked | `torch>=2.5,<2.12`, `torchvision>=0.20,<0.27`, `transformers==4.51.3`, `tokenizers==0.21.4` |
| Declared simulation packages, not fully locked | `mujoco==3.8.1`, `onnxruntime>=1.20,<2`, `casadi==3.7.2` |
| Inference source | EmbodiInfer revision pinned by `third_party/embodiinfer` |
| Execution | B=1, eager, bfloat16, SDPA, greedy decoding; recurrent session per episode |
| Scene / limits | `val_2`, default spawn `(6.5, 13.8, 0)`, at most 60 primitive actions |
| Success | STOP and the last three action endpoints strictly inside a 1.0 m radius |

These ranges do not automatically combine with the pinned inference project's
own uv lock/profile constraints. A dedicated MicroDuck lock and fresh
target-host validation are still required; none of the listed ranges establishes
a tested CUDA wheel, native/Docker match, or resolved full environment.

## Read the results

Use the example guide's 40-episode command for an evaluation. Each run writes
`results.json`: `status=complete` means the run finished, while the success
metric uses the STOP and distance criteria above. `spl_euclidean` uses
straight-line distance rather than a geodesic path length.
