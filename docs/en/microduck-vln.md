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
cp examples/microduck_vln/example.yaml examples/microduck_vln/example.local.yaml
# Edit the YAML with asset paths, Python environment, and resource selection.
bash examples/run.sh examples/microduck_vln/example.local.yaml check
bash examples/run.sh examples/microduck_vln/example.local.yaml run
```

The example runs in simulation and uses locally prepared checkpoints, datasets,
and scene assets. The source guide also shows how to submit it with `sbatch`.

## Reference configuration

MicroDuck is an experimental integration with a separate simulation environment.

| Setting | Reference configuration |
| --- | --- |
| Hardware / runtime | Linux, one A800 80 GB GPU, four CPU cores, EGL; Python 3.12 |
| Policy | Qwen2.5-VL-3B ActiveVLN, externally supplied merged SFT-v3 checkpoint |
| Model environment | Torch 2.11 / torchvision 0.26, Transformers 4.51.3, tokenizers 0.21.4 |
| Simulation environment | MuJoCo 3.8.1, ONNX Runtime 1.30, CasADi 3.7.2 |
| Inference source | EmbodiInfer revision pinned by `third_party/embodiinfer` |
| Execution | B=1, eager, bfloat16, SDPA, greedy decoding; recurrent session per episode |
| Scene / limits | `val_2`, default spawn `(6.5, 13.8, 0)`, at most 60 primitive actions |
| Success | STOP and the last three action endpoints strictly inside a 1.0 m radius |

## Read the results

Use the example guide's 40-episode command for an evaluation. Each run writes
`results.json`: `status=complete` means the run finished, while the success
metric uses the STOP and distance criteria above. `spl_euclidean` uses
straight-line distance rather than a geodesic path length.
