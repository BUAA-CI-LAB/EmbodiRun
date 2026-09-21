# Third-Party Notices

EmbodiRun itself is licensed under the Apache License, Version 2.0 (see
[LICENSE](LICENSE)). EmbodiRun does not vendor model checkpoints, model
frameworks, or robot SDKs. It depends on the third-party packages below, which
remain under their own licenses.

This file lists the licenses of the dependencies that EmbodiRun can install.
Optional robot and simulator SDKs are **not** distributed with EmbodiRun; their
licenses apply when you install them yourself.

## Host and development dependencies

| Package | License | Used by |
|---|---|---|
| [paramiko](https://pypi.org/project/paramiko/) | LGPL-2.1-or-later | SSH orchestration (`host` group) |
| [PyYAML](https://pypi.org/project/PyYAML/) | MIT | Configuration loading |
| [rich](https://pypi.org/project/rich/) | MIT | CLI progress output |
| [pytest](https://pypi.org/project/pytest/) | MIT | Development and tests |

`paramiko` is licensed under the GNU Lesser General Public License v2.1. It is
used as a separate, dynamically imported library and is not modified or
redistributed by EmbodiRun.

## Robot dependencies (optional groups)

| Package | License | Group |
|---|---|---|
| [feetech-servo-sdk](https://pypi.org/project/feetech-servo-sdk/) | Unlicense | `robot-so101` |
| [opencv-python-headless](https://pypi.org/project/opencv-python-headless/) | Apache-2.0 | `robot-so101`, `robot-arx5` |
| [Pillow](https://pypi.org/project/Pillow/) | MIT-CMU | `robot-arx5` |
| [pyrealsense2](https://pypi.org/project/pyrealsense2/) | Apache-2.0 | `robot-arx5` |
| [franky-control](https://pypi.org/project/franky-control/) | MIT | `robot-fr3` |

ARX5 also requires a vendor `bimanual` motor driver that is not distributed
here. Unitree Go2 requires the vendor Unitree SDK2 and is not distributed here.
See the vendor terms before use.

## Simulator dependencies (optional groups)

| Package | License | Group |
|---|---|---|
| [lerobot](https://pypi.org/project/lerobot/) | Apache-2.0 | `sim-libero`, `sim-vlabench` |
| [habitat-sim](https://github.com/facebookresearch/habitat-sim) | MIT | `sim-habitat` |
| [VLABench](https://github.com/OpenMOSS/VLABench) | MIT | `sim-vlabench` |
| [Isaac Sim](https://developer.nvidia.com/isaac-sim) | NVIDIA Isaac Sim EULA (proprietary) | `sim-isaac` |

Isaac Sim is distributed under NVIDIA's own license, not an open-source
license. Review it before installing the `sim-isaac` group.

The separately installed `integrations/microduck_vln` package declares NumPy,
Pillow, MuJoCo, ONNX Runtime, CasADi and imageio-ffmpeg. Its optional service
extra adds PyTorch/torchvision, Transformers, tokenizers, huggingface-hub,
safetensors, einops and accelerate. These distributions are not vendored;
their own license files and third-party notices apply, including those for any
FFmpeg executable supplied by imageio-ffmpeg. The external `vln_mujoco` backend,
scene/robot assets, walking ONNX policy and SFT-v3 weights must be supplied
separately. See the [recipe](examples/microduck_vln/README.md) for the boundary
and the integration's `pyproject.toml` for dependency versions.

## Inference backend dependencies

| Package | License | Notes |
|---|---|---|
| [sglang](https://pypi.org/project/sglang/) | Apache-2.0 | Optional `sglang` extra |
| [EmbodiInfer](https://github.com/BUAA-CI-LAB/EmbodiInfer) | Apache-2.0 | First-party inference engine, pinned under `third_party/embodiinfer` |
| [WirelessComm](https://github.com/BUAA-CI-LAB/WirelessComm) | Apache-2.0 | Optional `wireless` transport, from the `wireless` extra |

`third_party/embodiinfer` is a Git submodule pinned to a specific EmbodiInfer
commit. EmbodiInfer carries its own `LICENSE`, `NOTICE`, and
`THIRD_PARTY_NOTICES.md`; refer to them when redistributing the pinned
inference server.

## Documentation tooling

The documentation site is built with [MkDocs](https://www.mkdocs.org/) and
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/), both MIT
licensed. They are build-time-only dependencies declared in
`docs/requirements.txt` and are not installed with the runtime.

## Models and datasets

Model checkpoints, tokenizers, normalization statistics, and datasets are not
covered by EmbodiRun's license. Each upstream model or dataset keeps its own
license and access conditions. This includes, but is not limited to, π0.5 /
LeRobot checkpoints, LIBERO, Habitat scenes, VLABench assets, StreamVLN, and
OpenVLA-OFT weights. Check the upstream terms before redistribution.

## Adapted code

At the time of this notice, the EmbodiRun source tree does not contain vendored
third-party source files. Code adapted from an upstream project must carry an
attribution comment and an entry in this file.
