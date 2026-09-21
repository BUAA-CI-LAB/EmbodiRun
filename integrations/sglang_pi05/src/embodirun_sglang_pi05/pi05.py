"""Deploy's opt-in LeRobot compatibility for SGLang's native Pi05 pipeline.

SGLang owns all model execution and HTTP serving. This entry point only separates
model tensors from processor statistics and restores checkpoint normalization.
This module runs only in the SGLang service environment. Host and HTTP clients
must not import it; they do not need Torch or model-processing dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file
from sglang.multimodal_gen.runtime.models.vlas.pi05_policy import Pi05PolicyModel
from sglang.multimodal_gen.runtime.pipelines.pi05 import Pi05Pipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.pi05_preprocess import (
    Pi05Preprocessor,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.vla import (
    VLAActionDenoisingStage,
    VLAActionPostprocessStage,
    VLAObservationPreprocessStage,
    VLAPrefixEncodingStage,
)

from embodirun.model_services.backends.sglang import (
    prepare_sglang_environment,
)


@dataclass(frozen=True)
class _Statistics:
    """Checkpoint normalization statistics for one feature.

    ``mean``/``std`` keep their historical names, but hold the mode's offset and
    scale so that the two supported modes share one code path:

        mode="MEAN_STD"    mean = mean,  std = std
        mode="QUANTILES"   mean = q01,   std = q99 - q01  (``eps`` substituted
                           where the two quantiles coincide)

    Both formulas mirror ``lerobot.processor.normalize_processor`` exactly, so the
    SGLang path stays numerically identical to the LeRobot path.
    """

    mean: torch.Tensor
    std: torch.Tensor
    eps: float
    mode: str = "MEAN_STD"

    SUPPORTED_MODES = ("MEAN_STD", "QUANTILES")

    # Delta-action processors are training/export artefacts. LeRobot emits them
    # with ``enabled: false`` when the checkpoint's actions are already absolute,
    # which makes them pure no-ops. An *enabled* delta processor changes action
    # semantics, and this adapter has nowhere to apply the inverse, so it stays
    # unsupported rather than silently producing wrong commands.
    _OPTIONAL_NOOP_PROCESSORS = ("delta_actions_processor", "absolute_actions_processor")

    @classmethod
    def load(cls, root: Path, pipeline: str, registry: str, feature: str, kind: str) -> _Statistics:
        payload = json.loads((root / f"policy_{pipeline}.json").read_text())
        steps = payload["steps"]
        allowed = {
            registry,
            "rename_observations_processor",
            "to_batch_processor",
            "pi05_prepare_state_tokenizer_processor_step",
            "tokenizer_processor",
            "device_processor",
        }
        for step in steps:
            name = step["registry_name"]
            if name in cls._OPTIONAL_NOOP_PROCESSORS:
                if (step.get("config") or {}).get("enabled") is not False:
                    raise ValueError(f"SGLang LeRobot compatibility requires explicitly disabled {name}")
                continue
            if name not in allowed:
                raise ValueError(f"unsupported LeRobot processor: {name}")
            if name == "rename_observations_processor" and step["config"].get("rename_map"):
                raise ValueError("SGLang LeRobot serving requires already named input features")
        matches = [step for step in steps if step["registry_name"] == registry]
        if len(matches) != 1:
            raise ValueError(f"checkpoint must contain exactly one {registry}")
        step = matches[0]
        config = step["config"]
        mode = config["norm_map"].get(kind)
        if mode not in cls.SUPPORTED_MODES:
            raise ValueError(
                f"SGLang LeRobot compatibility requires {kind} in {list(cls.SUPPORTED_MODES)}, got {mode!r}"
            )
        if pipeline == "preprocessor" and config["norm_map"].get("VISUAL") != "IDENTITY":
            raise ValueError("SGLang LeRobot compatibility requires VISUAL=IDENTITY")
        filename = step["state_file"]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("processor statistics must name a checkpoint file")
        path = root / filename
        stats = load_file(str(path))
        shape = tuple(config["features"][feature]["shape"])
        eps = float(config.get("eps", 1e-8))
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("normalizer eps must be finite and positive")

        if mode == "MEAN_STD":
            mean = stats[f"{feature}.mean"].float()
            std = stats[f"{feature}.std"].float()
            if mean.shape != shape or std.shape != shape or len(shape) != 1:
                raise ValueError(f"invalid normalization dimensions for {feature}")
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std < 0).any():
                raise ValueError(f"invalid normalization statistics for {feature}")
            return cls(mean, std, eps, mode)

        try:
            q01 = stats[f"{feature}.q01"].float()
            q99 = stats[f"{feature}.q99"].float()
        except KeyError as error:
            raise ValueError(
                f"QUANTILES normalization requires {feature}.q01 / {feature}.q99 statistics: {error}"
            ) from error
        if q01.shape != shape or q99.shape != shape or len(shape) != 1:
            raise ValueError(f"invalid normalization dimensions for {feature}")
        if not torch.isfinite(q01).all() or not torch.isfinite(q99).all() or (q99 < q01).any():
            raise ValueError(f"invalid normalization statistics for {feature}")
        # lerobot substitutes eps only where the quantiles coincide.
        width = q99 - q01
        scale = torch.where(width == 0, torch.full_like(width, eps), width)
        return cls(q01, scale, eps, mode)

    def normalize(self, values: Any) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32, device="cpu")
        if tensor.ndim not in (1, 2) or tensor.shape[-1] != self.mean.numel():
            raise ValueError("state dimensions do not match checkpoint statistics")
        if not torch.isfinite(tensor).all():
            raise ValueError("state must contain only finite values")
        if self.mode == "QUANTILES":
            return 2.0 * (tensor - self.mean) / self.std - 1.0
        return (tensor - self.mean) / (self.std + self.eps)

    def denormalize(self, values: Any) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32, device="cpu")
        if tensor.ndim < 1 or tensor.shape[-1] != self.mean.numel() or not torch.isfinite(tensor).all():
            raise ValueError("actions do not match checkpoint normalization statistics")
        if self.mode == "QUANTILES":
            return (tensor + 1.0) * self.std / 2.0 + self.mean
        return tensor * self.std + self.mean


class _LeRobotPolicyModel(Pi05PolicyModel):
    @staticmethod
    def _inspect_checkpoint(model_path: str, config: Any) -> Any:
        manifest = Pi05PolicyModel._inspect_checkpoint(model_path, config)
        root = Path(model_path)
        index = root / "model.safetensors.index.json"
        names = set(json.loads(index.read_text())["weight_map"].values()) if index.is_file() else {"model.safetensors"}
        files = [str(root / name) for name in sorted(names)]
        if any(Path(name).name != name for name in names) or any(not Path(path).is_file() for path in files):
            raise ValueError("checkpoint model safetensors are missing or outside the checkpoint")
        return replace(manifest, safetensor_files=files)


class _LeRobotPreprocessor:
    def __init__(self, processor: Pi05Preprocessor, statistics: _Statistics) -> None:
        self.processor = processor
        self.statistics = statistics

    def __call__(self, raw_observation: dict[str, Any]) -> Any:
        observation = dict(raw_observation)
        observation["state"] = self.statistics.normalize(observation["state"])
        return self.processor(observation)


class _LeRobotPostprocess(VLAActionPostprocessStage):
    def __init__(self, statistics: _Statistics) -> None:
        super().__init__()
        self.statistics = statistics

    def forward(self, batch: Any, server_args: Any) -> Any:
        output = super().forward(batch, server_args)
        for payload in output.output:
            actions = payload["actions"]
            restored = self.statistics.denormalize(actions)
            payload["actions"] = restored.numpy() if isinstance(actions, np.ndarray) else restored.tolist()
        return output


class LeRobotPi05Pipeline(Pi05Pipeline):
    """Native SGLang Pi05 execution with LeRobot MEAN_STD / QUANTILES processors."""

    pipeline_name = "LeRobotPi05Pipeline"

    def load_modules(self, server_args: Any, loaded_modules: Any = None) -> dict[str, Any]:
        """Load model safetensors only; processor tensors are not model parameters."""
        if loaded_modules is not None:
            return loaded_modules
        config = server_args.pipeline_config
        if config.prefix_parallel_strategy == config.action_parallel_strategy == "tp":
            raise ValueError("VLA action expert must not share the prefix TP layout")
        config.offload_prefix_image_encoder |= bool(server_args.image_encoder_cpu_offload)
        config.offload_prefix_token_embedding |= bool(server_args.text_encoder_cpu_offload)
        model = _LeRobotPolicyModel.from_pretrained(self.model_path, config)
        # LeRobot exports may already list the empty cameras in input_features.
        config.image_keys = tuple(dict.fromkeys(config.image_keys))
        return {"policy_model": model}

    def initialize_pipeline(self, server_args: Any) -> None:
        """Read the checkpoint's pre/postprocessor statistics without LeRobot imports."""
        super().initialize_pipeline(server_args)
        root = Path(self.get_module("policy_model").model_path)
        state = _Statistics.load(root, "preprocessor", "normalizer_processor", "observation.state", "STATE")
        self.action_statistics = _Statistics.load(root, "postprocessor", "unnormalizer_processor", "action", "ACTION")
        self.preprocessor = _LeRobotPreprocessor(self.preprocessor, state)

    def create_pipeline_stages(self, server_args: Any) -> None:
        """Retain SGLang's prefix/denoising stages and restore action units last."""
        self.add_stage(VLAObservationPreprocessStage(self.preprocessor), "pi05_preprocess")
        self.add_stage(
            VLAPrefixEncodingStage(self.get_module("policy_model"), self.prefix_cache),
            "pi05_prefix",
        )
        self.add_stage(
            VLAActionDenoisingStage(self.get_module("policy_model")),
            "pi05_action_denoise",
        )
        self.add_stage(_LeRobotPostprocess(self.action_statistics), "pi05_postprocess")


def _register_pipeline() -> None:
    # The console entry point imports this module in spawned workers too.
    from sglang.multimodal_gen import registry

    prepare_sglang_environment()
    registry._discover_and_register_pipelines()
    registry._PIPELINE_REGISTRY[LeRobotPi05Pipeline.pipeline_name] = LeRobotPi05Pipeline
    registry._PIPELINE_CONFIG_REGISTRY[LeRobotPi05Pipeline.pipeline_name] = (
        LeRobotPi05Pipeline.pipeline_config_cls,
        LeRobotPi05Pipeline.sampling_params_cls,
    )


def main() -> None:
    """Run SGLang's CLI with the opt-in LeRobotPi05Pipeline registered."""
    from sglang.cli.main import main as sglang_main

    sglang_main()


_register_pipeline()
