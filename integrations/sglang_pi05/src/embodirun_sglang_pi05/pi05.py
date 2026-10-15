"""Deploy's opt-in LeRobot compatibility for SGLang's native Pi05 pipeline.

SGLang owns all model execution and HTTP serving. This entry point only separates
model tensors from processor statistics and restores checkpoint normalization.
This module runs only in the SGLang service environment. Host and HTTP clients
must not import it; they do not need Torch or model-processing dependencies.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
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
    _cuda_graph_enabled,
    synchronize_vla_action_tensor,
    vla_state,
    vla_timings,
)

from embodirun.model_services.backends.sglang import (
    prepare_sglang_environment,
)

_ASYNC_SCHEMA = "embodiinfer.async.v1"
_RTC_ACTIVATION_LOCK = threading.Lock()
_RTC_TELEMETRY_LOCK = threading.Lock()
_RTC_TELEMETRY_REQUESTS = 0
logger = logging.getLogger(__name__)


def _async_block(observation: dict[str, Any]) -> dict[str, Any]:
    """Read and validate the shared async metadata from an action request."""

    metadata = observation.get("metadata")
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise ValueError("observation.metadata must be an object")
    block = metadata.get("async")
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise ValueError("observation.metadata.async must be an object")
    if block.get("schema") != _ASYNC_SCHEMA:
        raise ValueError(f"unsupported async metadata schema {block.get('schema')!r}")
    return block


def _prefix_weights(start: int, end: int, total: int, schedule: str) -> torch.Tensor:
    start = min(start, end)
    if schedule not in {"zeros", "ones", "linear", "exp"}:
        raise ValueError(f"unsupported RTC prefix_attention_schedule {schedule!r}")
    if schedule == "zeros":
        out = torch.zeros(total)
        out[:start] = 1
        return out
    if schedule == "ones":
        out = torch.ones(total)
        out[end:] = 0
        return out
    interior = torch.linspace(1, 0, max(total - max(total - end, 0) - start, 0) + 2)[1:-1]
    if schedule == "exp":
        interior = interior * torch.expm1(interior) / (math.e - 1.0)
    return torch.cat([torch.ones(min(start, total)), interior, torch.zeros(max(total - end, 0))])


def _rtc_strength(action_ness: torch.Tensor, maximum: float) -> torch.Tensor:
    one_minus = (1 - action_ness) ** 2
    inverse_r2 = (one_minus + action_ness**2) / one_minus
    c = torch.nan_to_num((1 - action_ness) / action_ness, posinf=maximum)
    value = torch.nan_to_num(c * inverse_r2, posinf=maximum)
    return torch.minimum(value, torch.as_tensor(maximum, device=value.device, dtype=value.dtype))


class _AutogradSafeGeluAndMul(torch.nn.Module):
    """Use SGLang's pure-Torch GEGLU whenever a VJP is being recorded.

    The CUDA ``GeluAndMul`` path writes into a preallocated output tensor and
    does not expose an autograd formula.  Plain SGLang sampling still delegates
    to that optimized implementation; only soft RTC's grad-enabled action
    expert pass uses ``forward_native``.
    """

    def __init__(self, operation: torch.nn.Module) -> None:
        super().__init__()
        self.operation = operation

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and value.requires_grad:
            return self.operation.forward_native(value)
        return self.operation(value)


def _install_rtc_autograd_activations(model: Any) -> None:
    """Make the Pi0.5 action expert's fused activations VJP-safe once."""

    if bool(getattr(model, "_embodirun_rtc_autograd_activations", False)):
        return
    with _RTC_ACTIVATION_LOCK:
        if bool(getattr(model, "_embodirun_rtc_autograd_activations", False)):
            return
        core = getattr(getattr(model, "action_expert", None), "core_model", None)
        expert = getattr(getattr(core, "paligemma_with_expert", None), "gemma_expert", None)
        expert_model = getattr(expert, "model", None)
        layers = getattr(expert_model, "layers", None)
        if layers is None:
            raise RuntimeError("soft RTC could not locate the Pi0.5 action-expert layers")
        wrapped = 0
        for layer in layers:
            mlp = getattr(layer, "mlp", None)
            activation = getattr(mlp, "act_fn", None)
            if isinstance(activation, _AutogradSafeGeluAndMul):
                wrapped += 1
                continue
            if activation is None or not callable(getattr(activation, "forward_native", None)):
                raise RuntimeError("soft RTC requires an autograd-safe action-expert activation")
            mlp.act_fn = _AutogradSafeGeluAndMul(activation).eval()
            wrapped += 1
        if wrapped == 0:
            raise RuntimeError("soft RTC found no Pi0.5 action-expert activations")
        model._embodirun_rtc_autograd_activations = True


def _grad_enabled_denoise(
    model: Any,
    prefix_context: Any,
    x: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    """Run the upstream Pi0.5 core without its inference-only ``no_grad``.

    SGLang decorates ``Pi05CoreModel.denoise_step`` with ``@torch.no_grad``.
    Calling ``Pi05PolicyModel.denoise_step`` from an outer ``enable_grad``
    therefore produces a detached velocity and silently reduces RTC's VJP to
    the identity.  ``torch.no_grad`` preserves the wrapped implementation, so
    soft RTC can call that exact body while retaining all model semantics.
    """

    _install_rtc_autograd_activations(model)
    core = getattr(getattr(model, "action_expert", None), "core_model", None)
    decorated = getattr(type(core), "denoise_step", None) if core is not None else None
    implementation = getattr(decorated, "__wrapped__", None)
    if implementation is None:
        raise RuntimeError("soft RTC requires access to the grad-enabled Pi0.5 denoise implementation")
    # The enclosing SGLang action stage runs in inference_mode, so its
    # per-step timestep is an inference tensor.  Pi0.5's suffix embedder saves
    # that scalar while differentiating with respect to the action tensor;
    # PyTorch rejects inference tensors saved for backward.  Materialize the
    # tiny batch as a normal tensor before entering the grad-enabled core.
    with torch.inference_mode(False):
        timestep = timestep.detach().clone()
    return implementation(
        core,
        prefix_context.prefix_pad_masks,
        prefix_context.past_key_values,
        x,
        timestep,
        bool(prefix_context.layout.get("full_attention", False)),
        action_position_offset=0,
    )


def _rtc_autograd_prefix_context(prefix_context: Any) -> Any:
    """Copy request prefix tensors out of the server's inference-mode arena.

    SDPA saves the fixed prefix K/V tensors while differentiating with respect
    to the suffix query.  PyTorch refuses to save inference tensors for
    backward, so soft RTC materializes one normal-tensor cache per request and
    reuses it across all denoise steps.
    """

    with torch.inference_mode(False):
        cache = prefix_context.past_key_values
        normal_cache = type(cache)(
            (
                keys.detach().clone(),
                values.detach().clone(),
                sliding_window,
            )
            for keys, values, sliding_window in cache
        )
        normal_masks = prefix_context.prefix_pad_masks.detach().clone()
    return replace(
        prefix_context,
        past_key_values=normal_cache,
        prefix_pad_masks=normal_masks,
    )


def _rtc_velocity(
    denoise: Any,
    x: torch.Tensor,
    time_value: torch.Tensor,
    previous: torch.Tensor,
    weights: torch.Tensor,
    maximum: float,
    *,
    telemetry: bool = False,
) -> torch.Tensor:
    """Compute the soft RTC VJP used by EmbodiInfer's flow decoder."""

    global _RTC_TELEMETRY_REQUESTS

    # SGLang may invoke the action pipeline under inference_mode.  Unlike
    # no_grad, inference_mode cannot be overridden by enable_grad alone, so
    # leave it explicitly for this one VJP and return to the caller's mode.
    with torch.inference_mode(False), torch.enable_grad():
        leaf = x.detach().clone().requires_grad_(True)
        # ``time_value`` is a scalar view from an inference-mode linspace in
        # the server loop.  The clean-action product also saves it for VJP.
        time_value = time_value.detach().clone().to(device=leaf.device, dtype=leaf.dtype)
        previous = previous.detach().clone().to(device=leaf.device, dtype=leaf.dtype)
        weights = weights.detach().clone().to(device=leaf.device, dtype=leaf.dtype)
        velocity = denoise(leaf)
        if not velocity.requires_grad or velocity.grad_fn is None:
            raise RuntimeError("soft RTC denoiser returned a detached velocity; the VJP would degenerate to identity")
        if velocity.shape != leaf.shape:
            raise RuntimeError("soft RTC denoiser returned an unexpected action shape")
        action_ness = 1.0 - time_value
        clean = leaf - (1.0 - action_ness) * velocity
        error = (previous - clean) * weights.reshape(1, -1, 1)
        correction = torch.autograd.grad(clean, leaf, error, retain_graph=False)[0]
        telemetry_request = None
        if telemetry and _RTC_TELEMETRY_REQUESTS < 2:
            with _RTC_TELEMETRY_LOCK:
                if _RTC_TELEMETRY_REQUESTS < 2:
                    _RTC_TELEMETRY_REQUESTS += 1
                    telemetry_request = _RTC_TELEMETRY_REQUESTS
        if telemetry_request is not None:
            if not torch.isfinite(velocity).all() or not torch.isfinite(correction).all():
                raise RuntimeError("soft RTC produced non-finite velocity or VJP correction")
            identity_delta = (correction - error).abs().max().detach()
            if not torch.isfinite(identity_delta) or float(identity_delta) <= 0.0:
                raise RuntimeError("soft RTC VJP correction is identical to the identity error")
            logger.info(
                "soft RTC VJP verified request=%d velocity_requires_grad=%s "
                "velocity_grad_fn=%s correction_identity_max_abs=%.9g",
                telemetry_request,
                velocity.requires_grad,
                type(velocity.grad_fn).__name__,
                float(identity_delta),
            )
    return (velocity - _rtc_strength(action_ness, maximum) * correction).detach()


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
        processor_path = root / f"policy_{pipeline}.json"
        if not processor_path.is_file():
            raise FileNotFoundError(f"checkpoint is missing policy_{pipeline}.json")
        payload = json.loads(processor_path.read_text())
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
    def _apply_checkpoint_config(model_path: str, config: Any) -> None:
        """Apply local tokenizer and LeRobot feature metadata."""

        Pi05PolicyModel._apply_checkpoint_config(model_path, config)
        root = Path(model_path)
        local_tokenizer = root / "tokenizer_local"
        if (local_tokenizer / "tokenizer_config.json").is_file():
            config.tokenizer_name = str(local_tokenizer)
        config_path = root / "config.json"
        if not config_path.is_file():
            return
        payload = json.loads(config_path.read_text())
        input_features = payload.get("input_features") or {}
        image_prefix = "observation.images."
        image_keys = tuple(
            key[len(image_prefix) :]
            for key, feature in input_features.items()
            if key.startswith(image_prefix) and isinstance(feature, dict) and feature.get("type") == "VISUAL"
        )
        if image_keys:
            # SGLang's Pi0.5 defaults name the three OpenPI cameras. Use the
            # checkpoint's actual camera contract when it is available.
            config.image_keys = image_keys
        state = input_features.get("observation.state") or {}
        shape = state.get("shape") or []
        if shape:
            config.state_dim = int(shape[0])

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
    def __init__(
        self,
        processor: Pi05Preprocessor,
        statistics: _Statistics,
    ) -> None:
        self.processor = processor
        self.statistics = statistics

    def __call__(self, raw_observation: dict[str, Any]) -> Any:
        observation = dict(raw_observation)
        async_block = _async_block(observation)
        observation["state"] = self.statistics.normalize(observation["state"])
        batch = self.processor(observation)
        # ``VLAObservationBatch.metadata`` is the only object carried from the
        # request adapter into the denoising stage.  Keep the validated async
        # block there; SGLang's generic action endpoint otherwise drops the
        # request metadata after building ActionSamplingParams.
        batch.metadata = {
            **dict(getattr(batch, "metadata", {}) or {}),
            "async": async_block,
            "session_id": observation.get("session_id"),
            "session_revision": observation.get("session_revision", 0),
            "step_id": observation.get("step_id"),
            "reset": bool(observation.get("reset", False)),
        }
        return batch


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


class _AsyncVLAActionDenoisingStage(VLAActionDenoisingStage):
    """Pi05 action stage with the shared RTC contract and session cache.

    SGLang's stock ``sample_actions`` loop is intentionally synchronous and
    has no hook for a committed prefix.  This stage mirrors that loop for the
    monolithic action path, adding the hard-prefix and soft-VJP RTC variants.
    """

    def __init__(self, policy_model: Any) -> None:
        super().__init__(policy_model)
        self._model_chunks: dict[tuple[str, int], torch.Tensor] = {}
        self._model_chunks_lock = threading.RLock()

    def run_grouped_requests(self, batches: list[Any], server_args: Any) -> list[Any]:
        """Keep async requests isolated from SGLang's ordinary action grouping.

        A grouped action call shares one denoising trajectory and therefore
        cannot carry a different session prefix per row. The
        stock stage may group by shape when prefix caching is disabled, so the
        async marker must explicitly force the one-request path.
        """

        for batch in batches:
            observation = vla_state(batch).get("observation_batch")
            metadata = getattr(observation, "metadata", {}) if observation is not None else {}
            async_block = metadata.get("async") if isinstance(metadata, dict) else None
            if isinstance(async_block, dict) and "rtc" in async_block:
                return [self(batch, server_args) for batch in batches]
        return super().run_grouped_requests(batches, server_args)

    def forward(self, batch: Any, server_args: Any) -> Any:
        state = vla_state(batch)
        observation = state.get("observation_batch")
        metadata = getattr(observation, "metadata", {}) if observation is not None else {}
        async_block = metadata.get("async") if isinstance(metadata, dict) else None
        rtc = async_block.get("rtc") if isinstance(async_block, dict) else None
        # Warmup must retain SGLang's exact behavior and never populate a
        # user-session action cache.
        if batch.is_warmup or not isinstance(rtc, dict):
            return super().forward(batch, server_args)

        split = __import__(
            "sglang.multimodal_gen.runtime.vla.parallel", fromlist=["get_vla_split_group"]
        ).get_vla_split_group()
        if split is not None:
            raise RuntimeError("SGLang Pi05 RTC currently requires monolithic execution (no split group)")
        prefix_context = state.get("prefix_context")
        if prefix_context is None:
            raise RuntimeError("SGLang Pi05 RTC requires a prefix context")

        session = metadata.get("session_id")
        revision = metadata.get("session_revision", 0)
        key = (str(session), int(revision)) if session is not None else None
        if metadata.get("reset") and key is not None:
            with self._model_chunks_lock:
                for cached_key in tuple(self._model_chunks):
                    if cached_key[0] == key[0]:
                        self._model_chunks.pop(cached_key, None)
        previous = self._rtc_prefix(rtc, key, observation)

        start = time.perf_counter()
        actions = self._sample_actions(
            observation,
            prefix_context,
            noise=observation.noise,
            num_steps=batch.num_inference_steps,
            use_cuda_graph=_cuda_graph_enabled(batch),
            generator=batch.generator,
            rtc=rtc if previous is not None else None,
            previous=previous,
        )
        synchronize_vla_action_tensor(actions)
        vla_timings(batch)["action_denoise_ms"] = (time.perf_counter() - start) * 1000
        state["parallel"] = self.policy_model.action_parallel_info(prefix_context)
        state["actions"] = actions
        if key is not None:
            with self._model_chunks_lock:
                self._model_chunks[key] = actions.detach().float().cpu()
        return batch

    def _rtc_prefix(self, rtc: dict[str, Any], key: tuple[str, int] | None, observation: Any) -> torch.Tensor | None:
        payload = rtc.get("prev_chunk_left_over")
        if payload is not None:
            if rtc.get("model_space") is not True:
                raise ValueError("explicit RTC prefix requires model_space=true")
            value = torch.as_tensor(payload, dtype=torch.float32)
            return value if value.numel() else None
        # The prefix starts at the action index already executed from the
        # previous full model chunk. That index is the replan stride (for the
        # paper recipe, 10), not inference latency Δ. Keep the delay fallback so
        # older wire clients remain readable.
        cache_start_index = rtc.get("cache_start_index", rtc.get("inference_delay", 0))
        if isinstance(cache_start_index, bool) or not isinstance(cache_start_index, int) or cache_start_index < 0:
            raise ValueError("async.rtc.cache_start_index must be a non-negative integer")
        explicit_cache_offset = "cache_start_index" in rtc
        if key is None:
            if explicit_cache_offset and cache_start_index > 0:
                raise ValueError("RTC cache_start_index requires session metadata and a previous model chunk")
            return None
        with self._model_chunks_lock:
            cached = self._model_chunks.get(key)
        if cached is None:
            if explicit_cache_offset and cache_start_index > 0:
                raise ValueError("RTC cache_start_index requires a previous model chunk for this session")
            return None
        if cached.ndim == 3:
            horizon = int(cached.shape[1])
            remaining = cached[:, cache_start_index:, :]
        elif cached.ndim == 2:
            # Kept for compatibility with caches populated by an older stage
            # that squeezed the batch dimension.
            horizon = int(cached.shape[0])
            remaining = cached[cache_start_index:, :]
        else:
            raise ValueError("SGLang RTC session cache must have shape [B, H, A] or [H, A]")
        if explicit_cache_offset and cache_start_index > 0 and cache_start_index >= horizon:
            raise ValueError("RTC cache_start_index must be smaller than the cached model horizon")
        return remaining if remaining.numel() else None

    def _sample_actions(
        self,
        observation: Any,
        prefix_context: Any,
        *,
        noise: Any,
        num_steps: int,
        use_cuda_graph: bool,
        generator: Any,
        rtc: dict[str, Any] | None,
        previous: torch.Tensor | None,
    ) -> torch.Tensor:
        model = self.policy_model
        if rtc is None or previous is None:
            return model.sample_actions(
                observation,
                prefix_context,
                noise=noise,
                num_steps=num_steps,
                use_cuda_graph=use_cuda_graph,
                generator=generator,
            )
        if previous.ndim == 2:
            previous = previous.unsqueeze(0)
        if previous.ndim != 3:
            raise ValueError("async.rtc.prev_chunk_left_over must have shape [T, D]")
        offload = model._offload_action_expert_between_requests()
        if offload:
            model._move_action_modules_to_device(model.device)
            use_cuda_graph = False
        x = noise
        if x is None:
            x = model.sample_noise(observation.batch_size, generator=generator)
        else:
            x = x.to(device=model.device, dtype=torch.float32).clone()
        if tuple(previous.shape[0:1]) != tuple(x.shape[0:1]):
            raise ValueError("RTC prefix batch size does not match the action batch")
        aligned = torch.zeros_like(x)
        rows = min(previous.shape[1], x.shape[1])
        cols = min(previous.shape[2], x.shape[2])
        aligned[:, :rows, :cols] = previous[:, :rows, :cols].to(device=x.device, dtype=x.dtype)
        delay = rtc.get("inference_delay", 0)
        horizon = rtc.get("execution_horizon", x.shape[1])
        maximum = float(rtc.get("max_guidance_weight", 10.0))
        schedule = str(rtc.get("prefix_attention_schedule", "linear"))
        if isinstance(delay, bool) or not isinstance(delay, int) or delay < 0:
            raise ValueError("async.rtc.inference_delay must be a non-negative integer")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0:
            raise ValueError("async.rtc.execution_horizon must be a non-negative integer")
        weights = _prefix_weights(delay, min(horizon, rows, x.shape[1]), x.shape[1], schedule).to(
            device=x.device, dtype=x.dtype
        )
        hard = bool(rtc.get("hard_prefix", False))
        rtc_prefix_context = prefix_context if hard else _rtc_autograd_prefix_context(prefix_context)
        mask = torch.arange(x.shape[1], device=x.device) < min(delay, x.shape[1])
        dt = -1.0 / num_steps
        for step_index, timestep_value in enumerate(torch.linspace(1.0, 1.0 / num_steps, num_steps, device=x.device)):
            timestep = timestep_value.expand(observation.batch_size)
            if hard:
                x[:, mask, :] = aligned[:, mask, :]
                velocity = model.denoise_step(
                    prefix_context,
                    x,
                    timestep,
                    use_cuda_graph=use_cuda_graph,
                )
            else:
                velocity = _rtc_velocity(
                    lambda x_in, timestep=timestep: _grad_enabled_denoise(model, rtc_prefix_context, x_in, timestep),
                    x,
                    timestep_value,
                    aligned,
                    weights,
                    maximum,
                    telemetry=step_index == 0,
                )
            x.add_(velocity, alpha=dt)
            if hard:
                x[:, mask, :] = aligned[:, mask, :]
        if offload:
            model._move_action_modules_to_device(torch.device("cpu"))
            torch.cuda.empty_cache()
        return x


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
        self.preprocessor = _LeRobotPreprocessor(
            self.preprocessor,
            state,
        )

    def create_pipeline_stages(self, server_args: Any) -> None:
        """Retain SGLang's prefix/denoising stages and restore action units last."""
        self.add_stage(VLAObservationPreprocessStage(self.preprocessor), "pi05_preprocess")
        self.add_stage(
            VLAPrefixEncodingStage(self.get_module("policy_model"), self.prefix_cache),
            "pi05_prefix",
        )
        self.add_stage(
            _AsyncVLAActionDenoisingStage(self.get_module("policy_model")),
            "pi05_action_denoise",
        )
        self.add_stage(_LeRobotPostprocess(self.action_statistics), "pi05_postprocess")


def _sync_api_checkpoint_config(server_args: Any) -> bool:
    """Apply checkpoint metadata in SGLang's HTTP process too.

    Model loading happens in a scheduler process, so mutations made by
    ``Pi05PolicyModel.from_pretrained`` do not reach the API process's separate
    ``server_args.pipeline_config``.  The action protocol uses that copy for
    metadata and image routing; synchronize it before either operation.
    """

    selected = getattr(server_args, "pipeline_class_name", None)
    if selected not in (None, LeRobotPi05Pipeline.pipeline_name):
        return False
    model_path = getattr(server_args, "model_path", None)
    config = getattr(server_args, "pipeline_config", None)
    if model_path is not None and config is not None:
        _LeRobotPolicyModel._apply_checkpoint_config(model_path, config)
    return True


def _register_pipeline() -> None:
    # The console entry point imports this module in spawned workers too.
    from sglang.multimodal_gen import registry
    from sglang.multimodal_gen.runtime.entrypoints.action import api as action_api

    prepare_sglang_environment()
    registry._discover_and_register_pipelines()
    registry._PIPELINE_REGISTRY[LeRobotPi05Pipeline.pipeline_name] = LeRobotPi05Pipeline
    registry._PIPELINE_CONFIG_REGISTRY[LeRobotPi05Pipeline.pipeline_name] = (
        LeRobotPi05Pipeline.pipeline_config_cls,
        LeRobotPi05Pipeline.sampling_params_cls,
    )
    # EmbodiRun probes this endpoint before enabling async execution.  SGLang's
    # generic action metadata has no knowledge of deployment-side RTC, so
    # annotate the response only for the pipeline that implements it.
    if not getattr(action_api, "_embodirun_async_metadata", False):
        original_metadata = action_api.action_metadata
        original_infer_action = action_api.infer_action

        def _metadata_with_async(server_args: Any) -> dict[str, Any]:
            selected = _sync_api_checkpoint_config(server_args)
            metadata = dict(original_metadata(server_args))
            # This entry point may still be used with another registered
            # SGLang pipeline.  Advertise the deployment contract only when
            # the server actually selected the pipeline that implements it.
            if not selected:
                return metadata
            async_capabilities = {"schema": _ASYNC_SCHEMA, "rtc": True}
            metadata["async_inference"] = async_capabilities
            capabilities = dict(metadata.get("capabilities") or {})
            capabilities["async_inference"] = async_capabilities
            metadata["capabilities"] = capabilities
            return metadata

        async def _infer_action_with_checkpoint_config(payload: dict[str, Any], server_args: Any) -> dict[str, Any]:
            _sync_api_checkpoint_config(server_args)
            return await original_infer_action(payload, server_args)

        action_api.action_metadata = _metadata_with_async
        action_api.infer_action = _infer_action_with_checkpoint_config
        action_api._embodirun_async_metadata = True


def main() -> None:
    """Run SGLang's CLI with the opt-in LeRobotPi05Pipeline registered."""
    from sglang.cli.main import main as sglang_main

    sglang_main()


_register_pipeline()
