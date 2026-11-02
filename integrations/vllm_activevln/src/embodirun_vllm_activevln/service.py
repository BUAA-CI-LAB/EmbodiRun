"""Expose native vLLM ActiveVLN inference through the shared action protocol.

EmbodiInfer supplies the checkpoint-specific prompt/parser and HTTP contract;
only vLLM loads weights and executes the model. No benchmark modules are used.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from embodiinfer.engine.serve.contracts import ModelAction, ModelResult, RawPolicyRequest
from embodiinfer.engine.serve.http_server import PolicyHttpService, create_http_server
from embodiinfer.policies.activevln.prompt_activevln import actions_to_tensor, parse_r2r_actions, render_turn_text
from PIL import Image


class Generator(Protocol):
    """Native model boundary; accepts a complete, private recurrent history."""

    processor: Any

    def generate(self, tokens: list[int], images: list[Image.Image], identity: str) -> list[int]:
        """Return generated tokens, retaining the first complete STOP or EOS."""
        ...


@dataclass
class History:
    """Committed tokens and images for one episode, never shared across sessions."""

    instruction: str
    step: int = -1
    identity: str = field(default_factory=lambda: uuid.uuid4().hex)
    tokens: list[int] = field(default_factory=list)
    images: list[Image.Image] = field(default_factory=list)


class ActiveVLNVllmAdapter:
    """Serialize inference and commit history only after a successful model call."""

    action_space = "activevln.r2r.discrete.v1"
    image_field = "observation.images.rgb"

    def __init__(self, generator: Generator, *, maximum_history_images: int = 200) -> None:
        if type(maximum_history_images) is not int or maximum_history_images < 1:
            raise ValueError("maximum_history_images must be positive")
        self.generator = generator
        self.maximum_history_images = maximum_history_images
        self._history: dict[str, History] = {}
        self._lock = threading.RLock()

    def capabilities(self) -> dict[str, Any]:
        """Advertise synchronous R2R serving; execution-prefix RTC is unsupported."""
        return dict(
            model="activevln",
            engine="vllm",
            action_space=self.action_space,
            image_fields=[self.image_field],
            recurrent=True,
            max_batch_size=1,
        )

    def infer(self, request: RawPolicyRequest) -> ModelResult:
        """Validate the observation and advance exactly one session-local turn."""
        if len(request.images) != 1 or request.images[0].name != self.image_field:
            raise ValueError(f"expected exactly one {self.image_field} image")
        image = request.images[0]
        if image.mime_type not in {"image/png", "image/jpeg"}:
            raise ValueError("ActiveVLN requires PNG or JPEG")
        if "activevln_execution" in request.metadata:
            raise ValueError("vLLM adapter does not support partial execution prefixes")
        with Image.open(io.BytesIO(image.data)) as opened:
            if opened.width * opened.height > 8_000_000:
                raise ValueError("image exceeds eight million pixels")
            rgb = opened.convert("RGB")
        with self._lock:
            previous = self._history.get(request.session_id, History(request.instruction))
            if request.step_id != previous.step + 1:
                raise ValueError("steps must be consecutive within a session")
            if request.instruction != previous.instruction:
                raise ValueError("reset the session before changing its instruction")
            if len(previous.images) >= self.maximum_history_images:
                raise ValueError("episode image history limit reached; reset the session")
            started = time.perf_counter()
            processor = self.generator.processor
            turn = render_turn_text(processor, request.instruction, initial=not previous.images, action_space="r2r")
            tokens = previous.tokens + processor.tokenizer.encode(turn, add_special_tokens=False)
            images = [*previous.images, rgb]
            generated = self.generator.generate(tokens, images, previous.identity)
            if not generated:
                raise RuntimeError("native vLLM returned no tokens")
            text = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
            parsed = parse_r2r_actions(text)
            rows, mask = actions_to_tensor(parsed)
            self._history[request.session_id] = History(
                request.instruction, request.step_id, previous.identity, tokens + generated, images
            )
            return ModelResult(
                self.action_space,
                (
                    ModelAction(
                        "discrete_chunk",
                        dict(
                            rows=rows.tolist(),
                            action_mask=mask.tolist(),
                            text=text,
                            parsed_valid=parsed.valid,
                            token_ids=generated,
                        ),
                    ),
                ),
                {"e2e_ms": (time.perf_counter() - started) * 1000},
                "vllm-0.30.0",
            )

    def reset(self, session_id: str) -> None:
        """Drop only the requested episode; a new cache salt prevents stale reuse."""
        with self._lock:
            self._history.pop(session_id, None)


class NativeVLLM:
    """Own the pinned native engine; no private benchmark instrumentation."""

    def __init__(
        self, checkpoint: str, *, max_context: int, max_new_tokens: int, engine_options: dict[str, Any] | None = None
    ) -> None:
        from transformers import AutoProcessor
        from vllm import EngineArgs, LLMEngine, SamplingParams

        self.processor = AutoProcessor.from_pretrained(checkpoint, use_fast=False)
        options = dict(
            dtype="bfloat16",
            max_model_len=max_context,
            max_num_seqs=1,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            gpu_memory_utilization=0.85,
            seed=42,
            limit_mm_per_prompt={"image": 200, "video": 0},
            mm_processor_kwargs={"min_pixels": 1024, "max_pixels": 76800, "use_fast": False},
        )
        overrides = dict(engine_options or {})
        if {"model", "tokenizer", "async_scheduling", "max_num_seqs"} & overrides.keys():
            raise ValueError("engine options cannot override model identity or serial scheduling")
        options.update(overrides, model=checkpoint, tokenizer=checkpoint, async_scheduling=False, max_num_seqs=1)
        self.engine = LLMEngine.from_engine_args(EngineArgs(**options))
        self.sampling = SamplingParams(
            temperature=0,
            top_p=1,
            top_k=-1,
            repetition_penalty=1.0,
            max_tokens=max_new_tokens,
            stop_token_ids=[151645, 151643],
            detokenize=False,
        )

    def generate(self, tokens: list[int], images: list[Image.Image], identity: str) -> list[int]:
        """Keep exactly the first EOS/STOP prefix, including speculative blocks."""
        request_id = uuid.uuid4().hex
        (prompt,) = self.engine.renderer.render_cmpl(
            [
                dict(
                    prompt_token_ids=list(tokens),
                    multi_modal_data={"image": list(images)},
                    multi_modal_uuids={"image": [f"{identity}-{i}" for i in range(len(images))]},
                    cache_salt=identity,
                    mm_processor_kwargs={"use_fast": False},
                )
            ]
        )
        self.engine.add_request(request_id, prompt, self.sampling)
        retained: list[int] = []
        try:
            while self.engine.has_unfinished_requests():
                for output in self.engine.step():
                    if output.request_id != request_id or len(output.outputs) != 1:
                        raise RuntimeError("unexpected native vLLM response")
                    complete = list(output.outputs[0].token_ids)
                    if complete[: len(retained)] != retained:
                        raise RuntimeError("native vLLM rewrote generated tokens")
                    for end in range(len(retained) + 1, len(complete) + 1):
                        prefix = complete[:end]
                        parsed = parse_r2r_actions(
                            self.processor.tokenizer.decode(prefix, skip_special_tokens=True).strip()
                        )
                        if prefix[-1] in (151645, 151643) or (parsed.valid and parsed.actions[-1].name == "stop"):
                            return prefix
                    retained = complete
            return retained
        finally:
            self.engine.abort_request([request_id])

    def close(self) -> None:
        """Release the native engine when the service exits."""
        self.engine.engine_core.shutdown()


def main(argv: list[str] | None = None) -> None:
    """Start a separately installed ActiveVLN action service, loopback by default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--engine-options", type=Path)
    parser.add_argument("--token-env", default="EMBODIRUN_VLLM_TOKEN")
    args = parser.parse_args(argv)
    if not args.device.startswith("cuda:") or not args.device[5:].isdigit():
        parser.error("device must use cuda:<index>")
    if args.max_context < 1 or args.max_new_tokens < 1:
        parser.error("context and generation limits must be positive")
    token = os.environ.get(args.token_env)
    if args.host not in {"localhost", "127.0.0.1", "::1"} and (not token or len(token) < 32):
        parser.error("a non-loopback listener requires a token of at least 32 characters")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device[5:]
    options = json.loads(args.engine_options.read_text()) if args.engine_options else None
    if options is not None and not isinstance(options, dict):
        parser.error("engine options must be a JSON object")
    native = NativeVLLM(
        args.checkpoint, max_context=args.max_context, max_new_tokens=args.max_new_tokens, engine_options=options
    )
    try:
        server = create_http_server(
            PolicyHttpService(
                ActiveVLNVllmAdapter(native),
                token=token,
                max_body_bytes=10 * 1024 * 1024,
                maximum_sessions=1,
                maximum_images=1,
            ),
            host=args.host,
            port=args.port,
        )
        try:
            server.serve_forever()
        finally:
            server.server_close()
    finally:
        native.close()


if __name__ == "__main__":
    main()
