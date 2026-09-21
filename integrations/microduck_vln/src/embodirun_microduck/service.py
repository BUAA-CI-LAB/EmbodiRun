"""Optional model service; all model imports stay in this separate process.

EmbodiInfer's ActiveVLN snapshot lacks build_serving_adapter. This thin bridge
uses its existing policy, EngineCore and versioned HTTP server without patches.
Prompt construction and text parsing remain owned by the upstream policy.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import threading
from pathlib import Path

from .protocol import ACTION_SPACE, IMAGE_FIELD


class ActiveVLNAdapter:
    action_space = ACTION_SPACE

    def __init__(self, core):
        self.core = core
        self.lock = threading.RLock()

    def capabilities(self):
        return {
            "model": "activevln",
            "action_space": self.action_space,
            "image_fields": [IMAGE_FIELD],
            "recurrent": True,
            "max_batch_size": 1,
            "execution_mode": "serialized_b1",
        }

    def infer(self, request):
        import numpy as np
        import torch
        from PIL import Image
        from vvla.engine.serve.contracts import ModelAction, ModelResult
        from vvla.types import Observation, SessionKey

        if len(request.images) != 1 or request.images[0].name != IMAGE_FIELD:
            raise ValueError(f"Expected exactly one image named {IMAGE_FIELD}")
        with Image.open(io.BytesIO(request.images[0].data)) as source:
            if source.width * source.height > 8_000_000:
                raise ValueError("Image exceeds pixel limit")
            pixels = np.array(source.convert("RGB"), dtype=np.float32) / 255
        observation = Observation(
            images=torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0),
            state=torch.zeros(1),
            instruction_tokens=torch.zeros(1, dtype=torch.long),
            instruction=request.instruction,
        )
        with self.lock:
            batch = self.core.policy.collate([observation], [request.request_id])
            chunk = self.core.execute(batch, session_ids=[SessionKey(request.session_id, "http")])[0]
        rows = chunk.actions.detach().float().cpu()
        if rows.shape != (3, 2) or not torch.isfinite(rows).all():
            raise RuntimeError("ActiveVLN returned an invalid action tensor")
        trace = chunk.trace
        valid = bool(trace and trace.parsed_actions and trace.parsed_actions.valid)
        return ModelResult(
            action_space=self.action_space,
            actions=(
                ModelAction(
                    "discrete_chunk",
                    {
                        "rows": rows.tolist(),
                        "valid": valid,
                        "raw_text": trace.text if trace else None,
                        "stop_reason": trace.stop_reason if trace else None,
                    },
                ),
            ),
            timing={**{k: float(v) for k, v in chunk.timing.items()}, "policy_ms": float(chunk.latency_ms)},
            policy_revision=str(chunk.policy_version),
        )

    def reset(self, session_id):
        from vvla.types import SessionKey

        with self.lock:
            self.core.reset_sessions([SessionKey(session_id, "http")])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    import numpy as np
    import torch
    from vvla import make_policy
    from vvla.engine.config import EngineConfig
    from vvla.engine.core import EngineCore
    from vvla.engine.serve.http_server import PolicyHttpService, create_http_server

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    policy = make_policy(
        "activevln",
        checkpoint=args.checkpoint,
        attention="sdpa",
        max_new_tokens=args.max_new_tokens,
        max_context=args.max_context,
        do_sample=args.sample,
        temperature=0.2,
        top_p=0.8,
    )
    core = EngineCore(
        policy,
        EngineConfig(device="cuda", dtype="bfloat16", max_batch_size=1, use_cuda_graph=False, batch_buckets=(1,)),
    )
    # A per-run credential also isolates parallel jobs on shared compute nodes.
    token = os.environ["MICRODUCK_SERVICE_TOKEN"]
    service = PolicyHttpService(
        ActiveVLNAdapter(core),
        token=token,
        maximum_sessions=1,
        maximum_images=1,
        maximum_image_bytes=8 * 1024 * 1024,
        max_body_bytes=10 * 1024 * 1024,
        idempotency_cache_size=128,
    )
    server = create_http_server(service, host="127.0.0.1", port=0)
    temporary = args.ready_file.with_suffix(".tmp")
    temporary.write_text(json.dumps({"pid": os.getpid(), "port": server.server_address[1]}), encoding="utf-8")
    temporary.replace(args.ready_file)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
