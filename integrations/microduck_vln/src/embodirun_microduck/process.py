"""Bounded lifecycle of the one inference process owned by a demo run."""

import json
import os
import secrets
import subprocess
import sys
import time
from contextlib import contextmanager

from embodirun.model_services.backends.embodiinfer.http import EmbodiInferHttpClient, EmbodiInferHttpError

from .protocol import ACTION_SPACE, IMAGE_FIELD


@contextmanager
def managed_service(args, output):
    ready = output / "service.ready.json"
    token = secrets.token_urlsafe(32)
    env = dict(os.environ, MICRODUCK_SERVICE_TOKEN=token)
    # Only the inference child sees the inference checkout.
    env["PYTHONPATH"] = str(args.inference_root) + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        os.environ.get("MICRODUCK_INFERENCE_PYTHON", sys.executable),
        "-m",
        "embodirun_microduck.service",
        "--checkpoint",
        str(args.checkpoint),
        "--ready-file",
        str(ready),
        "--seed",
        str(args.seed),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--max-context",
        str(args.max_context),
    ]
    if args.sample:
        command.append("--sample")
    with (output / "inference.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + args.startup_timeout
            client = None
            last_error = None

            def remaining():
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Inference exited ({process.returncode}); see {output / 'inference.log'}"
                    ) from last_error
                seconds = deadline - time.monotonic()
                if seconds <= 0:
                    raise TimeoutError(f"Inference startup timed out; see {output / 'inference.log'}") from last_error
                return seconds

            while True:
                remaining()
                if client is None and ready.exists():
                    # The atomic file announces the bound port, not a responsive server.
                    address = json.loads(ready.read_text(encoding="utf-8"))
                    if address["pid"] != process.pid:
                        raise RuntimeError("Readiness file belongs to a different process")
                    client = EmbodiInferHttpClient(
                        f"http://127.0.0.1:{address['port']}", token=token, timeout_s=args.request_timeout
                    )
                if client is not None:
                    try:
                        client.timeout_s = min(1.0, args.request_timeout, remaining())
                        if client.health().get("status") == "ok":
                            client.timeout_s = min(1.0, args.request_timeout, remaining())
                            capabilities = client.capabilities().get("adapter", {})
                            if capabilities.get("action_space") != ACTION_SPACE or capabilities.get("image_fields") != [
                                IMAGE_FIELD
                            ]:
                                raise RuntimeError("Inference capabilities do not match the MicroDuck binding")
                            remaining()
                            client.timeout_s = args.request_timeout
                            break
                        last_error = RuntimeError("Inference health check failed")
                    except EmbodiInferHttpError as error:
                        last_error = error
                time.sleep(min(0.2, remaining()))
            yield client
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=15)
            ready.unlink(missing_ok=True)
