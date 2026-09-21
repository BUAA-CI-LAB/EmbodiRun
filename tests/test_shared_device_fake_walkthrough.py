from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_fake_walkthrough_uses_real_server_and_host_cli() -> None:
    """Exercise the documented no-hardware path through the real HTTP server.

    This intentionally starts the public helper in a subprocess.  The helper
    creates a temporary Host state fixture and invokes the actual CLI entry
    point for each request; no fake transport or test-only HTTP service is
    involved.  A fixed port in the public example means this test should not
    run concurrently with another copy of the walkthrough.
    """

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "run_shared_device_fake.py")],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr

    payloads = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert payloads
    assert any(item.get("inference_enabled") is False for item in payloads)
    media_payload = next(item for item in payloads if "media" in item)
    media = media_payload["media"][0]["data_base64"]
    assert base64.b64decode(media).startswith(b"\x89PNG\r\n\x1a\n")
    assert any(item.get("status") == "completed" for item in payloads)
    assert any(item.get("status") == "cancelled" for item in payloads)
    assert "$ embodirun" in result.stderr
    assert "--state-dir" in result.stderr
    assert "recording-start" in result.stderr
    assert "cancel" in result.stderr
