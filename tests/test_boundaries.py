import ast
import subprocess
import sys
from pathlib import Path


def test_model_dependencies_are_isolated_to_the_sglang_service() -> None:
    root = Path(__file__).parents[1] / "src" / "embodirun"
    forbidden = {"models", "policies", "backends", "engine"}
    assert not forbidden.intersection(path.name for path in root.iterdir())
    integration = root / "services/inference/adapters/sglang/pi05.py"
    # Opt-in, lazily imported experiments that may touch a model runtime but are
    # never on the Host/client import path.
    opt_in = {
        integration,
        root / "services/rollout/nixl_tensors.py",
    }
    for source in root.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                package = module.partition(".")[0]
                assert package not in {"embodiinfer", "transformers"}, source
                if package in {"torch", "sglang", "safetensors"}:
                    # Opt-in integration, never Host/client runtime code.
                    assert source in opt_in, source


def test_host_and_client_import_without_inference_frameworks() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import embodirun\n"
                "import embodirun.services.host.plan\n"
                "import embodirun.services.inference.adapters.sglang\n"
                "assert not {'torch', 'sglang', 'embodiinfer'} & sys.modules.keys()\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_go2_adapter_is_retained() -> None:
    from embodirun.robots.unitree.go2 import Go2ControlClient

    assert Go2ControlClient.__name__ == "Go2ControlClient"
