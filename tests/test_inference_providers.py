from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib

from embodirun.model_services import (
    InferenceProvider as CanonicalProvider,
)
from embodirun.model_services import (
    provider as canonical_provider,
)
from embodirun.model_services.backends.sglang.http import (
    SglangHttpClient as CanonicalSglangHttpClient,
)
from embodirun.services.control.contracts import ControlServiceConfig
from embodirun.services.host.cli import main
from embodirun.services.host.config import ConfigError, load_config
from embodirun.services.host.environment import environment_profiles
from embodirun.services.host.executor import CommandResult, JsonHttpResponse
from embodirun.services.host.plan import build_plan
from embodirun.services.inference import (
    InferenceProvider,
    build_inference_client,
    register_provider,
)
from embodirun.services.inference.backends.sglang.http import (
    SglangHttpClient as LegacySglangHttpClient,
)
from embodirun.services.inference.providers import (
    InferenceProvider as LegacyProvider,
)


def _config(tmp_path: Path, model: str, provider: str = "embodiinfer") -> Path:
    path = tmp_path / "deployment.yaml"
    path.write_text(
        f"""metadata:\n  name: external\n  deploy-commit: test\nnodes:\n  host:\n    type: workstation\n    connection: {{type: local}}\nrobots:\n  robot:\n    type: lerobot.so101\n    node: host\n    port: /dev/null\nsensors:\n  state:\n    type: state\n    node: host\nmodels:\n  {model}:\n    provider: {provider}\n    transport: http\n    service: external\n    endpoint: http://127.0.0.1:9999\n    type: pi05\nruntimes:\n  runtime:\n    robot: robot\n    model: {model}\n    binding: lerobot.so101.pi05\n    inputs: {{state: state}}\n    server: {{bind: 127.0.0.1, port: 8101}}\n""",
        encoding="utf-8",
    )
    return path


class _LifecycleExecutor:
    """Small node fake that records Host init/up/down without network or git."""

    def __init__(self) -> None:
        self.commands = []
        self.files = {}
        self.symlinks = {}
        self.closed = False

    def run(self, command, *, check=True):
        self.commands.append((command, check))
        argv = command.argv
        if argv == ("printenv", "HOME"):
            return CommandResult(0, "/home/test\n", "")
        if argv == ("printenv", "PATH"):
            return CommandResult(0, "/usr/bin\n", "")
        if argv == ("uname", "-s"):
            return CommandResult(0, "Linux\n", "")
        if argv == ("uname", "-m"):
            return CommandResult(0, "x86_64\n", "")
        if argv == ("/usr/bin/python3", "--version"):
            return CommandResult(0, "Python 3.12.1\n", "")
        if "remote" in argv and "get-url" in argv:
            return CommandResult(
                0,
                "https://github.com/BUAA-CI-LAB/EmbodiRun.git\n",
                "",
            )
        if "rev-parse" in argv and "HEAD" in argv:
            return CommandResult(0, "resolved\nresolved\n", "")
        if len(argv) > 2 and argv[1].endswith("/supervisor.py"):
            return CommandResult(0, "running\t321\n", "")
        return CommandResult(0, "", "")

    def write_text(self, path, content, *, mode=0o600):
        self.files[path] = (content, mode)

    def replace_symlink(self, path, target):
        self.symlinks[path] = target

    def get_json(self, url, *, timeout_s):
        return JsonHttpResponse(200, {"status": "ok"})

    def close(self):
        self.closed = True


def _run_lifecycle(config: Path, tmp_path: Path):
    executors = []

    def factory(_node):
        executor = _LifecycleExecutor()
        executors.append(executor)
        return executor

    args = ("--config", str(config), "--state-dir", str(tmp_path / "state"))
    assert main((*args, "init"), executor_factory=factory) == 0
    assert main((*args, "up", "--wait-timeout", "1"), executor_factory=factory) == 0
    assert main((*args, "down"), executor_factory=factory) == 0
    return executors


def test_legacy_and_canonical_model_service_imports_share_registry_and_types() -> None:
    assert LegacyProvider is CanonicalProvider
    assert LegacySglangHttpClient is CanonicalSglangHttpClient
    assert canonical_provider("embodiinfer") is not None


@pytest.mark.parametrize("order", ["legacy-first", "canonical-first"])
def test_legacy_and_canonical_import_order_keeps_one_registry(order: str) -> None:
    """A fresh interpreter must register and resolve providers through one registry."""

    if order == "legacy-first":
        imports = """
from importlib import import_module
legacy = import_module("embodirun.services.inference.providers")
canonical = import_module("embodirun.model_services.providers")
"""
    else:
        imports = """
from importlib import import_module
canonical = import_module("embodirun.model_services.providers")
legacy = import_module("embodirun.services.inference.providers")
"""
    script = (
        imports
        + """
assert legacy.InferenceProvider is canonical.InferenceProvider
candidate = canonical.InferenceProvider(
    "subprocess-order-provider", frozenset({"http"}), True,
    lambda endpoint, options, timeout: object(),
)
canonical.register_provider(candidate)
assert legacy.provider("subprocess-order-provider") is candidate
assert canonical.provider("subprocess-order-provider") is candidate
print("registry-ok")
"""
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "registry-ok"


def test_external_model_needs_no_node_server_or_checkpoint(tmp_path: Path) -> None:
    config = load_config(_config(tmp_path, "policy"))
    assert config.models["policy"].node is None
    assert config.models["policy"].server is None
    plan = build_plan(config)
    assert [service.kind for service in plan.services] == ["control"]
    assert plan.runtimes[0].model_endpoint == "http://127.0.0.1:9999"


def test_device_only_init_up_down_does_not_prepare_or_launch_model(
    tmp_path: Path,
) -> None:
    config = Path(__file__).parents[1] / "configs/examples/device-only.yaml"
    executors = _run_lifecycle(config, tmp_path)
    commands = [command for executor in executors for command, _ in executor.commands]
    assert not any("third_party/embodiinfer" in item for command in commands for item in command.argv)
    starts = [
        json.loads(command.stdin)
        for command in commands
        if len(command.argv) > 2 and command.argv[1].endswith("/supervisor.py") and command.argv[2] == "start"
    ]
    assert len(starts) == 1
    assert starts[0]["argv"][0].endswith("/embodirun-control-serve")


def test_external_init_up_down_never_checks_checkpoint_or_stops_external_service(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, "policy")
    executors = _run_lifecycle(config, tmp_path)
    commands = [command for executor in executors for command, _ in executor.commands]
    starts = [
        json.loads(command.stdin)
        for command in commands
        if len(command.argv) > 2 and command.argv[1].endswith("/supervisor.py") and command.argv[2] == "start"
    ]
    stops = [
        command.argv[4]
        for command in commands
        if len(command.argv) > 4 and command.argv[1].endswith("/supervisor.py") and command.argv[2] == "stop"
    ]
    assert len(starts) == len(stops) == 1
    assert starts[0]["argv"][0].endswith("/embodirun-control-serve")
    assert all("9999" not in item for command in commands for item in command.argv)


def test_registered_managed_provider_runs_host_init_up_down_lifecycle(
    tmp_path: Path,
) -> None:
    provider_name = "test-lifecycle-provider"
    register_provider(
        InferenceProvider(
            provider_name,
            frozenset({"http"}),
            True,
            lambda endpoint, options, timeout: object(),
            environment_group="test-lifecycle",
            source_project="deploy",
            managed_command=lambda options: (
                "test-lifecycle-serve",
                options.checkpoint,
                "--port",
                str(options.port),
            ),
            health_suffix="/health",
        )
    )
    text = _config(tmp_path, "policy", provider=provider_name).read_text()
    text = text.replace("service: external", "service: managed")
    text = text.replace(
        "endpoint: http://127.0.0.1:9999",
        "node: host\n    source: /models/test\n    server: {bind: 127.0.0.1, port: 9999}",
    )
    config = tmp_path / "managed-lifecycle.yaml"
    config.write_text(text, encoding="utf-8")
    executors = _run_lifecycle(config, tmp_path)
    commands = [command for executor in executors for command, _ in executor.commands]
    starts = [
        json.loads(command.stdin)
        for command in commands
        if len(command.argv) > 2 and command.argv[1].endswith("/supervisor.py") and command.argv[2] == "start"
    ]
    model = next(item for item in starts if item["argv"][0].endswith("/test-lifecycle-serve"))
    assert model["argv"][1:4] == ["/models/test", "--port", "9999"]
    assert any(
        len(command.argv) >= 3 and command.argv[0].endswith("/uv") and command.argv[1:3] == ("sync", "--frozen")
        for command in commands
    )
    assert any("remote" in command.argv and "get-url" in command.argv for command in commands)
    assert any("rev-parse" in command.argv for command in commands)


def test_external_model_reports_endpoint_field_locally(tmp_path: Path) -> None:
    path = _config(tmp_path, "policy")
    text = path.read_text().replace("endpoint: http://127.0.0.1:9999", "endpoint: local")
    path.write_text(text)
    with pytest.raises(ConfigError, match=r"models\.policy\.endpoint"):
        load_config(path)


@pytest.mark.parametrize(
    "replacement, message",
    [
        ("endpoint: http://127.0.0.1:bad", "valid URL"),
        ("endpoint: http:///missing-host", "absolute URL"),
        ("endpoint: http://user:secret@127.0.0.1:9999", "credentials"),
        ("endpoint: http://127.0.0.1:9999/path?token=oops", "query"),
        ("endpoint: http://127.0.0.1:9999/path#fragment", "query or fragment"),
    ],
)
def test_external_endpoint_rejects_unsafe_url_forms(tmp_path: Path, replacement: str, message: str) -> None:
    path = _config(tmp_path, "policy").read_text()
    path = path.replace("endpoint: http://127.0.0.1:9999", replacement)
    endpoint = tmp_path / "endpoint.yaml"
    endpoint.write_text(path)
    with pytest.raises(ConfigError, match=message):
        load_config(endpoint)


def test_model_aliases_and_managed_fields_are_validated_locally(tmp_path: Path) -> None:
    path = _config(tmp_path, "policy").read_text()
    path = path.replace("    type: pi05", "    type: pi05\n    typo_option: true")
    typo = tmp_path / "typo.yaml"
    typo.write_text(path)
    with pytest.raises(ConfigError, match="unknown fields: typo_option"):
        load_config(typo)

    mismatch = tmp_path / "mismatch.yaml"
    mismatch.write_text(
        _config(tmp_path, "policy")
        .read_text()
        .replace("    service: external", "    service: external\n    lifecycle: managed")
    )
    with pytest.raises(ConfigError, match="service and lifecycle must match"):
        load_config(mismatch)

    external_managed_field = tmp_path / "external-managed-field.yaml"
    external_managed_field.write_text(
        _config(tmp_path, "policy")
        .read_text()
        .replace(
            "    endpoint: http://127.0.0.1:9999",
            "    endpoint: http://127.0.0.1:9999\n    server: {bind: 127.0.0.1, port: 9999}",
        )
    )
    with pytest.raises(ConfigError, match="external services cannot declare"):
        load_config(external_managed_field)


def test_sglang_environment_installs_local_optional_package_from_deploy_source(
    tmp_path: Path,
) -> None:
    text = _config(tmp_path, "policy", provider="sglang").read_text()
    text = text.replace("service: external", "service: managed")
    text = text.replace(
        "endpoint: http://127.0.0.1:9999",
        "node: host\n    source: /models/pi05\n    environment_packages: ['sglang[diffusion]==0.5.18']\n    server: {bind: 127.0.0.1, port: 9999}",
    )
    config_path = tmp_path / "managed-sglang.yaml"
    config_path.write_text(text)
    profile = next(item for item in environment_profiles(load_config(config_path)) if item.group == "sglang")
    assert profile.project == "inference"
    assert profile.packages == (
        "sglang[diffusion]==0.5.18",
        "-e",
        "../deploy",
        "-e",
        "../deploy/integrations/sglang_pi05",
    )


def test_sglang_package_metadata_keeps_core_dependency_and_new_entrypoint() -> None:
    metadata = tomllib.loads(
        (Path(__file__).parents[1] / "integrations" / "sglang_pi05" / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    assert "embodirun>=0.1.0" in metadata["dependencies"]
    assert metadata["scripts"]["embodirun-sglang-pi05-serve"].endswith(".pi05:main")
    assert metadata["scripts"]["rlinf-sglang-pi05-serve"].endswith(".pi05:main")
    assert "rlinf-sglang-serve" not in metadata["scripts"]


def test_host_plan_control_config_preserves_sglang_options(tmp_path: Path) -> None:
    config_path = _config(tmp_path, "policy", provider="sglang")
    config = load_config(config_path)
    plan = build_plan(config)
    control = next(service for service in plan.services if service.kind == "control")
    control_config = ControlServiceConfig.from_json(control.control_config_json or "")
    assert control_config.inference_backend == "sglang"
    client = build_inference_client(
        control_config.inference_transport,
        control_config.inference_endpoint,
        control_config.inference_options,
        backend=control_config.inference_backend,
        timeout_s=1,
    )
    assert client.__class__.__name__ == "SglangHttpClient"


def test_registered_managed_provider_builds_host_plan_without_host_enum(
    tmp_path: Path,
) -> None:
    provider_name = "test-managed-provider"
    register_provider(
        InferenceProvider(
            provider_name,
            frozenset({"http"}),
            True,
            lambda endpoint, options, timeout: object(),
            environment_group="test-managed",
            managed_command=lambda options: (
                "test-provider-serve",
                options.checkpoint,
                "--port",
                str(options.port),
            ),
            health_suffix="/health",
        )
    )
    path = _config(tmp_path, "policy").read_text()
    path = path.replace("provider: embodiinfer", f"provider: {provider_name}")
    path = path.replace("service: external", "service: managed")
    path = path.replace(
        "endpoint: http://127.0.0.1:9999",
        "node: host\n    source: /models/test\n    server: {bind: 127.0.0.1, port: 9999}",
    )
    config_path = tmp_path / "managed.yaml"
    config_path.write_text(path)
    plan = build_plan(load_config(config_path))
    model_service = next(item for item in plan.services if item.kind == "model")
    assert model_service.command.argv == (
        "test-provider-serve",
        "/models/test",
        "--port",
        "9999",
    )
    assert model_service.health_endpoint.endswith("/health")


def test_registered_third_provider_does_not_change_control_enums() -> None:
    class FakeClient:
        pass

    register_provider(
        InferenceProvider(
            "test-provider",
            frozenset({"http"}),
            True,
            lambda endpoint, options, timeout: FakeClient(),
        )
    )
    client = build_inference_client("http", "http://policy", {}, backend="test-provider", timeout_s=1)
    assert isinstance(client, FakeClient)


def test_factory_rejects_unknown_transport_and_non_action_provider() -> None:
    transport_provider = "test-http-only"
    register_provider(
        InferenceProvider(
            "test-chat-only",
            frozenset({"http"}),
            False,
            lambda endpoint, options, timeout: object(),
        )
    )
    register_provider(
        InferenceProvider(
            transport_provider,
            frozenset({"http"}),
            True,
            lambda endpoint, options, timeout: object(),
        )
    )
    with pytest.raises(ValueError, match="no action capability"):
        build_inference_client("http", "http://chat", {}, backend="test-chat-only", timeout_s=1)
    with pytest.raises(ValueError, match="unsupported inference provider"):
        build_inference_client("http", "http://unknown", {}, backend="missing", timeout_s=1)
    with pytest.raises(ValueError, match="does not support"):
        build_inference_client("wireless", "wireless://x", {}, backend=transport_provider, timeout_s=1)
