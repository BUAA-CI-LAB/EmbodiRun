import json
import logging
import shlex
import shutil
import socket
import sys
import tarfile
from copy import deepcopy
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from threading import Barrier, Thread
from types import SimpleNamespace

import pytest
import yaml

from embodirun.services.control.contracts import TaskResult, error_payload
from embodirun.services.host.cli import main
from embodirun.services.host.cli.command.init import (
    DEPLOY_REPOSITORY,
    INFERENCE_REPOSITORY,
)
from embodirun.services.host.config import ConfigError, load_config
from embodirun.services.host.environment import (
    EnvironmentError,
    EnvironmentProfile,
    UvEnvironmentManager,
    environment_profiles,
)
from embodirun.services.host.executor import (
    Command,
    CommandError,
    CommandResult,
    JsonHttpResponse,
    LocalExecutor,
    SshExecutor,
)
from embodirun.services.host.plan import ServiceError, ServiceSpec, build_plan
from embodirun.services.host.source import (
    ProjectManager,
    SourceError,
    active_deploy_project,
)
from embodirun.services.host.state import (
    DeploymentState,
    EnvironmentState,
    NodeState,
    ServiceState,
    StateError,
    StateStore,
)
from embodirun.services.host.supervisor import ServiceSupervisor, SupervisorError
from embodirun.services.simulation.contracts import SimulationServiceConfig

ROOT = Path(__file__).parents[1]
CONFIGS = ROOT / "configs" / "http-wireless-inference"
EXAMPLE = CONFIGS / "http.yaml"
WIRELESS_EXAMPLE = CONFIGS / "wireless.yaml"
INFERENCE_REVISION = "a" * 40


@pytest.fixture(scope="module", autouse=True)
def service_examples(tmp_path_factory):
    """Keep lifecycle inputs small without separate copies of the lab YAMLs."""
    directory = tmp_path_factory.mktemp("service-examples")
    with pytest.MonkeyPatch.context() as patch:
        for transport, variable, arm in (
            ("http", "EXAMPLE", "so101-1"),
            ("wireless", "WIRELESS_EXAMPLE", "so101-2"),
        ):
            document = yaml.safe_load((CONFIGS / f"{transport}.yaml").read_text())
            runtime = document["runtimes"][f"{arm}-runtime"]
            robot = document["robots"][arm]
            sensors = {name: document["sensors"][name] for name in runtime["inputs"].values()}
            model = document["models"]["pi05-01"]
            model.pop("server_args")
            if transport == "http":
                document["metadata"]["name"] = "thor-so101-pi05"
                robot.update(node=model["node"], port="/dev/ttyACM0")
                for sensor in sensors.values():
                    sensor["node"] = model["node"]
                model["server"]["bind"] = "127.0.0.1"
            else:
                # Exercise explicit transport limits independently of lab defaults.
                limits = {
                    "max_peer_queued_messages": 16,
                    "max_peer_queued_bytes": 67108864,
                    "egress_quantum_bytes": 65536,
                    "egress_rate_bytes_per_second": 6250000,
                    "egress_burst_bytes": 65536,
                }
                model["transport_options"] = dict(limits)
                runtime["inference_client"]["transport_options"] = dict(limits)
            document["nodes"] = {
                name: node for name, node in document["nodes"].items() if name in {model["node"], robot["node"]}
            }
            for node in document["nodes"].values():
                node["connection"].pop("proxy_command", None)
            document.update(robots={arm: robot}, sensors=sensors, runtimes={f"{arm}-runtime": runtime})
            path = directory / f"{transport}.yaml"
            path.write_text(yaml.safe_dump(document, sort_keys=False))
            patch.setattr(sys.modules[__name__], variable, path)
        yield


def simulation_deployment(tmp_path, kind, backend, transport):
    options = {
        "vlabench": {"task": "select_fruit"},
        "libero": {"suite": "libero_spatial", "task_id": 0},
        "habitat": {
            "dataset": "/data/episodes.json.gz",
            "scenes_dir": "/data/scenes",
            "episode_id": "episode-1",
        },
        "isaac": {
            "task": "navigation",
            "instruction": "walk to the door",
            "goal_position": [1, 0, 0],
        },
    }
    navigation = kind in {"habitat", "isaac"}
    model = {
        "backend": backend,
        "transport": transport,
        "type": "streamvln" if navigation else "pi05",
        "node": "workstation",
        "source": "/models/checkpoint",
        "server": {"bind": "127.0.0.1", "port": 8000},
    }
    if backend == "sglang":
        model["environment_packages"] = ["sglang[diffusion]==0.5.18"]
    if transport == "wireless":
        model["server"]["port"] = 9300
    config_path = tmp_path / "simulation.yaml"
    config_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "name": "simulation",
                    "deploy-commit": "deploy123",
                },
                "nodes": {
                    "workstation": {
                        "type": "workstation",
                        "connection": {"type": "local"},
                    },
                },
                "simulators": {
                    "sim": {"type": kind, "node": "workstation", **options[kind]},
                },
                "models": {"policy": model},
                "runtimes": {
                    "sim-runtime": {
                        "simulator": "sim",
                        "model": "policy",
                        "binding": ("unitree.go2.streamvln" if navigation else "franka.panda.pi05"),
                        "server": {"bind": "127.0.0.1", "port": 8100},
                        **(
                            {"inference_client": {"bind": "127.0.0.1", "port": 9301}} if transport == "wireless" else {}
                        ),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return load_config(config_path)


@pytest.mark.parametrize(
    "kind,backend,transport",
    [
        (kind, "embodiinfer", transport)
        for kind in ("vlabench", "libero", "habitat", "isaac")
        for transport in ("http", "wireless")
    ]
    + [(kind, "sglang", "http") for kind in ("vlabench", "libero")],
)
def test_simulation_plan_preserves_backend_and_environment_boundaries(tmp_path, kind, backend, transport) -> None:
    config = simulation_deployment(tmp_path, kind, backend, transport)
    profiles = environment_profiles(config)
    simulation_profile = next(p for p in profiles if p.project == "deploy")
    model_profile = next(p for p in profiles if p.project == "inference")
    expected_extras = ("wireless",) if transport == "wireless" else ()
    assert simulation_profile.group == f"sim-{kind}"
    assert simulation_profile.extras == model_profile.extras == expected_extras
    manager = UvEnvironmentManager(RecordingExecutor())
    command = manager.sync_command(simulation_profile, project_dir="/opt/deploy")
    assert ("--extra" in command.argv) == (transport == "wireless")
    if transport == "wireless":
        assert command.argv[-2:] == ("--extra", "wireless")
    assert model_profile.install == ("packages" if backend == "sglang" else "project-group")

    model_service, simulation_service = build_plan(config).services
    assert model_service.command.argv[0] == ("sglang" if backend == "sglang" else f"embodiinfer-{transport}-serve")
    assert simulation_service.command.argv[0] == "embodirun-simulation-serve"
    runtime = SimulationServiceConfig.from_json(simulation_service.simulation_config_json)
    assert runtime.inference_backend == backend
    assert runtime.inference_transport == transport
    if transport == "wireless":
        assert runtime.inference_endpoint == "wireless://model.policy"
        assert runtime.inference_options["comm_config"] == ("simulation-sim-runtime.wireless.json")
        server = json.loads(model_service.wireless_config_json)
        client = json.loads(simulation_service.wireless_config_json)
        assert server["local"]["port"] == 9300
        assert client["local"]["port"] == 9301
        assert client["peers"] == [{key: value for key, value in server["local"].items() if key != "bind_host"}]
    else:
        assert model_service.wireless_config_json is None
        assert simulation_service.wireless_config_json is None


@pytest.mark.parametrize("wireless_first", [False, True])
def test_simulators_sharing_environment_union_wireless_dependencies(tmp_path, wireless_first) -> None:
    config = simulation_deployment(tmp_path, "habitat", "embodiinfer", "http")
    other_id = "aaa-wireless" if wireless_first else "zzz-wireless"
    config = replace(
        config,
        simulators={
            **config.simulators,
            other_id: replace(config.simulators["sim"], simulator_id=other_id),
        },
        models={
            **config.models,
            "wireless": replace(
                config.models["policy"],
                model_id="wireless",
                transport="wireless",
            ),
        },
        runtimes={
            **config.runtimes,
            "wireless": replace(
                config.runtimes["sim-runtime"],
                runtime_id="wireless",
                simulator=other_id,
                model="wireless",
            ),
        },
    )
    profiles = environment_profiles(config)
    assert len(profiles) == 2
    assert all(profile.extras == ("wireless",) for profile in profiles)


class RecordingExecutor:
    def __init__(self) -> None:
        self.commands = []

    def run(self, command, *, check=True):
        self.commands.append((command, check))
        return CommandResult(0, "ready\n", "")

    def close(self) -> None:
        return None


class ResultExecutor:
    def __init__(self, *results) -> None:
        self.results = list(results)
        self.commands = []

    def run(self, command, *, check=True):
        self.commands.append(command)
        return self.results.pop(0)

    def close(self) -> None:
        return None


class FakeNodeExecutor:
    def __init__(self, inference_commit=INFERENCE_REVISION) -> None:
        self.inference_commit = inference_commit
        self.commands = []
        self.files = {}
        self.health_requests = []
        self.symlinks = {}
        self.closed = False

    def run(self, command, *, check=True):
        self.commands.append((command, check))
        argv = command.argv
        if argv == ("printenv", "HOME"):
            return CommandResult(0, "/home/operator\n")
        if argv == ("printenv", "PATH"):
            return CommandResult(0, "/usr/bin:/home/operator/.local/bin\n")
        if argv == ("uname", "-s"):
            return CommandResult(0, "Linux\n")
        if argv == ("uname", "-m"):
            return CommandResult(0, "aarch64\n")
        if argv == ("/usr/bin/python3", "--version"):
            return CommandResult(0, "Python 3.10.12\n")
        if "remote" in argv and "get-url" in argv:
            repository = DEPLOY_REPOSITORY if argv[2].endswith("/deploy") else INFERENCE_REPOSITORY
            return CommandResult(0, f"{repository}\n", "")
        if "rev-parse" in argv and "HEAD" in argv:
            return CommandResult(0, "resolved\nresolved\n", "")
        if "ls-tree" in argv:
            entry = (
                f"160000 commit {self.inference_commit}\tthird_party/embodiinfer\n"
                if self.inference_commit is not None
                else ""
            )
            return CommandResult(0, entry)
        if len(argv) > 2 and argv[1].endswith("/supervisor.py"):
            return CommandResult(0, "running\t321\n", "")
        return CommandResult(0, "", "")

    def write_text(self, path, content, *, mode=0o600):
        self.files[path] = (content, mode)

    def get_json(self, url, *, timeout_s):
        self.health_requests.append((url, timeout_s))
        return JsonHttpResponse(200, {"status": "ok"})

    def replace_symlink(self, path, target):
        self.symlinks[path] = target

    def close(self) -> None:
        self.closed = True


class FailingExecutor:
    def __init__(self) -> None:
        self.closed = False

    def run(self, command, *, check=True):
        raise RuntimeError("connection refused")

    def close(self) -> None:
        self.closed = True


class ServiceReadinessExecutor:
    def __init__(
        self,
        *,
        process_state="running",
        health_exit_code=0,
    ) -> None:
        self.process_state = process_state
        self.health_exit_code = health_exit_code
        self.commands = []
        self.files = {}
        self.health_requests = []
        self.closed = False

    def run(self, command, *, check=True):
        self.commands.append((command, check))
        if command.argv[:2] == ("test", "-x"):
            return CommandResult(0)
        if len(command.argv) > 2 and command.argv[1].endswith("/supervisor.py"):
            if command.argv[2] == "start":
                return CommandResult(0, "running\t321\n")
            return CommandResult(0, f"{self.process_state}\t321\n")
        raise AssertionError(f"unexpected command: {command.argv!r}")

    def write_text(self, path, content, *, mode=0o600):
        self.files[path] = (content, mode)

    def get_json(self, url, *, timeout_s):
        self.health_requests.append((url, timeout_s))
        if self.health_exit_code != 0:
            raise OSError("not ready")
        return JsonHttpResponse(200, {"status": "ok"})

    def close(self) -> None:
        self.closed = True


class RuntimeExecutor:
    def __init__(self, result=None) -> None:
        self.requests = []
        self.closed = False
        self.result = result

    def run(self, command, *, check=True):
        raise AssertionError(f"unexpected command: {command.argv!r}")

    def request_json(self, method, url, payload, *, timeout_s):
        self.requests.append((method, url, payload, timeout_s))
        if self.result is not None:
            return self.result
        return JsonHttpResponse(
            200,
            TaskResult(
                payload["request_id"],
                payload["runtime_id"],
                payload["max_steps"],
            ).to_payload(),
        )

    def close(self) -> None:
        self.closed = True


class SyncExecutor:
    def __init__(self, *, worker_running=False, dependencies_changed=False) -> None:
        self.worker_running = worker_running
        self.dependencies_changed = dependencies_changed
        self.commands = []
        self.files = {}
        self.symlinks = {}
        self.closed = False

    def run(self, command, *, check=True):
        self.commands.append((command, check))
        argv = command.argv
        if argv[:2] == ("pgrep", "-f"):
            if self.worker_running:
                return CommandResult(0, "8675\n")
            return CommandResult(1)
        if argv[:2] == ("test", "-f"):
            path = argv[2]
            if path.endswith(("/pyproject.toml", "/uv.lock")) and "/sources/deploy/" in path:
                return CommandResult(0)
            return CommandResult(1)
        if argv[:2] == ("test", "-x"):
            return CommandResult(0)
        if argv[:1] == ("readlink",):
            target = self.symlinks.get(argv[1])
            return CommandResult(0, f"{target}\n") if target else CommandResult(1)
        if argv[0] in {"mkdir", "tar"}:
            return CommandResult(0)
        if "sync" in argv and "--group" in argv:
            return CommandResult(0)
        raise AssertionError(f"unexpected command: {argv!r}")

    def read_bytes(self, path):
        name = Path(path).name
        if self.dependencies_changed and name == "pyproject.toml":
            return b"different\n"
        return (ROOT / name).read_bytes()

    def write_bytes(self, path, content, *, mode=0o600):
        self.files[path] = (content, mode)

    def replace_symlink(self, path, target):
        self.symlinks[path] = target

    def write_text(self, path, content, *, mode=0o600):
        self.files[path] = (content.encode(), mode)

    def close(self) -> None:
        self.closed = True


class DownExecutor:
    def __init__(self) -> None:
        self.commands = []
        self.closed = False

    def run(self, command, *, check=True):
        self.commands.append((command, check))
        if len(command.argv) > 2 and command.argv[1].endswith("/supervisor.py") and command.argv[2] == "stop":
            return CommandResult(0, "stopped\n")
        raise AssertionError(f"unexpected command: {command.argv!r}")

    def close(self) -> None:
        self.closed = True


class ConcurrentInitExecutor(FakeNodeExecutor):
    def __init__(self, barrier: Barrier) -> None:
        super().__init__()
        self.barrier = barrier

    def run(self, command, *, check=True):
        if command.argv == ("uname", "-s"):
            self.barrier.wait(timeout=1.0)
        return super().run(command, check=check)


class ConcurrentUpExecutor(ServiceReadinessExecutor):
    def __init__(self, barrier: Barrier) -> None:
        super().__init__()
        self.barrier = barrier
        self.waited = False

    def run(self, command, *, check=True):
        if command.argv[:2] == ("test", "-x") and not self.waited:
            self.waited = True
            self.barrier.wait(timeout=1.0)
        return super().run(command, check=check)


class ConcurrentDownExecutor(DownExecutor):
    def __init__(self, barrier: Barrier) -> None:
        super().__init__()
        self.barrier = barrier
        self.waited = False

    def run(self, command, *, check=True):
        if len(command.argv) > 2 and command.argv[1].endswith("/supervisor.py") and not self.waited:
            self.waited = True
            self.barrier.wait(timeout=1.0)
        return super().run(command, check=check)


def two_node_model_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "two-node-models.yaml"
    second_node = """
  jetson-worker:
    type: jetson.test
    connection:
      type: local
"""
    second_model = """
  pi05-02:
    backend: embodiinfer
    transport: http
    type: pi05
    node: jetson-worker
    environment: .venv-embodiinfer
    source: /models/pi05-02
    adapter_config: /configs/pi05-02.json
    server:
      bind: 127.0.0.1
      port: 8001
"""
    config_path.write_text(
        EXAMPLE.read_text(encoding="utf-8")
        .replace("\nrobots:\n", f"{second_node}\nrobots:\n")
        .replace("\nruntimes:\n", f"{second_model}\nruntimes:\n"),
        encoding="utf-8",
    )
    return config_path


@pytest.mark.parametrize("transport", ["http", "wireless"])
def test_lab_examples_preserve_shared_model_and_launch_arguments(transport):
    config = load_config(CONFIGS / f"{transport}.yaml")
    model, *controls = build_plan(config).services
    assert model.node == "jetson-agx-thor-232"
    assert len(controls) == 2
    assert {control.node for control in controls} == {"jetson-agx-orin-174", "jetson-orin-nx"}
    assert model.command.argv[-5:] == (
        "--dtype",
        "bfloat16",
        "--num-steps",
        "10",
        "--capture-full-loop",
    )
    assert all(json.loads(control.control_config_json)["inference"]["transport"] == transport for control in controls)


def test_single_node_environment_and_runtime() -> None:
    config = load_config(EXAMPLE)
    profiles = environment_profiles(config)
    plan = build_plan(config)

    assert config.metadata.name == "thor-so101-pi05"
    assert len(profiles) == 2
    assert {(item.project, item.group) for item in profiles} == {
        ("deploy", "robot-so101"),
        ("inference", "pi05"),
    }
    assert len(plan.services) == 2
    model_service, control_service = plan.services
    assert model_service.health_endpoint == "http://127.0.0.1:8000/healthz"
    assert model_service.command.argv == (
        "embodiinfer-http-serve",
        "--policy",
        "pi05",
        "--checkpoint",
        "/models/pi05_so101",
        "--device",
        "cuda:0",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    )
    adapter_config = json.loads(model_service.adapter_config_json)
    assert adapter_config == {
        "action_feature_names": [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ],
        "image_fields": [
            "observation.images.front",
            "observation.images.wrist",
        ],
        "return_steps": 50,
        "state_fields": ["joint_positions_deg", "gripper_position"],
    }
    assert {runtime.runtime_id for runtime in plan.runtimes} == {"so101-1-runtime"}
    assert {runtime.model_endpoint for runtime in plan.runtimes} == {"http://127.0.0.1:8000"}
    assert control_service.service_id == "control-so101-1-runtime"
    assert control_service.command.argv == ("embodirun-control-serve",)
    assert control_service.health_endpoint == "http://127.0.0.1:8100/healthz"
    control_config = json.loads(control_service.control_config_json)
    assert control_config["schema"] == "rlinf.control.config.v1"
    assert control_config["binding"] == "lerobot.so101.pi05"
    assert control_config["inference"] == {
        "backend": "embodiinfer",
        "transport": "http",
        "endpoint": "http://127.0.0.1:8000",
        "options": {},
    }
    assert control_config["server"] == {"bind": "127.0.0.1", "port": 8100}
    assert control_config["robot"]["options"]["step_limit_mode"] == "clip"


def test_wireless_model_resolves_server_and_control_client_boundaries(
    tmp_path,
) -> None:
    config_path = tmp_path / "wireless.yaml"
    config_path.write_text(
        EXAMPLE.read_text(encoding="utf-8")
        .replace("transport: http", "transport: wireless")
        .replace("      port: 8000", "      port: 9300")
        .replace(
            "    inputs:\n",
            "    inference_client:\n      bind: 127.0.0.1\n      port: 9301\n    inputs:\n",
        ),
        encoding="utf-8",
    )

    plan = build_plan(load_config(config_path))

    model_service, control_service = plan.services
    assert model_service.command.argv[-2:] == (
        "--comm-config",
        "pi05-01.wireless.json",
    )
    assert model_service.command.argv[0] == "embodiinfer-wireless-serve"
    assert model_service.health_endpoint is None
    control_config = json.loads(control_service.control_config_json)
    assert control_config["inference"] == {
        "backend": "embodiinfer",
        "transport": "wireless",
        "endpoint": "wireless://model.pi05-01",
        "options": {
            "comm_config": "control-so101-1-runtime.wireless.json",
            "server_node_id": "model.pi05-01",
        },
    }
    profiles = environment_profiles(load_config(config_path))
    assert {profile.extras for profile in profiles} == {("wireless",)}


def wireless_document():
    return yaml.safe_load(WIRELESS_EXAMPLE.read_text(encoding="utf-8"))


def load_document(tmp_path, document):
    path = tmp_path / "deployment.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return load_config(path)


def test_wireless_example_generates_complementary_endpoints() -> None:
    document = yaml.safe_load(WIRELESS_EXAMPLE.read_text())
    model, control = build_plan(load_config(WIRELESS_EXAMPLE)).services
    server = json.loads(model.wireless_config_json)
    client = json.loads(control.wireless_config_json)
    model_host = document["nodes"][document["models"]["pi05-01"]["node"]]["connection"]["host"]
    client_host = document["nodes"][document["robots"]["so101-2"]["node"]]["connection"]["host"]
    assert server["local"] == {
        "node_id": "model.pi05-01",
        "host": model_host,
        "bind_host": "0.0.0.0",
        "port": 9300,
    }
    assert client["local"] == {
        "node_id": "runtime.so101-2-runtime",
        "host": client_host,
        "bind_host": "0.0.0.0",
        "port": 9300,
    }
    assert server["peers"] == [{key: value for key, value in client["local"].items() if key != "bind_host"}]
    assert client["peers"] == [{key: value for key, value in server["local"].items() if key != "bind_host"}]
    assert (
        server["comm"]
        == client["comm"]
        == {
            "max_peer_queued_messages": 16,
            "max_peer_queued_bytes": 67108864,
            "egress_quantum_bytes": 65536,
            "egress_rate_bytes_per_second": 6250000,
            "egress_burst_bytes": 65536,
        }
    )


def test_wireless_shared_model_has_distinct_runtime_peers(tmp_path) -> None:
    document = wireless_document()
    robot = deepcopy(document["robots"]["so101-2"])
    robot["port"] = "/dev/second-arm"
    document["robots"]["second-arm"] = robot
    front = deepcopy(document["sensors"]["front-camera-2"])
    front["device"] = "/dev/second-front-camera"
    wrist = deepcopy(document["sensors"]["wrist-camera-2"])
    wrist["device"] = "/dev/second-wrist-camera"
    document["sensors"]["second-front-camera"] = front
    document["sensors"]["second-wrist-camera"] = wrist
    runtime = deepcopy(document["runtimes"]["so101-2-runtime"])
    runtime["robot"] = "second-arm"
    runtime["inputs"] = {
        "observation.images.front": "second-front-camera",
        "observation.images.wrist": "second-wrist-camera",
    }
    runtime["server"]["port"] = 8101
    runtime["inference_client"]["port"] = 9301
    runtime["inference_client"]["transport_options"]["egress_rate_bytes_per_second"] = 1000000
    document["runtimes"]["second-runtime"] = runtime
    plan = build_plan(load_document(tmp_path, document))
    assert plan == build_plan(load_document(tmp_path, document))
    server = json.loads(plan.services[0].wireless_config_json)
    assert {peer["node_id"] for peer in server["peers"]} == {
        "runtime.second-runtime",
        "runtime.so101-2-runtime",
    }
    assert {peer["port"] for peer in server["peers"]} == {9300, 9301}
    clients = {service.service_id: json.loads(service.wireless_config_json) for service in plan.services[1:]}
    for client in clients.values():
        assert [peer["node_id"] for peer in client["peers"]] == ["model.pi05-01"]
    assert clients["control-second-runtime"]["comm"]["egress_rate_bytes_per_second"] == 1000000
    assert clients["control-so101-2-runtime"]["comm"]["egress_rate_bytes_per_second"] == 6250000


def test_wireless_plan_rejects_cross_robot_shared_camera(tmp_path) -> None:
    document = wireless_document()
    robot = deepcopy(document["robots"]["so101-2"])
    robot["port"] = "/dev/second-arm"
    document["robots"]["second-arm"] = robot
    runtime = deepcopy(document["runtimes"]["so101-2-runtime"])
    runtime["robot"] = "second-arm"
    runtime["server"]["port"] = 8101
    runtime["inference_client"]["port"] = 9301
    # Keep both logical sensor IDs on the same physical node/path to prove
    # startup rejects the unsupported topology before a service is spawned.
    document["runtimes"]["second-runtime"] = runtime
    with pytest.raises(ServiceError, match="sensor resource"):
        build_plan(load_document(tmp_path, document))


@pytest.mark.parametrize("transport", ["http", "wireless"])
def test_data_addresses_override_ssh_hosts(tmp_path, transport) -> None:
    document = wireless_document()
    for index, node in enumerate(document["nodes"].values()):
        node["connection"]["host"] = "127.0.0.1"  # SSH forwarded through Host.
        node["address"] = f"192.168.8.{index + 1}"
    document["models"]["pi05-01"]["transport"] = transport
    if transport == "http":
        del document["models"]["pi05-01"]["transport_options"]
        del document["runtimes"]["so101-2-runtime"]["inference_client"]
    model, control = build_plan(load_document(tmp_path, document)).services
    if transport == "http":
        assert json.loads(control.control_config_json)["inference"]["endpoint"] == "http://192.168.8.1:9300"
    else:
        server = json.loads(model.wireless_config_json)
        client = json.loads(control.wireless_config_json)
        assert server["local"]["host"] == client["peers"][0]["host"] == "192.168.8.1"
        assert client["local"]["host"] == server["peers"][0]["host"] == "192.168.8.2"


def test_wireless_multiple_models_keep_their_peers_separate(tmp_path) -> None:
    config = simulation_deployment(tmp_path, "habitat", "embodiinfer", "wireless")
    document = json.loads(config.path.read_text(encoding="utf-8"))
    document["models"]["second-model"] = deepcopy(document["models"]["policy"])
    document["models"]["second-model"]["server"]["port"] = 9400
    runtime = deepcopy(document["runtimes"]["sim-runtime"])
    runtime["model"] = "second-model"
    runtime["server"]["port"] = 8101
    runtime["inference_client"]["port"] = 9401
    document["runtimes"]["second-runtime"] = runtime
    services = build_plan(load_document(tmp_path, document)).services
    endpoints = {service.service_id: json.loads(service.wireless_config_json) for service in services}
    assert [peer["node_id"] for peer in endpoints["policy"]["peers"]] == ["runtime.sim-runtime"]
    assert [peer["node_id"] for peer in endpoints["second-model"]["peers"]] == ["runtime.second-runtime"]
    client = endpoints["simulation-second-runtime"]
    assert [peer["node_id"] for peer in client["peers"]] == ["model.second-model"]
    assert len({endpoint["local"]["port"] for endpoint in endpoints.values()}) == 4


def test_generated_endpoint_files_cannot_overwrite_another_service(tmp_path) -> None:
    config = simulation_deployment(tmp_path, "habitat", "embodiinfer", "wireless")
    document = json.loads(config.path.read_text(encoding="utf-8"))
    document["models"]["simulation-sim-runtime"] = document["models"].pop("policy")
    document["runtimes"]["sim-runtime"]["model"] = "simulation-sim-runtime"
    with pytest.raises(ValueError, match="model IDs must not collide"):
        build_plan(load_document(tmp_path, document))


@pytest.mark.parametrize(
    "path,value,match",
    [
        (
            ("runtimes", "so101-2-runtime", "inference_client"),
            None,
            "inference_client is required",
        ),
        (
            ("runtimes", "so101-2-runtime", "inference_client", "port"),
            8100,
            "share port",
        ),
        (
            ("runtimes", "so101-2-runtime", "inference_client", "port"),
            True,
            "must be an integer",
        ),
        (
            ("runtimes", "so101-2-runtime", "inference_client", "port"),
            65536,
            "between 1 and 65535",
        ),
        (
            ("runtimes", "so101-2-runtime", "inference_client", "unknown"),
            1,
            "unknown fields",
        ),
        (("models", "pi05-01", "comm_config"), "old.yaml", "generated by Host"),
        (("models", "pi05-01", "client_comm_config"), "old.yaml", "generated by Host"),
        (("models", "pi05-01", "server_node_id"), "old-id", "generated by Host"),
        (("models", "pi05-01", "transport_options"), [], "must be a mapping"),
        (
            ("runtimes", "so101-2-runtime", "inference_client", "transport_options"),
            [],
            "must be a mapping",
        ),
        (
            ("models", "pi05-01", "transport_options", "egress_burst_bytes"),
            float("inf"),
            "JSON-compatible",
        ),
        (("models", "pi05-01", "server", "bind"), "127.0.0.2", "cannot advertise"),
        (
            ("runtimes", "so101-2-runtime", "inference_client", "bind"),
            "localhost",
            "cannot advertise",
        ),
        (("nodes", "jetson-orin-nx", "address"), "0.0.0.0", "cannot advertise"),
        (("nodes", "jetson-orin-nx", "address"), "127.0.0.1", "cannot advertise"),
        (
            ("nodes", "jetson-orin-nx", "address"),
            "http://192.168.8.2:9300",
            "without URL or port",
        ),
        (
            ("nodes", "jetson-orin-nx", "connection"),
            {"type": "local"},
            "no advertised address",
        ),
    ],
)
def test_invalid_wireless_configuration_fails_before_execution(tmp_path, path, value, match) -> None:
    document = wireless_document()
    target = document
    for key in path[:-1]:
        target = target[key]
    if value is None:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    with pytest.raises(ConfigError, match=match):
        build_plan(load_document(tmp_path, document))


@pytest.mark.parametrize("field", ["inference_client", "transport_options"])
def test_http_rejects_wireless_only_fields(tmp_path, field) -> None:
    document = wireless_document()
    document["models"]["pi05-01"]["transport"] = "http"
    if field == "inference_client":
        del document["models"]["pi05-01"]["transport_options"]
    with pytest.raises(ConfigError, match=rf"{field} requires wireless"):
        load_document(tmp_path, document)


@pytest.mark.parametrize("collision", ["model", "client", "runtime"])
def test_wireless_listeners_conflict_on_same_node(tmp_path, collision) -> None:
    config = simulation_deployment(tmp_path, "libero", "embodiinfer", "wireless")
    document = json.loads(config.path.read_text(encoding="utf-8"))
    runtime = document["runtimes"]["sim-runtime"]
    if collision == "model":
        runtime["inference_client"]["port"] = 9300
    else:
        second = deepcopy(runtime)
        second["server"]["port"] = 8101
        second["inference_client"]["port"] = 9301 if collision == "client" else 9302
        if collision == "runtime":
            second["server"]["port"] = 9301
        document["runtimes"]["second"] = second
    with pytest.raises(ConfigError, match="share port"):
        load_document(tmp_path, document)


@pytest.mark.parametrize("target", ["robot", "simulator"])
def test_wireless_up_writes_native_configs_and_absolute_references(tmp_path, capsys, target) -> None:
    config = (
        load_config(WIRELESS_EXAMPLE)
        if target == "robot"
        else simulation_deployment(tmp_path, "habitat", "embodiinfer", "wireless")
    )
    executors = []

    def factory(_node):
        executor = FakeNodeExecutor()
        executors.append(executor)
        return executor

    args = ("--config", str(config.path), "--state-dir", str(tmp_path / "state"))
    assert main((*args, "init"), executor_factory=factory) == 0
    executors.clear()
    assert main((*args, "up"), executor_factory=factory) == 0
    assert capsys.readouterr().err == ""
    files = {path: content for executor in executors for path, (content, mode) in executor.files.items()}
    wireless_files = {path: json.loads(content) for path, content in files.items() if path.endswith(".wireless.json")}
    assert len(wireless_files) == 2
    assert all("/generated/" in path for path in wireless_files)
    assert all(
        mode == 0o600 for executor in executors for path, (_, mode) in executor.files.items() if path in wireless_files
    )
    service_configs = [
        json.loads(content) for path, content in files.items() if path.endswith((".control.json", ".simulation.json"))
    ]
    assert len(service_configs) == 1
    options = service_configs[0]["inference"]["options"]
    client = wireless_files[options["comm_config"]]
    assert client["peers"][0]["node_id"] == options["server_node_id"]
    start_commands = [
        command
        for executor in executors
        for command, _ in executor.commands
        if len(command.argv) > 2 and command.argv[1].endswith("/supervisor.py") and command.argv[2] == "start"
    ]
    start_requests = [json.loads(command.stdin) for command in start_commands]
    model_argv = next(request["argv"] for request in start_requests if "--comm-config" in request["argv"])
    path = model_argv[model_argv.index("--comm-config") + 1]
    assert wireless_files[path]["local"]["node_id"] == options["server_node_id"]


def test_sglang_plan_renders_an_external_pipeline_contract(tmp_path) -> None:
    """Rendering an external pipeline name does not prove SGLang model support."""
    config_path = tmp_path / "sglang-habitat.yaml"
    config_path.write_text(
        """
metadata:
  name: sglang-habitat-streamvln
  deploy-commit: deploy123
nodes:
  workstation:
    type: workstation
    connection:
      type: local
simulators:
  habitat-demo:
    type: habitat
    node: workstation
    dataset: /data/episodes.json.gz
    scenes_dir: /data/scenes
    episode_id: episode-1
models:
  streamvln:
    backend: sglang
    transport: http
    type: streamvln
    node: workstation
    environment_packages:
      - sglang[diffusion]==0.5.18
    source: /models/streamvln
    pipeline: StreamVLNPipeline
    pipeline_config: configs/streamvln-sglang.json
    gpu: cuda:1
    server_args: [--tp-size, "1"]
    server:
      bind: 127.0.0.1
      port: 30000
runtimes:
  habitat-streamvln:
    simulator: habitat-demo
    model: streamvln
    binding: unitree.go2.streamvln
    server:
      bind: 127.0.0.1
      port: 31000
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_config(config_path)
    profiles = environment_profiles(config)
    plan = build_plan(config)

    identities = {(profile.project, profile.group, profile.install) for profile in profiles}
    assert identities == {
        ("deploy", "sim-habitat", "project-group"),
        ("inference", "sglang", "packages"),
    }
    model_service, simulation_service = plan.services
    assert model_service.command.argv == (
        "sglang",
        "serve",
        "/models/streamvln",
        "--model-type",
        "diffusion",
        "--pipeline-class-name",
        "StreamVLNPipeline",
        "--pipeline-config-path",
        "configs/streamvln-sglang.json",
        "--tp-size",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        "30000",
    )
    assert model_service.command.environment == {"CUDA_VISIBLE_DEVICES": "1"}
    assert model_service.health_endpoint == "http://127.0.0.1:30000/health"
    assert model_service.adapter_config_json is None
    simulation_config = json.loads(simulation_service.simulation_config_json)
    assert simulation_config["inference"] == {
        "backend": "sglang",
        "transport": "http",
        "endpoint": "http://127.0.0.1:30000",
        "options": {
            "image_keys": {"observation.images.rgb": "rgb"},
            "parameters": {"action_horizon": 4},
            "runtime": {},
        },
    }


def test_sglang_pi05_reuses_existing_so101_binding(tmp_path) -> None:
    config_path = tmp_path / "sglang-so101.yaml"
    config_path.write_text(
        EXAMPLE.read_text(encoding="utf-8")
        .replace("backend: embodiinfer", "backend: sglang")
        .replace(
            "    source: /models/pi05_so101\n",
            '    environment_packages: ["sglang[diffusion]==0.5.18"]\n    source: /models/pi05_so101\n',
        ),
        encoding="utf-8",
    )

    model_service, control_service = build_plan(load_config(config_path)).services

    assert model_service.command.argv[:5] == (
        "sglang",
        "serve",
        "/models/pi05_so101",
        "--model-type",
        "diffusion",
    )
    control_config = json.loads(control_service.control_config_json)
    assert control_config["inference"]["backend"] == "sglang"
    assert control_config["inference"]["options"] == {
        "action_feature_names": [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ],
        "image_keys": {
            "observation.images.front": "front",
            "observation.images.wrist": "wrist",
        },
        "parameters": {"action_horizon": 50},
        "runtime": {},
        "state_fields": ["joint_positions_deg", "gripper_position"],
    }


def test_uv_environment_manager_uses_only_the_selected_group() -> None:
    profile = next(item for item in environment_profiles(load_config(EXAMPLE)) if item.project == "deploy")
    executor = RecordingExecutor()
    result = UvEnvironmentManager(executor).prepare(
        profile,
        project_dir="/opt/rlinf-deploy",
    )

    command, check = executor.commands[0]
    assert result.stdout == "ready\n"
    assert check is True
    assert command.argv == (
        "uv",
        "sync",
        "--python",
        "3.12",
        "--frozen",
        "--no-dev",
        "--group",
        "robot-so101",
    )
    assert command.cwd == "/opt/rlinf-deploy"
    assert command.environment == {"UV_PROJECT_ENVIRONMENT": ".venv-robot-so101"}


def test_uv_environment_manager_applies_configured_package_overlay() -> None:
    profile = replace(
        next(item for item in environment_profiles(load_config(EXAMPLE)) if item.project == "inference"),
        package_index="https://download.pytorch.org/whl/cu130",
        packages=("torch==2.10.0+cu130", "torchvision==0.25.0+cu130"),
    )
    executor = RecordingExecutor()

    UvEnvironmentManager(executor).prepare(
        profile,
        project_dir="/opt/rlinf-inference",
    )

    assert len(executor.commands) == 2
    command, check = executor.commands[1]
    assert check is True
    assert command.argv == (
        "uv",
        "pip",
        "install",
        "--python",
        "/opt/rlinf-inference/.venv-embodiinfer/bin/python",
        "--index-url",
        "https://download.pytorch.org/whl/cu130",
        "torch==2.10.0+cu130",
        "torchvision==0.25.0+cu130",
    )
    assert command.timeout_s == 1800.0


@pytest.fixture
def standalone_environment_profile() -> EnvironmentProfile:
    return EnvironmentProfile(
        environment_id="node:inference:sglang:.venv-sglang",
        node="node",
        project="inference",
        group="sglang",
        path=".venv-sglang",
        python="3.12",
        packages=("sglang[diffusion]==0.5.18",),
        install="packages",
    )


def test_uv_environment_manager_creates_standalone_sglang_environment(
    standalone_environment_profile,
) -> None:
    executor = ResultExecutor(CommandResult(1), CommandResult(0), CommandResult(0))

    UvEnvironmentManager(executor).prepare(
        standalone_environment_profile,
        project_dir="/opt/rlinf-inference",
    )

    assert [command.argv for command in executor.commands] == [
        ("test", "-f", "/opt/rlinf-inference/.venv-sglang/pyvenv.cfg"),
        ("uv", "venv", "--python", "3.12", ".venv-sglang"),
        (
            "uv",
            "pip",
            "install",
            "--python",
            "/opt/rlinf-inference/.venv-sglang/bin/python",
            "sglang[diffusion]==0.5.18",
        ),
    ]


@pytest.mark.parametrize(
    "python_request,install_fails",
    [
        (None, False),
        ("version", False),
        ("range", False),
        ("path", False),
        ("version", True),
    ],
)
def test_standalone_environment_init_is_repeatable(
    tmp_path, standalone_environment_profile, python_request, install_fails
) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("environment lifecycle regression requires uv")
    major, minor = sys.version_info[:2]
    requests = {
        "version": f"{major}.{minor}",
        "range": f">={major}.{minor},<{major}.{minor + 1}",
        "path": sys.executable,
    }
    profile = replace(standalone_environment_profile, python=sys.executable)

    class PackageExecutor(LocalExecutor):
        def __init__(self):
            self.commands = []
            self.install_attempts = 0

        def run(self, command, *, check=True):
            self.commands.append(command)
            # Exercise real uv creation/discovery without downloading SGLang.
            if command.argv[1:3] == ("pip", "install"):
                self.install_attempts += 1
                if install_fails and self.install_attempts == 1:
                    raise CommandError(1, "simulated package download failure")
                return CommandResult(0, "packages ready\n")
            return super().run(command, check=check)

    executor = PackageExecutor()
    manager = UvEnvironmentManager(executor, uv_executable=uv)
    if install_fails:
        with pytest.raises(CommandError, match="simulated package download failure"):
            manager.prepare(profile, project_dir=str(tmp_path))
    else:
        manager.prepare(profile, project_dir=str(tmp_path))
    environment_path = tmp_path / profile.path
    config_stat = (environment_path / "pyvenv.cfg").stat()
    installed_module = environment_path / f"lib/python{major}.{minor}/site-packages/env_probe.py"
    installed_module.write_text("VALUE = 'preserved'\n", encoding="utf-8")

    result = manager.prepare(
        replace(profile, python=requests.get(python_request)),
        project_dir=str(tmp_path),
    )

    assert result.stdout == "packages ready\n"
    assert executor.install_attempts == 2
    assert sum(command.argv[1] == "venv" for command in executor.commands) == 1
    assert (environment_path / "pyvenv.cfg").stat().st_mtime_ns == config_stat.st_mtime_ns
    probe = executor.run(
        Command(
            (
                str(environment_path / "bin/python"),
                "-c",
                "import env_probe; print(env_probe.VALUE)",
            )
        )
    )
    assert probe.stdout == "preserved\n"


@pytest.mark.parametrize(
    "actual,requested",
    [
        (CommandResult(1, stderr="broken interpreter"), None),
        (
            CommandResult(0, "/usr/bin/python3.11\n"),
            CommandResult(0, "/usr/bin/python3.12\n"),
        ),
        (
            CommandResult(0, "/usr/bin/python3.11\n"),
            CommandResult(1, stderr="Python not found"),
        ),
    ],
)
def test_standalone_environment_rejects_broken_or_incompatible_python(
    standalone_environment_profile, actual, requested
) -> None:
    results = [CommandResult(0), actual]
    if requested is not None:
        results.append(requested)
    executor = ResultExecutor(*results)
    with pytest.raises(EnvironmentError, match="environment.*choose a different"):
        UvEnvironmentManager(executor).prepare(
            standalone_environment_profile,
            project_dir="/opt/rlinf-inference",
        )
    assert not executor.results
    assert not any(command.argv[1] in {"venv", "pip"} for command in executor.commands)


def test_project_manager_clones_and_fetches_only_when_missing() -> None:
    repository = "https://example.com/project.git"
    executor = ResultExecutor(
        CommandResult(0),
        CommandResult(1),
        CommandResult(0),
        CommandResult(0, f"{repository}\n"),
        CommandResult(1),
        CommandResult(0),
        CommandResult(0),
        CommandResult(0, "resolved\nresolved\n"),
    )

    ProjectManager(executor).prepare(
        repository=repository,
        revision="abc1234",
        project_dir="/opt/project",
    )

    argv = [command.argv for command in executor.commands]
    assert ("git", "clone", "--no-checkout", repository, "/opt/project") in argv
    assert ("git", "-C", "/opt/project", "fetch", "origin", "abc1234") in argv


def test_project_manager_reads_gitlink_from_requested_commit(tmp_path) -> None:
    executor = LocalExecutor()
    project = str(tmp_path / "deploy")

    def git(*arguments):
        return executor.run(Command(("git", "-C", project, *arguments)))

    executor.run(Command(("git", "init", project)))
    for revision in (INFERENCE_REVISION, "b" * 40):
        git(
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{revision},third_party/embodiinfer",
        )
        git(
            "-c",
            "user.name=Deploy test",
            "-c",
            "user.email=deploy@example.com",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-m",
            "Pin inference",
        )
        if revision == INFERENCE_REVISION:
            deploy_commit = git("rev-parse", "HEAD").stdout.strip()

    manager = ProjectManager(executor)
    assert (
        manager.submodule_revision(project_dir=project, revision=deploy_commit, path="third_party/embodiinfer")
        == INFERENCE_REVISION
    )
    assert manager.submodule_revision(project_dir=project, revision="HEAD", path="third_party/embodiinfer") == "b" * 40
    assert not (tmp_path / "deploy/third_party/embodiinfer").exists()


@pytest.mark.parametrize(
    "entry",
    [
        "",
        f"100644 blob {INFERENCE_REVISION}\tthird_party/embodiinfer\n",
        f"040000 tree {INFERENCE_REVISION}\tthird_party/embodiinfer\n",
        f"160000 commit {INFERENCE_REVISION}\tother/path\n",
        "160000 commit main\tthird_party/embodiinfer\n",
    ],
)
def test_project_manager_rejects_missing_or_invalid_gitlink(entry) -> None:
    executor = ResultExecutor(CommandResult(0, entry))
    with pytest.raises(SourceError, match="does not pin a valid submodule"):
        ProjectManager(executor).submodule_revision(
            project_dir="/opt/deploy", revision="deploy123", path="third_party/embodiinfer"
        )
    assert len(executor.commands) == 1


def test_local_executor_preserves_cwd_argv_environment_and_stdin(tmp_path) -> None:
    executor = LocalExecutor()

    cwd = executor.run(Command(("/bin/pwd",), cwd=str(tmp_path)))
    argument = executor.run(Command(("/usr/bin/printf", "%s\n", "argument with spaces")))
    environment = executor.run(
        Command(
            ("/usr/bin/printenv", "RLINF_TEST_VALUE"),
            environment={"RLINF_TEST_VALUE": "value with spaces"},
        )
    )
    stdin = executor.run(Command(("/bin/cat",), stdin="structured input\n"))

    assert cwd.stdout.strip() == str(tmp_path)
    assert argument.stdout == "argument with spaces\n"
    assert environment.stdout == "value with spaces\n"
    assert stdin.stdout == "structured input\n"


def test_local_executor_atomically_writes_private_text(tmp_path) -> None:
    target = tmp_path / "generated" / "adapter.json"
    executor = LocalExecutor()

    executor.write_text(str(target), '{"return_steps":50}\n', mode=0o600)

    assert target.read_text(encoding="utf-8") == '{"return_steps":50}\n'
    assert target.stat().st_mode & 0o777 == 0o600


def test_local_executor_atomically_reads_and_writes_private_bytes(tmp_path) -> None:
    target = tmp_path / "generated" / "source.tar.gz"
    executor = LocalExecutor()

    executor.write_bytes(str(target), b"\x1f\x8barchive", mode=0o600)

    assert executor.read_bytes(str(target)) == b"\x1f\x8barchive"
    assert target.stat().st_mode & 0o777 == 0o600


def test_local_executor_atomically_replaces_symlink(tmp_path) -> None:
    executor = LocalExecutor()
    current = tmp_path / "overlays" / "deploy" / "current"
    first = tmp_path / "releases" / "first"
    second = tmp_path / "releases" / "second"

    executor.replace_symlink(str(current), str(first))
    executor.replace_symlink(str(current), str(second))

    assert current.is_symlink()
    assert current.readlink() == second


def test_ssh_executor_reads_health_without_remote_script() -> None:
    client_socket, server_socket = socket.socketpair()
    opened = []

    class Transport:
        def is_active(self):
            return True

        def open_channel(self, kind, destination, source, timeout):
            opened.append((kind, destination, source, timeout))
            return client_socket

    def respond() -> None:
        try:
            request = b""
            while b"\r\n\r\n" not in request:
                request += server_socket.recv(4096)
            body = b'{"status":"ok"}'
            server_socket.sendall(
                b"HTTP/1.1 200 OK\r\n"
                + f"Content-Length: {len(body)}\r\n".encode()
                + b"Content-Type: application/json\r\n"
                + b"Connection: close\r\n\r\n"
                + body
            )
        finally:
            server_socket.close()

    thread = Thread(target=respond)
    thread.start()
    executor = object.__new__(SshExecutor)
    executor._client = SimpleNamespace(get_transport=lambda: Transport())
    executor._password = None

    response = executor.get_json(
        "http://127.0.0.1:8000/healthz",
        timeout_s=2.0,
    )
    thread.join(timeout=2.0)

    assert response == JsonHttpResponse(200, {"status": "ok"})
    assert opened == [("direct-tcpip", ("127.0.0.1", 8000), ("127.0.0.1", 0), 2.0)]
    assert not thread.is_alive()


def test_ssh_executor_posts_json_through_direct_tcp_channel() -> None:
    client_socket, server_socket = socket.socketpair()
    received = []

    class Transport:
        def is_active(self):
            return True

        def open_channel(self, kind, destination, source, timeout):
            return client_socket

    def respond() -> None:
        try:
            request = b""
            while b"\r\n\r\n" not in request:
                request += server_socket.recv(4096)
            headers, body = request.split(b"\r\n\r\n", 1)
            content_length = next(
                int(line.split(b":", 1)[1])
                for line in headers.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            )
            while len(body) < content_length:
                body += server_socket.recv(4096)
            received.append((headers, json.loads(body)))
            response = b'{"schema":"rlinf.control.result.v1","completed_steps":1}'
            server_socket.sendall(
                b"HTTP/1.1 200 OK\r\n"
                + f"Content-Length: {len(response)}\r\n".encode()
                + b"Content-Type: application/json\r\n"
                + b"Connection: close\r\n\r\n"
                + response
            )
        finally:
            server_socket.close()

    thread = Thread(target=respond)
    thread.start()
    executor = object.__new__(SshExecutor)
    executor._client = SimpleNamespace(get_transport=lambda: Transport())
    executor._password = None

    response = executor.request_json(
        "POST",
        "http://127.0.0.1:8100/v1/tasks",
        {"prompt": "抓取黄色格子"},
        timeout_s=3.0,
    )
    thread.join(timeout=2.0)

    assert response.status == 200
    headers, payload = received[0]
    assert headers.startswith(b"POST /v1/tasks HTTP/1.1\r\n")
    assert b"Content-Type: application/json" in headers
    assert payload == {"prompt": "抓取黄色格子"}
    assert not thread.is_alive()


def test_ssh_executor_sends_structured_command_stdin() -> None:
    writes = []
    invocations = []

    class Channel:
        def shutdown_write(self):
            writes.append("closed")

        def recv_exit_status(self):
            return 0

    class Input:
        channel = Channel()

        def write(self, value):
            writes.append(value)

        def flush(self):
            writes.append("flushed")

    class Output:
        channel = Channel()

        def __init__(self, value):
            self.value = value

        def read(self):
            return self.value

    class Client:
        def exec_command(self, invocation, timeout):
            invocations.append((invocation, timeout))
            return Input(), Output(b"ready\n"), Output(b"")

    executor = object.__new__(SshExecutor)
    executor._client = Client()
    executor._password = None
    executor.connection = SimpleNamespace(command_timeout_s=30.0)

    result = executor.run(Command(("worker", "argument with spaces"), stdin='{"task":"start"}\n'))

    assert result == CommandResult(0, "ready\n", "")
    assert invocations == [("worker 'argument with spaces'", 30.0)]
    assert writes == ['{"task":"start"}\n', "flushed", "closed"]


def test_ssh_executor_normalizes_health_channel_failure() -> None:
    class Transport:
        def is_active(self):
            return True

        def open_channel(self, kind, destination, source, timeout):
            raise Exception("Connect failed")

    executor = object.__new__(SshExecutor)
    executor._client = SimpleNamespace(get_transport=lambda: Transport())
    executor._password = None

    with pytest.raises(RuntimeError, match="SSH HTTP channel failed: Connect failed"):
        executor.get_json(
            "http://127.0.0.1:8000/healthz",
            timeout_s=2.0,
        )


@pytest.mark.parametrize("fail", [False, True])
def test_ssh_proxy_substitutes_endpoint_and_closes_on_failure(monkeypatch, fail):
    from embodirun.services.host.config.connection import ConnectionConfig

    events = []
    proxy = SimpleNamespace(close=lambda: events.append("proxy-closed"))

    def connect(**kwargs):
        assert kwargs["sock"] is proxy
        assert kwargs["hostname"] == "lab-node" and kwargs["port"] == 2222
        if fail:
            raise OSError("fixture connection failure")

    client = SimpleNamespace(
        load_system_host_keys=lambda: None,
        set_log_channel=lambda _: None,
        connect=connect,
        close=lambda: events.append("client-closed"),
    )

    def proxy_factory(command):
        assert shlex.split(command) == ["nc", "-X", "5", "-x", "127.0.0.1:1080", "lab-node", "2222"]
        return proxy

    monkeypatch.setitem(
        sys.modules,
        "paramiko",
        SimpleNamespace(
            SSHClient=lambda: client,
            ProxyCommand=proxy_factory,
        ),
    )
    executor = SshExecutor(
        ConnectionConfig(
            kind="ssh",
            host="lab-node",
            port=2222,
            username="user",
            proxy_command="nc -X 5 -x 127.0.0.1:1080 %h %p",
        )
    )
    if fail:
        with pytest.raises(RuntimeError, match="fixture connection failure"):
            executor._connect()
        assert events == ["client-closed", "proxy-closed"]
    else:
        assert executor._connect() is client
        executor.close()
        assert events == ["client-closed"]


def test_ssh_executor_does_not_leak_paramiko_transport_logs(
    monkeypatch,
    capsys,
) -> None:
    class Client:
        def load_system_host_keys(self):
            pass

        def set_log_channel(self, name):
            self.log_channel = name

        def connect(self, **_options):
            logging.getLogger(self.log_channel).error("Secsh channel 5 open FAILED: Connection refused")

    client = Client()
    monkeypatch.setitem(
        sys.modules,
        "paramiko",
        SimpleNamespace(SSHClient=lambda: client),
    )
    connection = load_config(EXAMPLE).nodes["jetson-agx-thor-232"].connection

    executor = SshExecutor(connection)

    assert executor._connect() is client
    assert capsys.readouterr().err == ""


def test_service_supervisor_uses_identity_checked_pid_lifecycle() -> None:
    service = build_plan(load_config(EXAMPLE)).services[0]
    executor = ResultExecutor(
        CommandResult(0, "running\t314\n", ""),
        CommandResult(0, "running\t314\n", ""),
        CommandResult(0, "stopped\n", ""),
    )
    supervisor = ServiceSupervisor(
        executor,
        python="/usr/bin/python3",
        agent_path="/opt/rlinf-deploy/src/embodirun/services/host/supervisor.py",
        run_root=".local/state/rlinf-deploy/run",
        log_root=".local/state/rlinf-deploy/logs",
    )

    started = supervisor.start(service)
    running = supervisor.status(service.service_id)
    stopped = supervisor.stop(service.service_id)

    assert started.state == running.state == "running"
    assert started.pid == running.pid == 314
    assert stopped.state == "stopped"
    assert all(
        command.argv[:2]
        == (
            "/usr/bin/python3",
            "/opt/rlinf-deploy/src/embodirun/services/host/supervisor.py",
        )
        for command in executor.commands
    )
    start_request = json.loads(executor.commands[0].stdin)
    assert start_request["environment"]["RLINF_DEPLOY_SERVICE_ID"] == "pi05-01"
    assert start_request["argv"][0] == "embodiinfer-http-serve"
    assert executor.commands[-1].timeout_s == 10.0


def test_service_environment_activates_venv_using_the_node_path(monkeypatch) -> None:
    import io
    from types import SimpleNamespace

    from embodirun.services.host import supervisor

    monkeypatch.setenv("PATH", "/node/bin:/usr/bin")
    monkeypatch.setenv("PYTHONHOME", "/unrelated/python")
    payload = {
        "argv": ["/env/sglang/bin/sglang"],
        "cwd": "/project",
        "environment": {
            "RLINF_DEPLOY_SERVICE_ID": "model",
            "VIRTUAL_ENV": "/env/sglang",
        },
    }
    monkeypatch.setattr(
        supervisor.sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO(json.dumps(payload).encode())),
    )
    _, _, environment = supervisor._read_start_request("model")
    assert environment["PATH"] == "/env/sglang/bin:/node/bin:/usr/bin"
    assert environment["VIRTUAL_ENV"] == "/env/sglang"
    assert "PYTHONHOME" not in environment


def test_service_supervisor_rejects_unsafe_service_id() -> None:
    supervisor = ServiceSupervisor(
        ResultExecutor(),
        python="/usr/bin/python3",
        agent_path="/opt/rlinf-deploy/src/embodirun/services/host/supervisor.py",
        run_root="run",
        log_root="logs",
    )

    with pytest.raises(SupervisorError, match="service IDs"):
        supervisor.status("../other-process")


def test_local_service_supervisor_runs_without_embedded_shell(tmp_path) -> None:
    supervisor = ServiceSupervisor(
        LocalExecutor(),
        python=sys.executable,
        agent_path=str(ROOT / "src/embodirun/services/host/supervisor.py"),
        run_root=str(tmp_path / "run"),
        log_root=str(tmp_path / "logs"),
    )
    service = ServiceSpec(
        service_id="test-sleeper",
        kind="model",
        node="local",
        environment_id="test",
        endpoint="http://127.0.0.1:1",
        health_endpoint="http://127.0.0.1:1/healthz",
        command=Command(("/bin/sleep", "30")),
    )

    started = supervisor.start(service)
    try:
        running = supervisor.status(service.service_id)
        assert started.state == running.state == "running"
        assert started.pid == running.pid
    finally:
        stopped = supervisor.stop(service.service_id)

    assert stopped.state == "stopped"


def test_duplicate_yaml_keys_are_rejected(tmp_path) -> None:
    config_path = tmp_path / "duplicate.yaml"
    config_path.write_text(
        "metadata:\n  name: one\n  name: two\n  deploy-commit: abc\nnodes: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="duplicate YAML key"):
        load_config(config_path)


def test_metadata_rejects_independent_inference_revision(tmp_path) -> None:
    config_path = tmp_path / "obsolete.yaml"
    config_path.write_text(
        EXAMPLE.read_text(encoding="utf-8").replace("metadata:\n", "metadata:\n  inference-commit: obsolete\n", 1),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="inference-commit"):
        load_config(config_path)


def test_unknown_node_reference_is_rejected(tmp_path) -> None:
    config_path = tmp_path / "unknown-node.yaml"
    config_path.write_text(
        EXAMPLE.read_text(encoding="utf-8").replace(
            "node: jetson-agx-thor-232",
            "node: missing-node",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="references unknown node"):
        load_config(config_path)


def test_two_robots_cannot_claim_the_same_device_port(tmp_path) -> None:
    config_path = tmp_path / "port-conflict.yaml"
    duplicate_robot = """
  so101-conflict:
    type: lerobot.so101
    node: jetson-agx-thor-232
    port: /dev/ttyACM0
"""
    config_path.write_text(
        EXAMPLE.read_text(encoding="utf-8").replace(
            "\nsensors:\n",
            f"{duplicate_robot}\nsensors:\n",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="share port"):
        load_config(config_path)


def test_control_and_model_ports_on_one_node_must_be_distinct(tmp_path) -> None:
    config_path = tmp_path / "service-port-conflict.yaml"
    config_path.write_text(
        EXAMPLE.read_text(encoding="utf-8").replace(
            "      port: 8100",
            "      port: 8000",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="share port"):
        load_config(config_path)


def test_cli_init_rejects_deploy_without_inference_gitlink(tmp_path, capsys) -> None:
    executor = FakeNodeExecutor(inference_commit=None)
    exit_code = main(
        ("--config", str(EXAMPLE), "--state-dir", str(tmp_path), "init"),
        executor_factory=lambda _node: executor,
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "does not pin a valid submodule" in captured.err
    assert "third_party/embodiinfer" in captured.err
    commands = [command.argv for command, _check in executor.commands]
    assert not any("--group" in argv for argv in commands)
    assert not any("-C" in argv and argv[2].endswith("/sources/inference") for argv in commands)
    state = StateStore(tmp_path / "thor-so101-pi05.json").load()
    assert state is not None
    assert state.inference_commit is None
    assert not state.nodes
    assert not state.environments


def test_cli_init_rejects_inconsistent_inference_gitlinks(tmp_path, capsys) -> None:
    config_path = two_node_model_config(tmp_path)

    def factory(node):
        return FakeNodeExecutor(inference_commit=(INFERENCE_REVISION if node.node_id == "jetson-worker" else "b" * 40))

    exit_code = main(
        ("--config", str(config_path), "--state-dir", str(tmp_path), "init"),
        executor_factory=factory,
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "different Inference revisions" in captured.err
    assert StateStore(tmp_path / "thor-so101-pi05.json").load() is None


def test_cli_init_preserves_resolved_revision_after_partial_failure(tmp_path, capsys) -> None:
    config_path = two_node_model_config(tmp_path)

    def factory(node):
        return FailingExecutor() if node.node_id == "jetson-worker" else FakeNodeExecutor()

    exit_code = main(
        ("--config", str(config_path), "--state-dir", str(tmp_path), "init"),
        executor_factory=factory,
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "connection refused" in captured.err
    state = StateStore(tmp_path / "thor-so101-pi05.json").load()
    assert state is not None
    assert state.inference_commit == INFERENCE_REVISION
    assert set(state.nodes) == {"jetson-agx-thor-232"}


def test_cli_init_then_up_uses_persisted_initialized_state(tmp_path, capsys) -> None:
    executors = []

    def factory(_node):
        executor = FakeNodeExecutor()
        executors.append(executor)
        return executor

    base_args = (
        "--config",
        str(EXAMPLE),
        "--state-dir",
        str(tmp_path),
    )
    init_exit_code = main((*base_args, "init"), executor_factory=factory)
    init_output = capsys.readouterr()

    assert init_exit_code == 0
    assert init_output.err == ""
    assert "Init deployment thor-so101-pi05" in init_output.out
    assert "2 environments ready" in init_output.out
    initialized = StateStore(tmp_path / "thor-so101-pi05.json").load()
    assert initialized is not None
    assert initialized.inference_commit == INFERENCE_REVISION
    assert {item.group for item in initialized.environments.values()} == {
        "pi05",
        "robot-so101",
    }
    init_argv = [command.argv for command, _check in executors[0].commands]
    deploy_project = initialized.nodes["jetson-agx-thor-232"].deploy_project
    inference_project = initialized.nodes["jetson-agx-thor-232"].inference_project
    assert (
        "/usr/bin/git",
        "-C",
        deploy_project,
        "ls-tree",
        load_config(EXAMPLE).metadata.deploy_commit,
        "--",
        "third_party/embodiinfer",
    ) in init_argv
    assert (
        "/usr/bin/git",
        "-C",
        inference_project,
        "checkout",
        "--detach",
        INFERENCE_REVISION,
    ) in init_argv
    assert any(argv[-2:] == ("--group", "pi05") for argv in init_argv)
    assert any(argv[-2:] == ("--group", "robot-so101") for argv in init_argv)
    assert executors[0].closed is True
    active_source = active_deploy_project("/home/operator/.local/share/rlinf-deploy/thor-so101-pi05")
    assert executors[0].symlinks == {
        active_source: ("/home/operator/.local/share/rlinf-deploy/thor-so101-pi05/sources/deploy")
    }

    up_exit_code = main((*base_args, "up"), executor_factory=factory)
    up_output = capsys.readouterr()

    assert up_exit_code == 0
    assert up_output.err == ""
    assert "Starting service" in up_output.out
    assert "2/2 services running" in up_output.out
    started = StateStore(tmp_path / "thor-so101-pi05.json").load()
    assert started is not None
    assert started.services["pi05-01"] == ServiceState(
        service_id="pi05-01",
        node="jetson-agx-thor-232",
        status="running",
        pid=321,
        endpoint="http://127.0.0.1:8000",
    )
    assert started.services["control-so101-1-runtime"] == ServiceState(
        service_id="control-so101-1-runtime",
        node="jetson-agx-thor-232",
        status="running",
        pid=321,
        endpoint="http://127.0.0.1:8100",
    )
    up_commands = [command for command, _check in executors[1].commands]
    start_commands = [
        command
        for command in up_commands
        if len(command.argv) > 2 and command.argv[1].endswith("/supervisor.py") and command.argv[2] == "start"
    ]
    assert len(start_commands) == 2
    assert all(command.argv[1].startswith(f"{active_source}/") for command in start_commands)
    start_requests = [json.loads(command.stdin) for command in start_commands]
    model_request = next(request for request in start_requests if request["argv"][0].endswith("embodiinfer-http-serve"))
    assert (
        model_request["argv"][0] == "/home/operator/.local/share/rlinf-deploy/thor-so101-pi05/"
        "sources/inference/.venv-embodiinfer/bin/embodiinfer-http-serve"
    )
    assert any(value.endswith("/thor-so101-pi05/generated/pi05-01.adapter.json") for value in model_request["argv"])
    control_request = next(
        request for request in start_requests if request["argv"][0].endswith("embodirun-control-serve")
    )
    assert control_request["cwd"] == active_source
    assert control_request["environment"]["PYTHONPATH"].endswith("/overlays/deploy/current/src")
    adapter_path, (adapter_content, adapter_mode) = next(
        item for item in executors[1].files.items() if item[0].endswith(".adapter.json")
    )
    assert adapter_path.endswith("/generated/pi05-01.adapter.json")
    assert json.loads(adapter_content)["return_steps"] == 50
    assert adapter_mode == 0o600
    control_path, (control_content, control_mode) = next(
        item for item in executors[1].files.items() if item[0].endswith(".control.json")
    )
    assert control_path.endswith("/generated/control-so101-1-runtime.control.json")
    assert json.loads(control_content)["runtime_id"] == "so101-1-runtime"
    assert control_mode == 0o600
    assert executors[1].health_requests == [
        ("http://127.0.0.1:8000/healthz", 2.0),
        ("http://127.0.0.1:8100/healthz", 2.0),
    ]
    assert "123456" not in up_output.out


def test_cli_routes_prompt_to_selected_runtime(tmp_path, capsys) -> None:
    state_dir = tmp_path / "state"
    base_args = (
        "--config",
        str(EXAMPLE),
        "--state-dir",
        str(state_dir),
    )
    assert main((*base_args, "init"), executor_factory=lambda _node: FakeNodeExecutor()) == 0
    capsys.readouterr()
    assert (
        main(
            (*base_args, "up"),
            executor_factory=lambda _node: ServiceReadinessExecutor(),
        )
        == 0
    )
    capsys.readouterr()
    executor = RuntimeExecutor()

    exit_code = main(
        (
            *base_args,
            "run",
            "--runtime",
            "so101-1-runtime",
            "--prompt",
            "把红色积木放进盒子",
        ),
        executor_factory=lambda _node: executor,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert "Run deployment thor-so101-pi05" in captured.out
    assert "Executing control task" in captured.out
    assert "Completed 1 chunk(s), 10 action(s) each" in captured.out
    method, url, payload, timeout_s = executor.requests[-1]
    assert method == "POST"
    assert url == "http://127.0.0.1:8100/v1/tasks"
    assert payload["schema"] == "rlinf.control.task.v1"
    assert payload["prompt"] == "把红色积木放进盒子"
    assert payload["max_steps"] == 1
    assert payload["chunk_steps"] == 10
    assert payload["control_hz"] == 5.0
    assert payload["inference_timeout_s"] == 60.0
    assert "robot" not in payload
    assert "inputs" not in payload
    assert "model_endpoint" not in payload
    assert timeout_s == pytest.approx(91.8)
    assert executor.closed is True


def test_cli_sync_uploads_deploy_overlay_without_touching_inference(tmp_path, capsys) -> None:
    state_dir = tmp_path / "state"
    base_args = (
        "--config",
        str(EXAMPLE),
        "--state-dir",
        str(state_dir),
    )
    assert main((*base_args, "init"), executor_factory=lambda _node: FakeNodeExecutor()) == 0
    capsys.readouterr()
    executor = SyncExecutor()

    exit_code = main(
        (*base_args, "sync", "--target", "deploy", "--source", str(ROOT)),
        executor_factory=lambda _node: executor,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert "inference" in captured.out
    assert "services were not restarted" in captured.out
    assert "restart control services" in captured.out
    commands = [command for command, _check in executor.commands]
    assert all(command.argv[:2] != ("sh", "-c") for command in commands)
    assert not any("inference" in argument for command in commands for argument in command.argv)
    assert not any("sync" in command.argv for command in commands)
    assert any(command.argv[0] == "tar" for command in commands)
    assert len(executor.symlinks) == 1
    current, release = next(iter(executor.symlinks.items()))
    assert current.endswith("/overlays/deploy/current")
    assert "/overlays/deploy/releases/" in release
    uploaded = next(content for path, (content, _mode) in executor.files.items() if path.endswith(".tar.gz"))
    with tarfile.open(fileobj=BytesIO(uploaded), mode="r:gz") as archive:
        names = set(archive.getnames())
    assert "src/embodirun/services/control/server.py" in names
    assert {"pyproject.toml", "uv.lock", "README.md"} <= names
    assert "integrations/sglang_pi05/pyproject.toml" in names
    assert executor.closed is True


def test_cli_sync_updates_only_deploy_dependencies_when_lock_changes(tmp_path, capsys) -> None:
    state_dir = tmp_path / "state"
    base_args = (
        "--config",
        str(EXAMPLE),
        "--state-dir",
        str(state_dir),
    )
    assert main((*base_args, "init"), executor_factory=lambda _node: FakeNodeExecutor()) == 0
    capsys.readouterr()
    executor = SyncExecutor(dependencies_changed=True)

    exit_code = main(
        (*base_args, "sync", "--source", str(ROOT)),
        executor_factory=lambda _node: executor,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    uv_commands = [
        command for command, _check in executor.commands if "sync" in command.argv and "--group" in command.argv
    ]
    assert len(uv_commands) == 1
    assert uv_commands[0].argv[-2:] == ("--group", "robot-so101")
    assert uv_commands[0].environment == {
        "UV_PROJECT_ENVIRONMENT": (
            "/home/operator/.local/share/rlinf-deploy/thor-so101-pi05/sources/deploy/.venv-robot-so101"
        )
    }
    assert "dependencies updated" in captured.out


def test_cli_sync_has_no_obsolete_binding_worker_probe(tmp_path, capsys) -> None:
    state_dir = tmp_path / "state"
    base_args = (
        "--config",
        str(EXAMPLE),
        "--state-dir",
        str(state_dir),
    )
    assert main((*base_args, "init"), executor_factory=lambda _node: FakeNodeExecutor()) == 0
    capsys.readouterr()
    executor = SyncExecutor(worker_running=True)

    exit_code = main(
        (*base_args, "sync", "--source", str(ROOT)),
        executor_factory=lambda _node: executor,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert not any(command.argv[:2] == ("pgrep", "-f") for command, _check in executor.commands)
    assert executor.closed is True


def test_cli_down_can_stop_control_without_stopping_inference(tmp_path, capsys) -> None:
    state_dir = tmp_path / "state"
    base_args = (
        "--config",
        str(EXAMPLE),
        "--state-dir",
        str(state_dir),
    )
    assert main((*base_args, "init"), executor_factory=lambda _node: FakeNodeExecutor()) == 0
    capsys.readouterr()
    assert (
        main(
            (*base_args, "up"),
            executor_factory=lambda _node: ServiceReadinessExecutor(),
        )
        == 0
    )
    capsys.readouterr()
    executor = DownExecutor()

    exit_code = main(
        (*base_args, "down", "--target", "control"),
        executor_factory=lambda _node: executor,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "1/1 control services stopped" in captured.out
    state = StateStore(state_dir / "thor-so101-pi05.json").load()
    assert state is not None
    assert state.services["control-so101-1-runtime"].status == "stopped"
    assert state.services["pi05-01"].status == "running"
    assert len(executor.commands) == 1


def test_cli_sync_requires_control_services_to_be_stopped(tmp_path, capsys) -> None:
    state_dir = tmp_path / "state"
    base_args = (
        "--config",
        str(EXAMPLE),
        "--state-dir",
        str(state_dir),
    )
    assert main((*base_args, "init"), executor_factory=lambda _node: FakeNodeExecutor()) == 0
    capsys.readouterr()
    assert (
        main(
            (*base_args, "up"),
            executor_factory=lambda _node: ServiceReadinessExecutor(),
        )
        == 0
    )
    capsys.readouterr()

    exit_code = main(
        (*base_args, "sync", "--source", str(ROOT)),
        executor_factory=lambda _node: pytest.fail("must not contact the node"),
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "down --target control" in captured.err


def test_cli_sync_allows_unrelated_config_change_after_init(tmp_path, capsys) -> None:
    config_path = tmp_path / "deployment.yaml"
    config_path.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    state_dir = tmp_path / "state"
    base_args = (
        "--config",
        str(config_path),
        "--state-dir",
        str(state_dir),
    )
    assert main((*base_args, "init"), executor_factory=lambda _node: FakeNodeExecutor()) == 0
    capsys.readouterr()
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "\n# local runtime edit\n",
        encoding="utf-8",
    )
    executor = SyncExecutor()

    exit_code = main(
        (*base_args, "sync", "--source", str(ROOT)),
        executor_factory=lambda _node: executor,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert executor.symlinks


def test_cli_run_rejects_any_config_change_after_control_started(tmp_path, capsys) -> None:
    config_path = tmp_path / "deployment.yaml"
    config_path.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    state_dir = tmp_path / "state"
    base_args = (
        "--config",
        str(config_path),
        "--state-dir",
        str(state_dir),
    )
    assert main((*base_args, "init"), executor_factory=lambda _node: FakeNodeExecutor()) == 0
    capsys.readouterr()
    assert (
        main(
            (*base_args, "up"),
            executor_factory=lambda _node: ServiceReadinessExecutor(),
        )
        == 0
    )
    capsys.readouterr()
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "\n# local runtime edit\n",
        encoding="utf-8",
    )

    exit_code = main(
        (
            *base_args,
            "run",
            "--runtime",
            "so101-1-runtime",
            "--prompt",
            "move",
        ),
        executor_factory=lambda _node: RuntimeExecutor(),
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "configuration changed since init" in captured.err


def test_cli_run_rejects_changed_model_endpoint(tmp_path, capsys) -> None:
    config_path = tmp_path / "deployment.yaml"
    config_path.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    state_dir = tmp_path / "state"
    base_args = (
        "--config",
        str(config_path),
        "--state-dir",
        str(state_dir),
    )
    assert main((*base_args, "init"), executor_factory=lambda _node: FakeNodeExecutor()) == 0
    capsys.readouterr()
    assert (
        main(
            (*base_args, "up"),
            executor_factory=lambda _node: ServiceReadinessExecutor(),
        )
        == 0
    )
    capsys.readouterr()
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("port: 8000", "port: 8001"),
        encoding="utf-8",
    )

    exit_code = main(
        (
            *base_args,
            "run",
            "--runtime",
            "so101-1-runtime",
            "--prompt",
            "move",
        ),
        executor_factory=lambda _node: pytest.fail("must not contact the node"),
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "configuration changed since init" in captured.err


def test_cli_rejects_chunk_steps_above_binding_maximum(capsys) -> None:
    exit_code = main(
        (
            "--config",
            str(EXAMPLE),
            "run",
            "--runtime",
            "so101-1-runtime",
            "--prompt",
            "move",
            "--chunk-steps",
            "51",
        ),
        executor_factory=lambda _node: pytest.fail("must not contact the node"),
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "--chunk-steps 51" in captured.err
    assert "maximum 50" in captured.err


def test_runtime_inputs_define_model_image_fields(tmp_path) -> None:
    config_path = tmp_path / "custom-runtime-input.yaml"
    config_path.write_text(
        EXAMPLE.read_text(encoding="utf-8").replace(
            "      observation.images.front: front-camera",
            "      observation.images.overhead: front-camera",
        ),
        encoding="utf-8",
    )

    plan = build_plan(load_config(config_path))

    adapter_config = json.loads(plan.services[0].adapter_config_json)
    assert adapter_config["image_fields"] == [
        "observation.images.overhead",
        "observation.images.wrist",
    ]


def test_embodiinfer_model_image_keys_are_forwarded_to_the_policy_adapter(tmp_path) -> None:
    config_path = tmp_path / "embodiinfer-image-keys.yaml"
    config_path.write_text(
        EXAMPLE.read_text(encoding="utf-8").replace(
            "    source: /models/pi05_so101\n",
            "    source: /models/pi05_so101\n"
            "    image_keys:\n"
            "      observation.images.front: observation.images.base_0_rgb\n"
            "      observation.images.wrist: observation.images.left_wrist_0_rgb\n",
        ),
        encoding="utf-8",
    )

    plan = build_plan(load_config(config_path))

    adapter_config = json.loads(plan.services[0].adapter_config_json)
    assert adapter_config["image_keys"] == {
        "observation.images.front": "observation.images.base_0_rgb",
        "observation.images.wrist": "observation.images.left_wrist_0_rgb",
    }


def test_cli_runtime_surfaces_remote_binding_error(tmp_path, capsys) -> None:
    state_dir = tmp_path / "state"
    base_args = (
        "--config",
        str(EXAMPLE),
        "--state-dir",
        str(state_dir),
    )
    assert main((*base_args, "init"), executor_factory=lambda _node: FakeNodeExecutor()) == 0
    capsys.readouterr()
    assert (
        main(
            (*base_args, "up"),
            executor_factory=lambda _node: ServiceReadinessExecutor(),
        )
        == 0
    )
    capsys.readouterr()
    executor = RuntimeExecutor(
        JsonHttpResponse(
            500,
            error_payload("joint step exceeds 12 degrees"),
        )
    )

    exit_code = main(
        (
            *base_args,
            "run",
            "--runtime",
            "so101-1-runtime",
            "--prompt",
            "move",
        ),
        executor_factory=lambda _node: executor,
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "joint step exceeds 12 degrees" in captured.err
    assert executor.requests[-1][0:2] == (
        "POST",
        "http://127.0.0.1:8100/v1/tasks",
    )


@pytest.mark.parametrize(
    ("process_state", "health_exit_code", "error"),
    [
        ("stale", 1, "exited before becoming ready"),
        ("running", 1, "did not become healthy"),
    ],
)
def test_cli_up_fails_until_service_is_healthy(
    tmp_path,
    capsys,
    process_state,
    health_exit_code,
    error,
) -> None:
    state_dir = tmp_path / "state"
    base_args = (
        "--config",
        str(EXAMPLE),
        "--state-dir",
        str(state_dir),
    )
    assert main((*base_args, "init"), executor_factory=lambda _node: FakeNodeExecutor()) == 0
    capsys.readouterr()
    executor = ServiceReadinessExecutor(
        process_state=process_state,
        health_exit_code=health_exit_code,
    )

    exit_code = main(
        (*base_args, "up", "--wait-timeout", "0.01"),
        executor_factory=lambda _node: executor,
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert error in captured.err
    assert "/logs/pi05-01.log" in captured.err
    state = StateStore(state_dir / "thor-so101-pi05.json").load()
    assert state is not None
    assert state.services["pi05-01"] == ServiceState(
        service_id="pi05-01",
        node="jetson-agx-thor-232",
        status="failed",
        pid=321,
        endpoint="http://127.0.0.1:8000",
    )
    assert executor.closed is True


def test_cli_probe_only_checks_connectivity_and_does_not_write_state(tmp_path, capsys) -> None:
    executors = []

    def factory(_node):
        executor = FakeNodeExecutor()
        executors.append(executor)
        return executor

    exit_code = main(
        (
            "--config",
            str(EXAMPLE),
            "--state-dir",
            str(tmp_path),
            "probe",
        ),
        executor_factory=factory,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert "Probe deployment thor-so101-pi05" in captured.out
    assert "jetson-agx-thor-232: Reachable" in captured.out
    assert "1/1 nodes ready" in captured.out
    assert len(executors[0].commands) == 8
    assert all(command.argv[:2] != ("sh", "-c") for command, _ in executors[0].commands)
    assert list(tmp_path.iterdir()) == []


def test_cli_probe_continues_after_one_node_is_unreachable(tmp_path, capsys) -> None:
    config_path = tmp_path / "two-nodes.yaml"
    config_path.write_text(
        EXAMPLE.read_text(encoding="utf-8").replace(
            "\nrobots:\n",
            "\n  offline-node:\n    type: test.node\n    connection:\n      type: local\n\nrobots:\n",
        ),
        encoding="utf-8",
    )

    def factory(node):
        if node.node_id == "offline-node":
            return FailingExecutor()
        return FakeNodeExecutor()

    exit_code = main(
        ("--config", str(config_path), "probe"),
        executor_factory=factory,
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "jetson-agx-thor-232: Reachable" in captured.out
    assert "offline-node: Failed: connection refused" in captured.out
    assert "1/2 nodes ready" in captured.out


def test_cli_lifecycle_runs_different_nodes_concurrently(tmp_path, capsys) -> None:
    config_path = two_node_model_config(tmp_path)
    state_dir = tmp_path / "state"
    arguments = ("--config", str(config_path), "--state-dir", str(state_dir))

    init_barrier = Barrier(2)
    assert (
        main(
            (*arguments, "init"),
            executor_factory=lambda _node: ConcurrentInitExecutor(init_barrier),
        )
        == 0
    )
    init_output = capsys.readouterr()
    assert "2/2 nodes ready" in init_output.out

    up_barrier = Barrier(2)
    assert (
        main(
            (*arguments, "up"),
            executor_factory=lambda _node: ConcurrentUpExecutor(up_barrier),
        )
        == 0
    )
    up_output = capsys.readouterr()
    assert "2/2 nodes ready" in up_output.out
    assert "3/3 services running" in up_output.out

    down_barrier = Barrier(2)
    assert (
        main(
            (*arguments, "down"),
            executor_factory=lambda _node: ConcurrentDownExecutor(down_barrier),
        )
        == 0
    )
    down_output = capsys.readouterr()
    assert "2/2 nodes ready" in down_output.out
    assert "3/3 services stopped" in down_output.out


def test_cli_probe_runs_different_nodes_concurrently(tmp_path, capsys) -> None:
    config_path = two_node_model_config(tmp_path)
    barrier = Barrier(2)

    exit_code = main(
        ("--config", str(config_path), "probe"),
        executor_factory=lambda _node: ConcurrentInitExecutor(barrier),
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "2/2 nodes ready" in captured.out


def test_cli_up_requires_successful_matching_init(tmp_path, capsys) -> None:
    exit_code = main(
        (
            "--config",
            str(EXAMPLE),
            "--state-dir",
            str(tmp_path),
            "up",
        ),
        executor_factory=lambda _node: FakeNodeExecutor(),
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "not initialized" in captured.err


def test_cli_down_stops_services_after_configuration_changes(tmp_path, capsys) -> None:
    config_path = tmp_path / "deployment.yaml"
    config_path.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    state_dir = tmp_path / "state"
    base_args = ("--config", str(config_path), "--state-dir", str(state_dir))
    assert main((*base_args, "init"), executor_factory=lambda _node: FakeNodeExecutor()) == 0
    capsys.readouterr()
    assert (
        main(
            (*base_args, "up"),
            executor_factory=lambda _node: ServiceReadinessExecutor(),
        )
        == 0
    )
    capsys.readouterr()
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "\n# safety changed\n",
        encoding="utf-8",
    )
    executor = DownExecutor()

    exit_code = main(
        (*base_args, "down"),
        executor_factory=lambda _node: executor,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "2/2 services stopped" in captured.out
    stopped = StateStore(state_dir / "thor-so101-pi05.json").load()
    assert stopped is not None
    assert stopped.services["pi05-01"].status == "stopped"
    assert stopped.services["control-so101-1-runtime"].status == "stopped"
    stop_command = executor.commands[0][0]
    assert stop_command.argv[2] == "stop"
    assert stop_command.stdin is None
    assert executor.closed is True


def test_cli_up_rejects_config_changed_after_init(tmp_path, capsys) -> None:
    config_path = tmp_path / "deployment.yaml"
    config_path.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    state_dir = tmp_path / "state"
    arguments = (
        "--config",
        str(config_path),
        "--state-dir",
        str(state_dir),
    )

    def factory(_node):
        return FakeNodeExecutor()

    assert main((*arguments, "init"), executor_factory=factory) == 0
    capsys.readouterr()
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "\n# changed\n",
        encoding="utf-8",
    )

    exit_code = main((*arguments, "up"), executor_factory=factory)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "configuration changed since init" in captured.err


@pytest.mark.parametrize("inference_commit", ["def", None])
def test_state_store_round_trip_is_atomic_and_secret_free(tmp_path, inference_commit) -> None:
    state_path = tmp_path / "state" / "deployment.json"
    state = DeploymentState(
        name="lab",
        config_digest="1234",
        deploy_commit="abc",
        inference_commit=inference_commit,
        nodes={
            "node": NodeState(
                node_id="node",
                home="/home/user",
                root="/home/user/.local/share/rlinf-deploy/lab",
                deploy_project="/home/user/project/deploy",
                inference_project="/home/user/project/inference",
                platform="Linux",
                machine="aarch64",
                python="/usr/bin/python3",
                python_version="3.10.12",
            )
        },
        environments={
            "node:deploy:robot-so101": EnvironmentState(
                environment_id="node:deploy:robot-so101",
                node="node",
                project="deploy",
                group="robot-so101",
                path=".venv-robot-so101",
                status="ready",
            )
        },
        services={
            "pi05-01": ServiceState(
                service_id="pi05-01",
                node="node",
                status="running",
                pid=123,
                endpoint="http://127.0.0.1:8000",
            )
        },
    )
    store = StateStore(state_path)

    store.save(state)

    assert store.load() == state
    assert state_path.stat().st_mode & 0o777 == 0o600
    assert "password" not in state_path.read_text(encoding="utf-8").lower()


def test_state_store_rejects_unknown_format(tmp_path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text('{"version": 999}', encoding="utf-8")

    with pytest.raises(StateError, match="unsupported state version"):
        StateStore(state_path).load()
