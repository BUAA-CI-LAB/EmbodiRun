import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodirun.bindings.unitree.go2.streamvln import StreamVLNGo2Mapper
from embodirun.robots.sensors.cameras import CameraFrame
from embodirun.robots.unitree.go2.navigation.discrete import NavigationCommandKind
from embodirun.services.inference import (
    HttpResponse,
    ImagePayload,
    PolicyObservation,
    SglangHttpClient,
    SglangHttpError,
    build_inference_client,
)
from embodirun.services.simulation.runtime import SimulationRuntime
from embodirun.simulators.habitat import HabitatAdapter, HabitatConfig
from embodirun.simulators.navigation import (
    NavigationObservation,
    NavigationTransition,
)


@pytest.mark.parametrize("selection", ["CUDA_HOME", "CUDA_PATH", "nvcc", "pip", "missing"])
def test_sglang_environment_preserves_or_discovers_toolkit(tmp_path, monkeypatch, selection):
    from embodirun.services.inference.backends import sglang

    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    if selection in {"CUDA_HOME", "CUDA_PATH"}:
        monkeypatch.setenv(selection, "/selected/cuda")
    monkeypatch.setattr(
        sglang.shutil,
        "which",
        lambda name: "/usr/local/cuda/bin/nvcc" if selection == "nvcc" else None,
    )
    monkeypatch.setattr(sglang.sysconfig, "get_path", lambda name: str(tmp_path))
    toolkit = tmp_path / "nvidia" / "cu13"
    if selection != "missing":
        (toolkit / "bin").mkdir(parents=True)
        (toolkit / "bin" / "nvcc").touch()

    sglang.prepare_sglang_environment()

    expected_home = "/selected/cuda" if selection == "CUDA_HOME" else str(toolkit) if selection == "pip" else None
    assert sglang.os.environ.get("CUDA_HOME") == expected_home
    assert sglang.os.environ.get("CUDA_PATH") == ("/selected/cuda" if selection == "CUDA_PATH" else None)


def test_sglang_lerobot_statistics_preserve_mean_std_math():
    pytest.importorskip("sglang.multimodal_gen")
    import torch

    from embodirun.services.inference.adapters.sglang.pi05 import _Statistics

    mean = torch.tensor([1.0, 2.0])
    std = torch.tensor([0.0, 0.5])
    statistics = _Statistics(mean, std, 1e-8)
    values = torch.tensor([[1.0, 3.0]])
    torch.testing.assert_close(statistics.normalize(values), (values - mean) / (std + 1e-8), rtol=0, atol=0)
    torch.testing.assert_close(statistics.denormalize(values), values * std + mean, rtol=0, atol=0)
    with pytest.raises(ValueError, match="dimensions"):
        statistics.normalize([1.0])
    with pytest.raises(ValueError, match="finite"):
        statistics.normalize([float("nan"), 1.0])
    for invalid in (1.0, [1.0], [float("nan"), 1.0]):
        with pytest.raises(ValueError, match="statistics"):
            statistics.denormalize(invalid)


def test_sglang_lerobot_statistics_preserve_quantiles_math():
    """QUANTILES must match lerobot.processor.normalize_processor bit for bit."""
    pytest.importorskip("sglang.multimodal_gen")
    import torch

    from embodirun.services.inference.adapters.sglang.pi05 import _Statistics

    q01 = torch.tensor([0.0, -2.0])
    width = torch.tensor([4.0, 1.0])  # q99 - q01
    statistics = _Statistics(q01, width, 1e-8, "QUANTILES")
    values = torch.tensor([[2.0, 3.0]])
    torch.testing.assert_close(
        statistics.normalize(values),
        2.0 * (values - q01) / width - 1.0,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        statistics.denormalize(values),
        (values + 1.0) * width / 2.0 + q01,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        statistics.denormalize(statistics.normalize(values)),
        values,
        rtol=0,
        atol=1e-5,
    )
    with pytest.raises(ValueError, match="dimensions"):
        statistics.normalize([1.0])


@pytest.mark.parametrize("delta_enabled", [False, True, None])
def test_sglang_lerobot_statistics_load_quantiles_checkpoint(tmp_path, delta_enabled):
    """QUANTILES loads from q01/q99; a disabled delta processor is a safe no-op."""
    pytest.importorskip("sglang.multimodal_gen")
    import torch
    from safetensors.torch import save_file

    from embodirun.services.inference.adapters.sglang.pi05 import _Statistics

    checkpoint = tmp_path / "snapshot"
    checkpoint.mkdir()
    save_file(
        {
            "observation.state.mean": torch.zeros(2),
            "observation.state.std": torch.ones(2),
            "observation.state.q01": torch.tensor([-1.0, -1.0]),
            "observation.state.q99": torch.tensor([1.0, 3.0]),
        },
        str(checkpoint / "statistics.safetensors"),
    )
    steps = [
        {
            "registry_name": "normalizer_processor",
            "config": {
                "norm_map": {"STATE": "QUANTILES", "VISUAL": "IDENTITY"},
                "features": {"observation.state": {"shape": [2]}},
                "eps": 1e-8,
            },
            "state_file": "statistics.safetensors",
        },
        {
            "registry_name": "delta_actions_processor",
            "config": {"enabled": delta_enabled},
        },
    ]
    (checkpoint / "policy_preprocessor.json").write_text(json.dumps({"steps": steps}))

    if delta_enabled is not False:
        # An enabled delta processor changes action semantics; refuse loudly.
        with pytest.raises(ValueError, match="explicitly disabled delta_actions_processor"):
            _Statistics.load(
                checkpoint,
                "preprocessor",
                "normalizer_processor",
                "observation.state",
                "STATE",
            )
        return

    stats = _Statistics.load(checkpoint, "preprocessor", "normalizer_processor", "observation.state", "STATE")
    assert stats.mode == "QUANTILES"
    # q01=[-1,-1], q99=[1,3] -> width=[2,4]; the midpoint maps to 0.
    torch.testing.assert_close(stats.normalize([0.0, 1.0]), torch.tensor([0.0, 0.0]), rtol=0, atol=0)


def test_sglang_lerobot_statistics_reject_unknown_normalization(tmp_path):
    """A mode outside the supported set must be rejected, not silently ignored."""
    pytest.importorskip("sglang.multimodal_gen")
    import torch
    from safetensors.torch import save_file

    from embodirun.services.inference.adapters.sglang.pi05 import _Statistics

    checkpoint = tmp_path / "snapshot"
    checkpoint.mkdir()
    save_file(
        {
            "observation.state.mean": torch.zeros(2),
            "observation.state.std": torch.ones(2),
        },
        str(checkpoint / "statistics.safetensors"),
    )
    (checkpoint / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "normalizer_processor",
                        "config": {
                            "norm_map": {"STATE": "MIN_MAX", "VISUAL": "IDENTITY"},
                            "features": {"observation.state": {"shape": [2]}},
                            "eps": 1e-8,
                        },
                        "state_file": "statistics.safetensors",
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="STATE in"):
        _Statistics.load(checkpoint, "preprocessor", "normalizer_processor", "observation.state", "STATE")


def test_sglang_lerobot_manifest_excludes_duplicate_processor_tensors(tmp_path):
    pytest.importorskip("sglang.multimodal_gen")
    import torch
    from safetensors.torch import save_file
    from sglang.multimodal_gen.configs.pipeline_configs.pi05 import Pi05PipelineConfig

    from embodirun.services.inference.adapters.sglang.pi05 import _LeRobotPolicyModel

    save_file(
        {"model.action_out_proj.bias": torch.zeros(32)},
        str(tmp_path / "model.safetensors"),
    )
    for name in ("policy_preprocessor", "policy_postprocessor"):
        save_file({"action.mean": torch.zeros(7)}, str(tmp_path / f"{name}.safetensors"))
    manifest = _LeRobotPolicyModel._inspect_checkpoint(str(tmp_path), Pi05PipelineConfig())
    assert manifest.safetensor_files == [str(tmp_path / "model.safetensors")]


@pytest.mark.parametrize("problem", [None, "normalization", "processor", "rename", "eps", "std", "shape"])
def test_sglang_lerobot_statistics_load_checkpoint_contract(tmp_path, problem):
    pytest.importorskip("sglang.multimodal_gen")
    import torch
    from safetensors.torch import save_file

    from embodirun.services.inference.adapters.sglang.pi05 import _Statistics

    checkpoint = tmp_path / "snapshot"
    checkpoint.mkdir()
    # Hugging Face snapshots legitimately link to blobs outside their directory.
    blob = tmp_path / "statistics.safetensors"
    std = [-1.0, 0.5] if problem == "std" else [0.0, 0.5]
    save_file(
        {
            "observation.state.mean": torch.ones(2),
            "observation.state.std": torch.tensor(std),
        },
        str(blob),
    )
    (checkpoint / "statistics.safetensors").symlink_to(blob)
    config = {
        "norm_map": {
            "STATE": "MIN_MAX" if problem == "normalization" else "MEAN_STD",
            "VISUAL": "IDENTITY",
        },
        "features": {"observation.state": {"shape": [3] if problem == "shape" else [2]}},
        "eps": 0.0 if problem == "eps" else 1e-8,
    }
    steps = [
        {
            "registry_name": "normalizer_processor",
            "config": config,
            "state_file": "statistics.safetensors",
        }
    ]
    if problem == "processor":
        steps.append({"registry_name": "custom_transform", "config": {}})
    if problem == "rename":
        steps.append(
            {
                "registry_name": "rename_observations_processor",
                "config": {"rename_map": {"a": "b"}},
            }
        )
    (checkpoint / "policy_preprocessor.json").write_text(json.dumps({"steps": steps}))
    if problem is not None:
        with pytest.raises(ValueError):
            _Statistics.load(
                checkpoint,
                "preprocessor",
                "normalizer_processor",
                "observation.state",
                "STATE",
            )
    else:
        stats = _Statistics.load(
            checkpoint,
            "preprocessor",
            "normalizer_processor",
            "observation.state",
            "STATE",
        )
        torch.testing.assert_close(stats.normalize([1.0, 2.0]), torch.tensor([0.0, 2.0]), rtol=0, atol=0)


def test_sglang_lerobot_pipeline_preserves_native_parallel_layout_rejection():
    pytest.importorskip("sglang.multimodal_gen")
    from embodirun.services.inference.adapters.sglang.pi05 import LeRobotPi05Pipeline

    pipeline = object.__new__(LeRobotPi05Pipeline)
    args = SimpleNamespace(
        pipeline_config=SimpleNamespace(prefix_parallel_strategy="tp", action_parallel_strategy="tp")
    )
    with pytest.raises(ValueError, match="TP layout"):
        pipeline.load_modules(args)


class FakeTransport:
    def __init__(self) -> None:
        self.calls = []

    def request(self, method, url, *, headers, body, timeout_s, maximum_bytes):
        self.calls.append((method, url, dict(headers), body, timeout_s, maximum_bytes))
        if url.endswith("/health"):
            return HttpResponse(200, {}, b"")
        if url.endswith("/v1/actions/metadata"):
            return HttpResponse(
                200,
                {},
                json.dumps({"object": "action.metadata"}).encode(),
            )
        request = json.loads(body)
        response = {
            "id": request["request_id"],
            "object": "action.generation",
            "model": "lerobot/pi05_so101",
            "data": [
                {
                    "action": {
                        "type": "continuous",
                        "values": [[float(index) for index in range(32)]],
                    }
                }
            ],
            "timings": {"total_ms": 12.5, "detail": {"ignored": True}},
        }
        return HttpResponse(200, {}, json.dumps(response).encode())


def test_sglang_command_selects_explicit_pipeline_and_optional_entrypoint() -> None:
    from embodirun.services.inference.backends.sglang.http import (
        sglang_server_command,
    )

    command = sglang_server_command(
        checkpoint="/models/pi05",
        bind="127.0.0.1",
        port=8000,
        executable="rlinf-sglang-serve",
        pipeline="LeRobotPi05Pipeline",
    )
    assert command[:3] == ("rlinf-sglang-serve", "serve", "/models/pi05")
    assert command[command.index("--pipeline-class-name") + 1] == "LeRobotPi05Pipeline"
    assert "--pipeline" not in command


def test_sglang_client_adapts_action_api_to_policy_contract() -> None:
    transport = FakeTransport()
    features = tuple(f"joint-{index}" for index in range(6))
    client = SglangHttpClient(
        "http://sglang:30000",
        token="secret",
        timeout_s=20,
        transport=transport,
        image_keys={"observation.images.front": "base_0_rgb"},
        state_fields=("joints", "gripper"),
        action_feature_names=features,
        parameters={"num_inference_steps": 5},
        runtime={"prefix_cache": "auto"},
    )

    assert client.health() == {"status": "ok"}
    assert client.capabilities()["object"] == "action.metadata"
    session = client.open_session(
        robot_id="so101-1",
        action_space="pi05.action_chunk.v1",
        metadata={"source": "robot"},
    )
    result = client.step(
        PolicyObservation(
            session_id=session.session_id,
            request_id="request-1",
            step_id=0,
            instruction="pick up the cube",
            state={"joints": [1, 2], "gripper": 3},
            images=(
                ImagePayload(
                    "observation.images.front",
                    "image/jpeg",
                    b"jpeg-bytes",
                ),
            ),
        )
    )

    method, url, headers, raw_body, timeout_s, _ = transport.calls[-1]
    request = json.loads(raw_body)
    assert (method, url, timeout_s) == (
        "POST",
        "http://sglang:30000/v1/actions/generations",
        20,
    )
    assert headers["Authorization"] == "Bearer secret"
    assert headers["Idempotency-Key"] == "request-1"
    assert request["input"]["task"] == "pick up the cube"
    observation = request["input"]["observation"]
    assert observation["state"] == [1.0, 2.0, 3.0]
    assert observation["session_id"] == session.session_id
    assert observation["session_revision"] == 0
    assert observation["step_id"] == 0
    assert observation["reset"] is True
    assert observation["metadata"] == {"source": "robot"}
    assert observation["images"] == {
        "base_0_rgb": {
            "base64": base64.b64encode(b"jpeg-bytes").decode(),
            "mime_type": "image/jpeg",
        }
    }
    assert request["parameters"] == {"num_inference_steps": 5}
    assert request["runtime"] == {
        "prefix_cache": "auto",
        "output_format": "list",
        "response_format": "envelope",
        "return_timing": True,
    }
    assert result.action_space == "pi05.action_chunk.v1"
    assert result.actions[0].kind == "action_chunk"
    assert result.actions[0].values["data"] == [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]
    assert result.actions[0].values["feature_names"] == list(features)
    assert result.timing == {"total_ms": 12.5}
    assert result.policy_revision == "lerobot/pi05_so101"


def test_sglang_client_keeps_session_sequence_and_marks_reset() -> None:
    transport = FakeTransport()
    client = SglangHttpClient("http://sglang:30000", transport=transport)
    session = client.open_session(robot_id="habitat", action_space="streamvln.v1")
    image = (ImagePayload("observation.images.rgb", "image/jpeg", b"rgb"),)

    with pytest.raises(SglangHttpError, match="expected step 0"):
        client.step(
            PolicyObservation(
                session.session_id,
                "request-wrong",
                1,
                "go forward",
                {},
                image,
            )
        )

    client.step(
        PolicyObservation(
            session.session_id,
            "request-1",
            0,
            "go forward",
            {},
            image,
        )
    )
    reset = client.reset(session.session_id, request_id="reset-1")
    assert reset.revision == 1
    client.step(
        PolicyObservation(
            session.session_id,
            "request-2",
            0,
            "go forward",
            {},
            image,
        )
    )
    request = json.loads(transport.calls[-1][3])
    assert request["input"]["observation"]["reset"] is True
    assert request["input"]["observation"]["session_revision"] == 1

    client.close(session.session_id)
    with pytest.raises(SglangHttpError, match="unknown SGLang session"):
        client.close(session.session_id)


def test_inference_factory_selects_backend_and_rejects_invalid_pair() -> None:
    client = build_inference_client(
        "http",
        "http://sglang:30000",
        {},
        backend="sglang",
        timeout_s=5,
    )
    assert isinstance(client, SglangHttpClient)

    with pytest.raises(
        ValueError,
        match=r"inference provider 'sglang' does not support 'wireless' transport",
    ):
        build_inference_client(
            "wireless",
            "wireless://policy",
            {},
            backend="sglang",
            timeout_s=5,
        )


def test_navigation_runtime_accepts_mock_sglang_action_responses(monkeypatch) -> None:
    """Contract-only test: neither Habitat nor a real SGLang model runs here."""
    from embodirun.simulators import navigation

    monkeypatch.setattr(
        navigation,
        "_rgb_array",
        lambda value, *, engine_name: value,
    )
    monkeypatch.setattr(
        navigation,
        "_encode_rgb",
        lambda value: CameraFrame("observation.images.rgb", "image/jpeg", b"rgb"),
    )

    class Environment:
        def __init__(self) -> None:
            self.closed = False

        def reset(self, *, seed):
            return NavigationObservation(
                rgb=object(),
                position=(0, 0, 0),
                rotation=(0, 0, 0, 1),
                instruction="walk to the painting",
            )

        def step(self, command):
            assert command.kind is NavigationCommandKind.MOVE_FORWARD
            return NavigationTransition(
                NavigationObservation(
                    rgb=object(),
                    position=(0, 0, -0.25),
                    rotation=(0, 0, 0, 1),
                    instruction="walk to the painting",
                ),
                reward=1,
                terminated=True,
            )

        def close(self):
            self.closed = True

    class StreamVlnTransport:
        def __init__(self) -> None:
            self.payload = None

        def request(
            self,
            method,
            url,
            *,
            headers,
            body,
            timeout_s,
            maximum_bytes,
        ):
            self.payload = json.loads(body)
            return HttpResponse(
                200,
                {},
                json.dumps(
                    {
                        "id": self.payload["request_id"],
                        "model": "streamvln",
                        "data": [
                            {
                                "action": {
                                    "type": "discrete",
                                    "values": [[1, 0.25]],
                                }
                            }
                        ],
                    }
                ).encode(),
            )

    environment = Environment()
    simulator = HabitatAdapter(
        HabitatConfig(
            simulator_id="habitat-demo",
            dataset=Path("episodes.json.gz"),
            scenes_dir=Path("scenes"),
            episode_id="episode-1",
        ),
        environment_factory=lambda config: environment,
    )
    transport = StreamVlnTransport()
    client = SglangHttpClient(
        "http://sglang:30000",
        transport=transport,
        image_keys={"observation.images.rgb": "rgb"},
        parameters={"action_horizon": 4},
    )
    runtime = SimulationRuntime(
        simulator,
        client,
        mapper=StreamVLNGo2Mapper(),
        chunk_steps=4,
    )

    outcome = runtime.run(instruction=None, max_policy_steps=1, seed=7)
    runtime.close()
    simulator.close()

    assert outcome.policy_steps == 1
    assert outcome.environment_steps == 1
    assert outcome.total_reward == 1
    assert outcome.terminated is True
    request = transport.payload
    assert request["input"]["task"] == "walk to the painting"
    assert request["input"]["observation"]["images"] == {
        "rgb": {
            "base64": base64.b64encode(b"rgb").decode(),
            "mime_type": "image/jpeg",
        }
    }
    assert request["input"]["observation"]["reset"] is True
    assert request["parameters"]["action_horizon"] == 4
    assert environment.closed is True
