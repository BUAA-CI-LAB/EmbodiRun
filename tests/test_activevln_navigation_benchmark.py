"""Navigation benchmark metrics must use executed trajectories and matched episodes."""

import base64
import copy
import hashlib
import json
import runpy
from email import policy as email_policy
from email.parser import BytesParser
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodirun.model_services.contracts import ImagePayload, PolicyAction, PolicyObservation
from embodirun.model_services.protocols.http import HttpResponse


@pytest.fixture
def runner(monkeypatch):
    folder = Path(__file__).parents[1] / "benchmarks/activevln-navigation"
    monkeypatch.syspath_prepend(str(folder))
    return runpy.run_path(str(folder / "benchmark.py"))


@pytest.mark.parametrize("namespace", ["embodiinfer", "vvla", "unknown"])
def test_navigation_client_negotiates_schema_and_preserves_observations(namespace):
    connect = runpy.run_path(str(Path(__file__).parents[1] / "benchmarks/activevln-navigation/http_client.py"))[
        "connect_client"
    ]
    image = b'\x89PNG\r\n{"schema":"embodiinfer.policy.step.v1"}\x00\xff'
    instruction = 'Follow the sign "embodiinfer.policy.step.v1".'

    class Transport:
        calls = []

        def request(self, method, url, *, headers, body, timeout_s, maximum_bytes):
            self.calls.append((method, url))
            assert headers["Authorization"] == "Bearer test-token"
            assert timeout_s == 240 and maximum_bytes > 0
            if url.endswith("/healthz"):
                payload = {"status": "ok"}
            elif url.endswith("/capabilities"):
                payload = {"schema": f"{namespace}.policy.capabilities.v1"}
            elif url.endswith("/v1/sessions"):
                metadata = json.loads(body)
                assert metadata["schema"] == f"{namespace}.policy.session.v1"
                assert metadata["metadata"] == {"schema": "embodiinfer.policy.session.v1"}
                payload = {"session_id": "session-1", "session_revision": 0}
            elif url.endswith("/steps"):
                mime = BytesParser(policy=email_policy.default).parsebytes(
                    f"Content-Type: {headers['Content-Type']}\r\n\r\n".encode() + body
                )
                metadata_part, image_part = list(mime.iter_parts())
                metadata = json.loads(metadata_part.get_payload(decode=True))
                assert metadata["schema"] == f"{namespace}.policy.step.v1"
                assert metadata["instruction"] == instruction
                assert metadata["state"] == {"schema": "embodiinfer.policy.step.v1"}
                assert image_part.get_payload(decode=True) == image
                assert headers["Idempotency-Key"] == "request-1"
                payload = {
                    "request_id": "request-1",
                    "session_id": "session-1",
                    "step_id": 0,
                    "session_revision": 1,
                    "action_space": "activevln.r2r.discrete.v1",
                    "actions": [{"type": "discrete_chunk", "values": {"rows": [[0, 0]]}}],
                }
            else:
                assert method == "DELETE" and body is None
                payload = {"ok": True}
            return HttpResponse(200, {}, json.dumps(payload).encode())

    transport = Transport()
    if namespace == "unknown":
        with pytest.raises(ValueError, match="unsupported.*schema"):
            connect("http://localhost:18767", "test-token", transport=transport)
        assert all(method == "GET" for method, _ in transport.calls)
        return
    client = connect("http://localhost:18767", "test-token", transport=transport)
    session = client.open_session(
        robot_id="habitat-sim",
        action_space="activevln.r2r.discrete.v1",
        metadata={"schema": "embodiinfer.policy.session.v1"},
    )
    result = client.step(
        PolicyObservation(
            session_id=session.session_id,
            request_id="request-1",
            step_id=0,
            instruction=instruction,
            state={"schema": "embodiinfer.policy.step.v1"},
            images=(ImagePayload("rgb", "image/png", image),),
        )
    )
    assert result.actions[0].values == {"rows": [[0, 0]]}
    client.close(session.session_id)


class FakeSimulator:
    def request(self, command):
        if command["op"] == "reset":
            self.steps = 0
            self.position = 0
            self.stopped = False
        else:
            self.steps += len(command["actions"])
            self.position += command["actions"].count(1)
            self.stopped = 0 in command["actions"]
        return {
            "png": base64.b64encode(f"live-position-{self.position}".encode()).decode(),
            "position": [self.position, 0, 0],
            "rotation": [0, 0, 0, 1],
            "habitat_sim_version": "fake",
            "stopped": self.stopped,
            "steps": self.steps,
            "distance_to_goal_m": 2.0,
            "shortest_path_m": 3.0,
            "path_length_m": float(self.position),
            "trajectory": [self.position],
        }


class FakeClient:
    def __init__(self, split=False, fail=False):
        self.requests, self.closed = [], []
        self.sessions = 0
        self.split, self.fail = split, fail

    def open_session(self, **kwargs):
        self.sessions += 1
        return SimpleNamespace(session_id=str(self.sessions))

    def close(self, session):
        self.closed.append(session)

    def step(self, request):
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("model OOM")
        assert request.images[0].data == f"live-position-{request.step_id}".encode()
        return SimpleNamespace(
            actions=[
                PolicyAction(
                    "discrete_chunk",
                    {
                        "parsed_valid": True,
                        "rows": ([[1, 25]] if request.step_id == 0 else [[0, 0]]) + [[-1, 0], [-1, 0]],
                        "tensor_batch": {
                            "index": request.state["benchmark_slot"] if self.split else request.step_id,
                            "observations": 2,
                        },
                    },
                )
            ],
            timing={},
        )


@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_batched_episodes_execute_own_actions_and_reset_histories(runner, tmp_path, batch_size):
    client = FakeClient()
    original_step = client.step

    def step(request):
        result = original_step(request)
        result.actions[0].values["tensor_batch"]["observations"] = batch_size
        return result

    client.step = step
    count = 2 * batch_size
    episodes = [{"episode_id": str(i), "instruction": {"instruction_text": "move and stop"}} for i in range(count)]
    results = list(
        runner["run_batched_episodes"](
            client, [FakeSimulator() for _ in range(batch_size)], episodes, tmp_path, tmp_path
        )
    )
    assert len(results) == count and all(r["success"] and r["turns"] == 2 for r in results)
    assert len(client.closed) == count
    assert [r.step_id for r in client.requests].count(0) == count
    assert all(r["trace"][1]["before_position"] == [1, 0, 0] for r in results)


@pytest.mark.parametrize("split,fail,message", [(True, False, "split across"), (False, True, "OOM")])
def test_batch_infrastructure_failure_never_becomes_success(runner, tmp_path, split, fail, message):
    client = FakeClient(split, fail)
    episodes = [{"episode_id": str(i), "instruction": {"instruction_text": "move"}} for i in range(2)]
    with pytest.raises(RuntimeError, match=message):
        list(runner["run_batched_episodes"](client, [FakeSimulator(), FakeSimulator()], episodes, tmp_path, tmp_path))
    assert len(client.closed) == 2
    for path in tmp_path.glob("episode-*.json"):
        result = json.loads(path.read_text())
        assert result["status"] == "failed" and "success" not in result


functions = runpy.run_path(str(Path(__file__).parents[1] / "benchmarks/activevln-navigation/metrics.py"))
expand = functions["primitive_actions"]
metrics = functions["episode_metrics"]
compare = functions["compare"]


def test_actions_expand_distance_and_angle_and_stop_ends_chunk():
    assert expand([[1, 75], [2, 30], [3, 45]]) == [1, 1, 1, 2, 2, 3, 3, 3]
    assert expand([[-1, 0], [0, 0], [1, 25]]) == [0]


def test_rxr_grammar_uses_thirty_degree_primitives():
    assert expand([[1, 75], [2, 60], [3, 90]], action_space="rxr") == [1, 1, 1, 2, 2, 3, 3, 3]
    assert expand([[2, 30], [0, 0], [3, 90]], action_space="rxr") == [2, 0]
    for angle in (15, 45):
        with pytest.raises(ValueError, match="RxR|RXR"):
            expand([[2, angle], [-1, 0], [-1, 0]], action_space="rxr")
    for angle in (60, 90):
        with pytest.raises(ValueError, match="R2R"):
            expand([[2, angle], [-1, 0], [-1, 0]])


def test_rxr_episode_configures_simulator_turns_and_counts_matching_primitives(runner, tmp_path):
    commands = []

    class Simulator(FakeSimulator):
        def request(self, command):
            commands.append(command)
            return super().request(command)

    class Client(FakeClient):
        def step(self, request):
            self.requests.append(request)
            return SimpleNamespace(
                actions=[PolicyAction("discrete_chunk", {"parsed_valid": True, "rows": [[2, 60], [3, 90], [0, 0]]})],
                timing={},
            )

    client = Client()
    episode = {"episode_id": "rxr", "instruction": {"instruction_text": "turn and stop"}}
    result = runner["run_episode"](client, Simulator(), episode, tmp_path, tmp_path, action_space="rxr")
    assert commands[0]["turn_degrees"] == 30
    assert commands[1]["actions"] == [2, 2, 3, 3, 3, 0]
    assert result["steps"] == 6 and result["success"] and result["termination"] == "stop"
    assert client.closed == ["1"]


@pytest.mark.parametrize(
    "rows",
    [
        [[1, 26], [-1, 0], [-1, 0]],
        [[4, 15], [-1, 0], [-1, 0]],
        [[-1, 0], [-1, 0], [-1, 0]],
        [[1, float("nan")], [0, 0], [-1, 0]],
    ],
)
def test_invalid_action_never_becomes_successful_stop(rows):
    with pytest.raises(ValueError):
        expand(rows)


def test_success_requires_stop_inside_radius_and_spl_uses_geodesic_reference():
    assert metrics(True, 2.99, 5.0, 10.0) == {"success": True, "spl": 0.5}
    assert metrics(False, 0.1, 5.0, 10.0) == {"success": False, "spl": 0.0}
    assert not metrics(True, 3.0, 5.0, 10.0)["success"]
    assert not metrics(True, None, 5.0, 10.0)["success"]


def report(count=48):
    return {
        "status": "complete",
        "dataset": "R2R",
        "manifest_sha256": "same",
        "protocol": {"limit": 500},
        "expected_episodes": count,
        "episodes": [
            {
                "episode_id": str(i),
                "status": "complete",
                "stopped": True,
                "distance_to_goal_m": 1.0,
                "shortest_path_m": 5.0,
                "path_length_m": 10.0,
                "success": True,
                "spl": 0.5,
            }
            for i in range(count)
        ],
    }


def test_paired_success_tolerance_and_incomplete_report_rejection():
    baseline = report()
    candidate = copy.deepcopy(baseline)
    for row in candidate["episodes"][:2]:
        row.update(stopped=False, success=False, spl=0.0)
    result = compare(baseline, candidate)
    assert result["success_rate_within_tolerance"]
    assert result["candidate_successes"] == 46
    assert result["lost_success_episode_ids"] == ["0", "1"]
    candidate["episodes"][2].update(stopped=False, success=False, spl=0.0)
    assert not compare(baseline, candidate)["success_rate_within_tolerance"]
    candidate["episodes"].pop()
    with pytest.raises(ValueError, match="48"):
        compare(baseline, candidate)


def test_comparison_rejects_changed_tasks_and_inconsistent_success_evidence():
    baseline = report()
    candidate = copy.deepcopy(baseline)
    candidate["manifest_sha256"] = "different"
    with pytest.raises(ValueError, match="conditions"):
        compare(baseline, candidate)
    candidate["manifest_sha256"] = "same"
    candidate["episodes"][0]["distance_to_goal_m"] = 4
    with pytest.raises(ValueError, match="evidence"):
        compare(baseline, candidate)


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("validation_split", [False, True])
def test_external_service_runs_closed_loop_and_records_calibration_overlap(
    runner, tmp_path, monkeypatch, fail, validation_split
):
    """A remote inference service must preserve live observations and paired metadata."""
    episodes = [{"episode_id": str(i), "instruction": {"instruction_text": "move"}} for i in range(1, 49)]
    manifest = {"episodes": episodes}
    if validation_split:
        manifest.update(dataset="r2r", split="val_unseen")
    (tmp_path / "selected-r2r-navigation-48.json").write_text(json.dumps(manifest))
    config = {"batch_size": 1, "dtype": "bfloat16"}
    ready = {
        "config": dict(
            config,
            precision_experiment={
                "calibration_episode_ids": [1],
                "calibration_dataset": {"name": "R2R", "split": "published_training_trajectories"},
            },
        )
    }
    for name, value in (("config.json", config), ("ready.json", ready)):
        (tmp_path / name).write_text(json.dumps(value))
    (tmp_path / "token").write_text("test-only-token")
    client = FakeClient(fail=fail)
    client.health = lambda: {}
    simulator = FakeSimulator()
    simulator.close = lambda: None
    globals_ = runner["main"].__globals__
    monkeypatch.setitem(globals_, "SimulatorProcess", lambda *args: simulator)
    monkeypatch.setitem(globals_, "connect_client", lambda *args, **kwargs: client)
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark.py",
            "--root",
            str(tmp_path),
            "--config",
            str(tmp_path / "config.json"),
            "--habitat-python",
            "/unused/python",
            "--scenes",
            str(tmp_path),
            "--output",
            str(tmp_path / "result"),
            "--endpoint",
            "http://127.0.0.1:12345",
            "--service-ready",
            str(tmp_path / "ready.json"),
            "--token-file",
            str(tmp_path / "token"),
            "--datasets",
            "r2r",
            "--episode-limit",
            "2",
            "--simulator-gpu",
            "0",
        ],
    )
    if fail:
        with pytest.raises(RuntimeError, match="model OOM"):
            runner["main"]()
        report = json.loads((tmp_path / "result/r2r.json").read_text())
        assert report["status"] == "infrastructure_failure"
        assert report["episodes"] == []
        assert len(client.closed) == 1
        return
    runner["main"]()
    report = json.loads((tmp_path / "result/r2r.json").read_text())
    assert report["status"] == "partial_smoke"
    assert report["successes"] == 2
    assert report["calibration_overlap_episode_ids"] == ([] if validation_split else ["1"])
    assert report["protocol"]["physical_gpu"] == "0"
    assert report["latency_reference_sha256"] is None
    assert len(client.closed) == 2


def test_large_comparison_uses_declared_denominator_and_rejects_truncation():
    baseline = report(1024)
    candidate = copy.deepcopy(baseline)
    candidate["episodes"][0].update(stopped=False, success=False, spl=0.0)
    result = compare(baseline, candidate)
    assert result["episodes"] == 1024
    assert result["candidate_sr"] == 1023 / 1024
    candidate["episodes"].pop()
    with pytest.raises(ValueError, match="1024"):
        compare(baseline, candidate)


def test_manifest_rejects_missing_and_duplicate_episodes(runner, tmp_path):
    path = tmp_path / "selected-rxr-navigation-1024.json"
    episodes = [{"episode_id": str(i)} for i in range(1024)]
    path.write_text(json.dumps({"episodes": episodes}))
    assert len(runner["load_selection"](tmp_path, "rxr", 1024)[1]) == 1024
    episodes[-1] = episodes[0]
    path.write_text(json.dumps({"episodes": episodes}))
    with pytest.raises(ValueError, match="duplicate"):
        runner["load_selection"](tmp_path, "rxr", 1024)
    path.write_text(json.dumps({"episodes": episodes[:-1]}))
    with pytest.raises(ValueError, match="exactly 1024"):
        runner["load_selection"](tmp_path, "rxr", 1024)


def test_scene_proportions_round_to_exact_total_and_sampling_is_reproducible():
    sample = runpy.run_path(str(Path(__file__).parents[1] / "benchmarks/activevln-navigation/prepare_selection.py"))[
        "stratified_sample"
    ]
    population = [
        {"episode_id": str(i), "scene_id": "a" if i < 1000 else "b" if i < 1700 else "c"} for i in range(2000)
    ]
    selected, quotas = sample(population, 1024, 42, "r2r")
    assert quotas == {"a": 512, "b": 358, "c": 154}
    assert len(selected) == len({ep["episode_id"] for ep in selected}) == 1024
    assert sample(population[::-1], 1024, 42, "r2r") == (selected, quotas)
    assert sample(population, 1024, 43, "r2r")[0] != selected
    with pytest.raises(ValueError, match="eligible population"):
        sample(population, 2001, 42, "r2r")


def test_scene_selection_retains_rare_scenes_and_rejects_insufficient_quota():
    sample = runpy.run_path(str(Path(__file__).parents[1] / "benchmarks/activevln-navigation/prepare_selection.py"))[
        "stratified_sample"
    ]
    population = [{"episode_id": str(i), "scene_id": "a" if i < 9 else "b"} for i in range(10)]
    selected, quotas = sample(population, 2, 42, "r2r")
    assert quotas == {"a": 1, "b": 1}
    assert {ep["scene_id"] for ep in selected} == {"a", "b"}
    with pytest.raises(ValueError, match="minimum-one"):
        sample(population, 1, 42, "r2r")


@pytest.mark.parametrize("mismatch", [None, "source", "navmesh"])
def test_reusing_reachability_preflight_checks_inputs_and_preserves_provenance(tmp_path, mismatch):
    prepare = runpy.run_path(str(Path(__file__).parents[1] / "benchmarks/activevln-navigation/prepare_selection.py"))[
        "prepare"
    ]
    episodes = [
        {"episode_id": str(i), "scene_id": "scene.glb", "instruction": {"instruction_text": "stop"}} for i in range(5)
    ]
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"episodes": episodes}))
    navmesh = tmp_path / "scene.navmesh"
    navmesh.write_bytes(b"checked navmesh")
    metadata = {
        "dataset": "r2r",
        "split": "val_unseen",
        "source": "/original/source.json",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "candidate_episodes": 5,
        "eligible_episodes": 4,
        "excluded_before_model_inference": [{"episode_id": "0"}],
        "navmesh_sha256": {"scene.glb": hashlib.sha256(navmesh.read_bytes()).hexdigest()},
        "habitat_sim_version": "measured",
    }
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps(metadata))
    if mismatch:
        (source if mismatch == "source" else navmesh).write_bytes(
            source.read_bytes() + b" " if mismatch == "source" else b"changed"
        )
        with pytest.raises(ValueError, match=mismatch):
            prepare(source, tmp_path, "r2r", 2, 42, preflight)
        return
    manifest = prepare(source, tmp_path, "r2r", 2, 42, preflight)
    assert manifest["source"] == "/original/source.json"
    assert manifest["habitat_sim_version"] == "measured"
    assert manifest["expected_episodes"] == 2
    assert "0" not in {ep["episode_id"] for ep in manifest["episodes"]}
    assert manifest["reachability_preflight_manifest_sha256"] == hashlib.sha256(preflight.read_bytes()).hexdigest()
    assert manifest == prepare(source, tmp_path, "r2r", 2, 42, preflight)


@pytest.mark.parametrize("mismatch", [None, "manifest", "capacity", "adapter"])
@pytest.mark.parametrize("datasets", [("r2r",), ("r2r", "rxr")])
def test_paired_reports_allow_output_paths_but_reject_changed_measurements(tmp_path, monkeypatch, mismatch, datasets):
    folder = Path(__file__).parents[1] / "benchmarks/activevln-navigation"
    monkeypatch.syspath_prepend(str(folder))
    summarize = runpy.run_path(str(folder / "summarize.py"))["paired_summary"]
    for split in datasets:
        for precision in ("bf16", "int8"):
            destination = tmp_path / f"{precision}-{split}"
            (destination / split).mkdir(parents=True)
            data = report(2)
            data["dataset"] = split
            service = {
                "environment": {"source_python_sha256": "frozen"},
                "allocator_config": "same",
                "navigation_manifest_sha256": "same",
                "adapter_sha256": "same",
                "config": {
                    "kv_pool_tokens": 65536,
                    "output_dir": str(destination),
                    "navigation_manifest": str(destination / "manifest.json"),
                },
            }
            if precision == "int8":
                service["config"]["precision_experiment"] = {"variant": "smooth_both"}
                if mismatch == "manifest":
                    service["navigation_manifest_sha256"] = "different"
                elif mismatch == "capacity":
                    service["config"]["kv_pool_tokens"] = 32768
                elif mismatch == "adapter":
                    service["adapter_sha256"] = "different"
            data["model_service"] = service
            (destination / f"{split}.json").write_text(json.dumps(data))
            (destination / "summary.json").write_text("{}")
            for episode in data["episodes"]:
                trace = dict(
                    episode,
                    start_position=[0, 0, 0],
                    start_rotation=[0, 0, 0, 1],
                    habitat_sim_version="fake",
                    trace=[{"image_sha256": "same initial pixels"}],
                )
                (destination / split / f"episode-{episode['episode_id']}.json").write_text(json.dumps(trace))
    if mismatch is None:
        result = summarize(tmp_path, datasets)
        assert set(result["datasets"]) == set(datasets)
        assert all(pair["paired_initial_observation_audits"] == 2 for pair in result["datasets"].values())
    else:
        with pytest.raises(ValueError, match="differ"):
            summarize(tmp_path, datasets)


@pytest.mark.parametrize("tamper", [None, "occupancy", "cache", "session", "grammar", "timing"])
def test_summary_audits_rxr_tensor_batches_and_counts_shared_model_time_once(tmp_path, monkeypatch, tamper):
    folder = Path(__file__).parents[1] / "benchmarks/activevln-navigation"
    monkeypatch.syspath_prepend(str(folder))
    summarize = runpy.run_path(str(folder / "summarize.py"))["summarize_stage"]
    stage = "split-v1-activevln-bf16-rxr-b2"
    destination = tmp_path / stage
    (destination / "rxr").mkdir(parents=True)
    model_timing = dict(prefill_ms=5.0, decode_ms=3.0, gpu_inference_ms=8.0, pure_inference_ms=8.0)
    episodes = []
    for slot in range(2):
        position = [slot, 0, 0]
        episode = dict(
            episode_id=str(slot),
            session_id=f"session-{slot}",
            benchmark_slot=slot,
            status="complete",
            success=True,
            spl=1.0,
            steps=3,
            stopped=True,
            start_position=position,
            start_rotation=[0, 0, 0, 1],
            shortest_path_m=1.0,
            path_length_m=0.0,
            distance_to_goal_m=1.0,
            termination="stop",
            wall_seconds=1.0,
            trace=[
                dict(
                    turn=0,
                    before_position=position,
                    timing=dict(e2e_ms=10.0, policy_ms=11.0 + slot, **model_timing),
                    http_roundtrip_ms=12.0 + slot,
                    model=dict(
                        rows=[[2, 60], [0, 0], [-1, 0]],
                        cache_length=12,
                        tensor_batch=dict(index=7, observations=2, capacity=2),
                    ),
                    executed=[dict(action=action, position=position, distance_to_goal_m=1.0) for action in [2, 2, 0]],
                )
            ],
        )
        if slot == 0 and tamper == "session":
            episode["session_id"] = "different"
        if slot == 0 and tamper == "timing":
            episode["trace"][0]["timing"]["e2e_ms"] = 11.0
        (destination / "rxr" / f"episode-{slot}.json").write_text(json.dumps(episode))
        episodes.append({key: value for key, value in episode.items() if key != "trace"})
    report = dict(
        dataset="RxR",
        manifest_sha256="fixed",
        protocol=dict(turn_degrees=30),
        status="complete",
        expected_episodes=2,
        episodes=episodes,
        model_service=dict(config=dict(action_space="r2r" if tamper == "grammar" else "rxr")),
    )
    (destination / "rxr.json").write_text(json.dumps(report))
    evidence = dict(
        index=7,
        observations=2,
        status="complete",
        latency_ms=10.0,
        model_timing_ms=model_timing,
        peak_allocated_bytes=100,
        peak_reserved_bytes=200,
        runtime=dict(hits=1),
        rows=[dict(slot=i, session_id=f"session-{i}", step_id=0, cache_length=0) for i in range(2)],
    )
    if tamper == "occupancy":
        evidence["rows"].pop()
    if tamper == "cache":
        evidence["rows"][0]["cache_length"] = 5
    (destination / "tensor-batches.jsonl").write_text(json.dumps(evidence) + "\n")
    if tamper:
        with pytest.raises(ValueError):
            summarize(tmp_path, stage)
        return
    result = summarize(tmp_path, stage)
    assert result["model_calls"] == 2 and result["model_invocations"] == 1
    assert result["model_inference_seconds"] == 0.01 and result["mean_batch_occupancy"] == 2
    assert result["calls_per_inference_second"] == 200 and result["amortized_ms_per_call"] == 5
    assert result["sr"] == 1 and result["spl"] == 1
