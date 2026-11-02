"""Validate the model-independent execution and metric boundary."""

import base64
import importlib.util
import json
import runpy
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodirun.model_services.contracts import PolicyAction


@pytest.fixture
def metrics():
    path = Path(__file__).parents[1] / "benchmarks/streamvln-navigation/metrics.py"
    spec = importlib.util.spec_from_file_location("streamvln_navigation_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_streamvln_executes_metres_and_stops_at_stop(metrics):
    assert metrics.primitive_actions([[1, 0.25], [2, 15], [0, 0], [1, 0.25]]) == [1, 2, 0]


@pytest.mark.parametrize("rows", [[], [[1, 25]], [[2, 30]], [[4, 0]], [[1.5, 0.25]], [[1, float("nan")]]])
def test_streamvln_rejects_incompatible_action_grammars(metrics, rows):
    with pytest.raises(ValueError):
        metrics.primitive_actions(rows)


def test_success_requires_stop_strictly_within_three_metres(metrics):
    assert metrics.episode_metrics(True, 2.99, 4, 8) == {"success": True, "spl": 0.5}
    for stopped, distance in ((False, 0), (True, 3), (True, None)):
        assert metrics.episode_metrics(stopped, distance, 4, 8) == {"success": False, "spl": 0.0}


def test_summary_refuses_partial_navigation_results(metrics, tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "benchmarks/streamvln-navigation/summarize.py"
    monkeypatch.setitem(sys.modules, "metrics", metrics)
    spec = importlib.util.spec_from_file_location("streamvln_navigation_summary", path)
    summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary)
    report = tmp_path / "r2r.json"
    report.write_text(json.dumps({"status": "running", "expected_episodes": 256, "episodes": []}))
    with pytest.raises(ValueError, match="256 complete episodes"):
        summary.load_complete(report)


@pytest.fixture
def runner(metrics, monkeypatch):
    monkeypatch.setitem(sys.modules, "metrics", metrics)
    folder = Path(__file__).parents[1] / "benchmarks/streamvln-navigation"
    return runpy.run_path(str(folder / "benchmark.py"))


class PrimitiveSimulator:
    def request(self, command):
        if command["op"] == "reset":
            self.steps = 0
            frames, stopped = [b"frame-0"], False
        else:
            frames = []
            stopped = False
            for action in command["actions"]:
                self.steps += 1
                frames.append(f"frame-{self.steps}".encode())
                if action == 0:
                    stopped = True
                    break
        return dict(
            png=base64.b64encode(frames[-1]).decode(),
            frames=[base64.b64encode(f).decode() for f in frames],
            position=[self.steps, 0, 0],
            rotation=[0, 0, 0, 1],
            habitat_sim_version="fake",
            stopped=stopped,
            steps=self.steps,
            distance_to_goal_m=0 if stopped else 4,
            shortest_path_m=3,
            path_length_m=3,
            trajectory=[dict(position=[self.steps, 0, 0])],
        )


class RoundClient:
    def __init__(self, *, split=False, fail=False):
        self.sessions, self.requests, self.closed = set(), [], []
        self.count = self.round = self.admitted = 0
        self.lock = threading.Lock()
        self.split, self.fail = split, fail

    def open_session(self, **kwargs):
        self.count += 1
        identity = str(self.count)
        self.sessions.add(identity)
        return SimpleNamespace(session_id=identity)

    def close(self, identity):
        self.sessions.remove(identity)
        self.closed.append(identity)

    def step(self, request):
        with self.lock:
            self.requests.append(request)
            if self.fail:
                raise RuntimeError("model OOM")
            step = request.metadata["stream_step_id"]
            assert step == (0 if request.step_id == 0 else 3)
            assert [image.data for image in request.images] == (
                [b"frame-0"] if step == 0 else [b"frame-1", b"frame-2", b"frame-3"]
            )
            index = self.round + (request.state["benchmark_slot"] if self.split else 0)
            result = SimpleNamespace(
                actions=[
                    PolicyAction(
                        "action_chunk",
                        dict(
                            parsed_valid=True,
                            stream_step_id=step,
                            data=[[1, 0.25], [1, 0.25], [2, 15]] if step == 0 else [[0, 0]],
                            tensor_batch=dict(index=index, observations=len(self.sessions)),
                        ),
                    )
                ],
                timing={},
            )
            self.admitted += 1
            if self.admitted == len(self.sessions):
                self.round += 1
                self.admitted = 0
            return result


@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_batched_navigation_preserves_primitive_frames_refill_and_partial_tails(runner, tmp_path, batch_size):
    client = RoundClient()
    episodes = [
        dict(episode_id=str(i), instruction=dict(instruction_text="move then stop")) for i in range(2 * batch_size + 1)
    ]
    results = list(
        runner["run_batched_episodes"](
            client, [PrimitiveSimulator() for _ in range(batch_size)], episodes, tmp_path, tmp_path
        )
    )
    assert len(results) == len(episodes) and len(client.closed) == len(episodes)
    assert all(r["success"] and r["turns"] == 2 and r["steps"] == 4 for r in results)
    assert all(r["trace"][1]["primitive_step"] == 3 for r in results)
    assert results[-1]["trace"][0]["model"]["tensor_batch"]["observations"] == 1
    assert not client.sessions


@pytest.mark.parametrize("split,fail,match", [(True, False, "split across"), (False, True, "OOM")])
def test_batch_failure_closes_all_sessions_without_reporting_navigation_success(runner, tmp_path, split, fail, match):
    client = RoundClient(split=split, fail=fail)
    episodes = [dict(episode_id=str(i), instruction=dict(instruction_text="move")) for i in range(2)]
    with pytest.raises(RuntimeError, match=match):
        list(
            runner["run_batched_episodes"](
                client, [PrimitiveSimulator(), PrimitiveSimulator()], episodes, tmp_path, tmp_path
            )
        )
    assert len(client.closed) == 2 and not client.sessions
    for path in tmp_path.glob("episode-*.json"):
        result = json.loads(path.read_text())
        assert result["status"] == "failed" and "success" not in result


def test_batch_summary_counts_shared_latency_once_and_rejects_incomplete_batches(metrics, monkeypatch):
    monkeypatch.setitem(sys.modules, "metrics", metrics)
    functions = runpy.run_path(str(Path(__file__).parents[1] / "benchmarks/streamvln-navigation/summarize.py"))
    timed = functions["timed_invocations"]

    def turn(index, occupancy, elapsed):
        return dict(
            model=dict(tensor_batch=dict(index=index, observations=occupancy, capacity=2)), timing=dict(e2e_ms=elapsed)
        )

    rows = [turn(0, 2, 100), turn(1, 2, 200), turn(0, 2, 100), turn(1, 2, 200), turn(2, 1, 50)]
    # Transport-owned policy_ms includes different queue waits per request.
    for index, row in enumerate(rows):
        row["timing"]["policy_ms"] = row["timing"]["e2e_ms"] + index
    assert sum(row["timing"]["e2e_ms"] for row in timed(rows)) == 350
    assert len(timed(rows)) == 3
    with pytest.raises(ValueError, match="missing rows"):
        timed(rows[1:])
    rows[2]["timing"]["e2e_ms"] = 101
    with pytest.raises(ValueError, match="disagree"):
        timed(rows)
