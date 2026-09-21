"""CPU tests for the real HTTP boundary and bounded episode execution."""

import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from embodirun.model_services.backends.vvla.http import VvlaHttpClient, VvlaHttpError
from embodirun.model_services.contracts import ImagePayload, PolicyObservation, Session

# Optional integration sources are not installed with EmbodiRun's core.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "integrations/microduck_vln/src"))

from embodirun_microduck.assets import asset_files, md5, verify_manifest  # noqa: E402
from embodirun_microduck.process import managed_service  # noqa: E402
from embodirun_microduck.protocol import ACTION_SPACE, IMAGE_FIELD, decode_actions  # noqa: E402
from embodirun_microduck.service import ActiveVLNAdapter  # noqa: E402

spec = importlib.util.spec_from_file_location("microduck_example", REPO / "examples/microduck_vln/run_demo.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def chunk(rows):
    import torch

    return SimpleNamespace(
        actions=torch.tensor(rows),
        latency_ms=1.0,
        timing={},
        policy_version=1,
        trace=SimpleNamespace(text="stop", stop_reason="eos", parsed_actions=SimpleNamespace(valid=True)),
    )


class FakeCore:
    def __init__(self):
        self.calls = []
        self.resets = []
        self.policy = SimpleNamespace(collate=lambda obs, ids: obs)

    def execute(self, batch, session_ids):
        self.calls.append((batch, session_ids))
        return [chunk([[0, 0], [-1, 0], [-1, 0]])]

    def reset_sessions(self, keys):
        self.resets.extend(keys)


class HttpBoundaryTests(unittest.TestCase):
    def setUp(self):
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("vvla") is None:
            self.skipTest("Optional EmbodiInfer/Torch dependencies unavailable")
        from vvla.engine.serve.http_server import PolicyHttpService, create_http_server

        self.core = FakeCore()
        service = PolicyHttpService(ActiveVLNAdapter(self.core), token="test-token", max_body_bytes=100000)
        self.server = create_http_server(service, host="127.0.0.1", port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.client = VvlaHttpClient(self.url, token="test-token")

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def observation(self, session, request_id="request-1", step=0):
        encoded = io.BytesIO()
        Image.fromarray(np.full((2, 2, 3), 127, dtype=np.uint8)).save(encoded, format="PNG")
        return PolicyObservation(
            session.session_id,
            request_id,
            step,
            "Go to the bin",
            {},
            (ImagePayload(IMAGE_FIELD, "image/png", encoded.getvalue()),),
        )

    def test_real_http_recurrence_idempotency_reset_and_close(self):
        self.assertEqual(self.client.capabilities()["adapter"]["action_space"], ACTION_SPACE)
        session = self.client.open_session(robot_id="microduck-sim", action_space=ACTION_SPACE)
        obs = self.observation(session)
        first = self.client.step(obs)
        self.assertEqual(first, self.client.step(obs))
        self.assertEqual(len(self.core.calls), 1)  # Retry must not advance recurrent state.
        pixels = self.core.calls[0][0][0].images
        self.assertAlmostEqual(float(pixels[0, 0, 0, 0]), 127 / 255, places=6)
        self.client.step(self.observation(session, "request-2", 1))
        self.assertEqual(self.core.calls[0][1], self.core.calls[1][1])
        self.assertEqual(decode_actions(first.actions[0].values["rows"]), [(0, 0)])
        reset = self.client.reset(session.session_id, request_id="reset-1")
        self.assertGreater(reset.revision, session.revision)
        self.client.step(self.observation(reset, "request-3"))
        self.client.close(session.session_id)
        with self.assertRaises(VvlaHttpError):
            self.client.step(self.observation(session, "request-4", 1))
        second = self.client.open_session(robot_id="microduck-sim", action_space=ACTION_SPACE)
        self.client.step(self.observation(second, "request-5"))
        self.assertNotEqual(self.core.calls[0][1], self.core.calls[-1][1])
        self.client.close(second.session_id)
        self.assertGreaterEqual(len(self.core.resets), 3)

    def test_unauthorized_request_and_step_gap_never_infer(self):
        with self.assertRaises(VvlaHttpError):
            VvlaHttpClient(self.url).capabilities()
        session = self.client.open_session(robot_id="microduck-sim", action_space=ACTION_SPACE)
        with self.assertRaises(VvlaHttpError):
            self.client.step(self.observation(session, step=3))
        self.assertEqual(self.core.calls, [])
        self.client.close(session.session_id)


class FakeSimulation:
    callback = None
    robot = SimpleNamespace(data=SimpleNamespace(time=0.0))
    mpc_ms = []

    def reset(self, start):
        self.executed = []
        self.path_length = 0.0
        self.physics_steps = 0

    def render(self):
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def execute(self, action):
        self.executed.append(action)
        self.physics_steps += 1
        return {"pose": self.pose()}

    def pose(self):
        return [0.0, 0.0, 0.0]

    def distance(self, goal):
        return 0.5


class FakeClient:
    def __init__(self, rows):
        self.rows = rows
        self.closed = []

    def open_session(self, **kwargs):
        return Session("session-1", 0)

    def step(self, obs):
        return SimpleNamespace(
            action_space=ACTION_SPACE,
            actions=[SimpleNamespace(kind="discrete_chunk", values={"rows": self.rows, "valid": True})],
            timing={"policy_ms": 1.0},
            request_id=obs.request_id,
            session_id=obs.session_id,
            session_revision=0,
        )

    def close(self, session_id):
        self.closed.append(session_id)


class AssetInventoryTests(unittest.TestCase):
    def test_inference_inventory_includes_non_python_runtime_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            project = root / "project"
            vvla = root / "inference"
            checkpoint = root / "checkpoint"
            (project / "src/robots_assets/mjcf_assets").mkdir(parents=True)
            (project / "data/train").mkdir(parents=True)
            (project / "data").mkdir(exist_ok=True)
            (vvla / "vvla").mkdir(parents=True)
            checkpoint.mkdir()
            (project / "src/robots_assets/alpha_walking.onnx").write_bytes(b"onnx")
            (project / "src/robots_assets/mjcf_assets/robot_allcollisions.xml").write_text("<mujoco/>")
            (project / "data/train/eval_val2_40_valid.jsonl").write_text("{}\n")
            (project / "data/demo_microduck_vln.jsonl").write_text("{}\n")
            (vvla / "vvla/model_config.json").write_text("{}")
            (checkpoint / "config.json").write_text("{}")
            (checkpoint / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"layer.safetensors": "layer.safetensors"}})
            )
            (checkpoint / "layer.safetensors").write_bytes(b"weights")

            files = asset_files({"project": project, "vvla": vvla, "checkpoint": checkpoint})

            self.assertIn("vvla/vvla/model_config.json", files)


class ExecutionTests(unittest.TestCase):
    def run_episode(self, rows, limit=60, hold=1, fail=False):
        args = SimpleNamespace(seed=7, no_video=True, max_steps=limit, success_radius=1.0, success_hold_steps=hold)
        record = {"id": "../untrusted-id", "instruction": "Go to the bin", "start": [0, 0, 0], "goal": [0.5, 0]}
        sim = FakeSimulation()
        client = FakeClient(rows)
        if fail:
            client.step = lambda obs: (_ for _ in ()).throw(TimeoutError("transport timeout"))
        with tempfile.TemporaryDirectory() as folder:
            if fail:
                with self.assertRaises(TimeoutError):
                    runner.run_episode(args, sim, client, record, 0, Path(folder), "")
            else:
                runner.run_episode(args, sim, client, record, 0, Path(folder), "")
            result = json.loads((Path(folder) / "episode_000.json").read_text())
        self.assertEqual(client.closed, ["session-1"])
        return result, sim

    def test_stop_discards_later_actions(self):
        result, sim = self.run_episode([[0, 0], [1, 75], [-1, 0]])
        self.assertEqual(sim.executed, [(0, 0)])
        self.assertTrue(result["success"])

    def test_success_requires_hold_endpoints(self):
        result, _ = self.run_episode([[0, 0], [-1, 0], [-1, 0]], hold=3)
        self.assertFalse(result["success"])
        result, _ = self.run_episode([[1, 25], [1, 25], [0, 0]], hold=3)
        self.assertTrue(result["success"])

    def test_action_cap_truncates_chunk(self):
        result, sim = self.run_episode([[1, 25], [1, 50], [1, 75]], limit=2)
        self.assertEqual(sim.executed, [(1, 25), (1, 50)])
        self.assertEqual(result["termination"], "max_steps")

    def test_invalid_rows_never_move(self):
        for rows in (
            [[1, 999], [-1, 0], [-1, 0]],
            [[-1, 0], [-1, 0], [-1, 0]],
            [[float("nan"), 0], [-1, 0], [-1, 0]],
            [[1.5, 25], [-1, 0], [-1, 0]],
        ):
            with self.assertRaises(ValueError):
                decode_actions(rows)
        result, sim = self.run_episode([[-1, 0], [-1, 0], [-1, 0]])
        self.assertEqual(sim.executed, [])
        self.assertEqual(result["termination"], "invalid_model_output")

    def test_transport_failure_persists_error_and_closes_session(self):
        result, sim = self.run_episode([[0, 0], [-1, 0], [-1, 0]], fail=True)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["session_cleared"])
        self.assertEqual(sim.executed, [])

    def test_video_cleanup_preserves_the_original_transport_failure(self):
        args = SimpleNamespace(no_video=False, fps=10, max_steps=1, success_radius=1.0, success_hold_steps=1)
        record = {"instruction": "Stop", "start": [0, 0, 0], "goal": [0.5, 0]}
        sim = FakeSimulation()
        sim.render = Mock(return_value=np.zeros((2, 2, 3), dtype=np.uint8))
        client = FakeClient([[0, 0], [-1, 0], [-1, 0]])
        client.step = Mock(side_effect=TimeoutError("transport timeout"))
        video = Mock()
        video.close.side_effect = RuntimeError("encoder cleanup failed")
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("embodirun_microduck.runtime.EpisodeVideo", return_value=video),
        ):
            with self.assertRaisesRegex(TimeoutError, "transport timeout"):
                runner.run_episode(args, sim, client, record, 0, Path(folder), "")
            result = json.loads((Path(folder) / "episode_000.json").read_text())
        self.assertIn("transport timeout", result["error"])
        self.assertIn("encoder cleanup failed", result["video_cleanup_error"])
        self.assertEqual(client.closed, ["session-1"])

    def test_integrity_detects_same_size_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            asset = root / "asset"
            asset.write_bytes(b"original")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"files": {"project/asset": {"md5": md5(asset), "bytes": 8}}}))
            with patch("embodirun_microduck.assets.asset_files", return_value={"project/asset": asset}):
                self.assertEqual(verify_manifest(manifest, {})["files_verified"], 1)
                asset.write_bytes(b"modified")
                with self.assertRaisesRegex(RuntimeError, "MD5 mismatch"):
                    verify_manifest(manifest, {})


class ProcessLifecycleTests(unittest.TestCase):
    def service_args(self, timeout=5.0, request_timeout=1.0):
        return SimpleNamespace(
            inference_root=Path("."),
            checkpoint=Path("."),
            seed=7,
            max_new_tokens=8,
            max_context=1024,
            sample=False,
            startup_timeout=timeout,
            request_timeout=request_timeout,
        )

    def launch_failure(self, code, timeout):
        children = []
        original = subprocess.Popen

        def launch(command, **kwargs):
            child = original([sys.executable, "-c", code], **kwargs)
            children.append(child)
            return child

        args = self.service_args(timeout)
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("embodirun_microduck.process.subprocess.Popen", side_effect=launch),
        ):
            with self.assertRaises((RuntimeError, TimeoutError)), managed_service(args, Path(folder)):
                self.fail("A failed service must not become ready")
            self.assertFalse((Path(folder) / "service.ready.json").exists())
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll())

    def test_startup_failure_reaps_child(self):
        self.launch_failure("raise SystemExit(3)", 5.0)

    def test_startup_timeout_terminates_child(self):
        self.launch_failure("import time; time.sleep(30)", 0.05)

    def test_bound_server_waits_for_failed_probe_before_serving(self):
        # A real listening socket deterministically reproduces publication before
        # serve_forever: the child cannot answer until the first probe times out.
        code = textwrap.dedent("""
            import json, os, sys, time
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
            from pathlib import Path

            ready = Path(sys.argv[1])
            allow_serving = ready.with_suffix('.allow')

            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.headers.get('Authorization') != 'Bearer ' + os.environ['MICRODUCK_SERVICE_TOKEN']:
                        self.send_error(401)
                        return
                    payload = ({'status': 'ok'} if self.path == '/healthz' else
                               {'adapter': {'action_space': sys.argv[2], 'image_fields': [sys.argv[3]]}})
                    body = json.dumps(payload).encode()
                    self.send_response(200)
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    try:
                        self.wfile.write(body)
                    except OSError:
                        pass  # The first probe has already timed out.

            server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
            temporary = ready.with_suffix('.tmp')
            temporary.write_text(json.dumps({'pid': os.getpid(), 'port': server.server_port}))
            temporary.replace(ready)
            while not allow_serving.exists():
                time.sleep(0.01)
            server.serve_forever()
        """)
        children = []
        failed_probes = []
        original = subprocess.Popen
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)

            def launch(command, **kwargs):
                ready = command[command.index("--ready-file") + 1]
                # Windows venv launchers delegate to another PID; this stdlib-only
                # fixture uses the underlying interpreter to match production Linux.
                python = getattr(sys, "_base_executable", sys.executable)
                child = original([python, "-c", code, ready, ACTION_SPACE, IMAGE_FIELD], **kwargs)
                children.append(child)
                return child

            class ProbeClient(VvlaHttpClient):
                def health(self):
                    try:
                        return super().health()
                    except VvlaHttpError as error:
                        failed_probes.append(error)
                        (output / "service.ready.allow").touch()
                        raise

            with (
                patch("embodirun_microduck.process.subprocess.Popen", side_effect=launch),
                patch("embodirun_microduck.process.VvlaHttpClient", ProbeClient),
                managed_service(self.service_args(timeout=10.0, request_timeout=0.1), output) as client,
            ):
                self.assertTrue(failed_probes)
                self.assertEqual(client.health()["status"], "ok")
                self.assertEqual(client.capabilities()["adapter"]["action_space"], ACTION_SPACE)
            self.assertFalse((output / "service.ready.json").exists())
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll())

    @contextmanager
    def startup_probes(self):
        # Virtual time keeps deadline and process-exit checks deterministic.
        clock = SimpleNamespace(now=0.0)
        process = Mock(pid=1234, returncode=None)
        process.poll.side_effect = lambda: process.returncode
        process.terminate.side_effect = lambda: setattr(process, "returncode", -15)
        client = Mock(timeout_s=120.0)
        client.health.return_value = {"status": "ok"}
        client.capabilities.return_value = {"adapter": {"action_space": ACTION_SPACE, "image_fields": [IMAGE_FIELD]}}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            ready = output / "service.ready.json"
            ready.write_text(json.dumps({"pid": process.pid, "port": 8000}))
            with (
                patch("embodirun_microduck.process.subprocess.Popen", return_value=process),
                patch("embodirun_microduck.process.VvlaHttpClient", return_value=client),
                patch("embodirun_microduck.process.time.monotonic", side_effect=lambda: clock.now),
                patch(
                    "embodirun_microduck.process.time.sleep",
                    side_effect=lambda seconds: setattr(clock, "now", clock.now + seconds),
                ),
            ):
                yield output, client, process, clock
            self.assertFalse(ready.exists())
            self.assertIsNotNone(process.poll())

    def test_retries_unhealthy_status_and_capabilities_transport_failure(self):
        with self.startup_probes() as (output, client, process, clock):
            client.health.side_effect = [{"status": "starting"}, {"status": "ok"}, {"status": "ok"}]
            client.capabilities.side_effect = [
                VvlaHttpError("temporarily unavailable"),
                client.capabilities.return_value,
            ]
            with managed_service(self.service_args(request_timeout=120.0), output) as result:
                self.assertIs(result, client)
                self.assertEqual(client.timeout_s, 120.0)
                self.assertEqual(client.health.call_count, 3)
                self.assertEqual(client.capabilities.call_count, 2)
                self.assertAlmostEqual(clock.now, 0.4)
            process.terminate.assert_called_once()

    def test_startup_deadline_bounds_health_and_capabilities(self):
        for endpoint in ("health", "capabilities"):
            with self.subTest(endpoint=endpoint), self.startup_probes() as (output, client, process, clock):
                timeouts = []

                def fail_probe(timeouts=timeouts, client=client, clock=clock):
                    timeouts.append(client.timeout_s)
                    clock.now += client.timeout_s
                    raise VvlaHttpError("probe timed out")

                getattr(client, endpoint).side_effect = fail_probe
                with (
                    self.assertRaisesRegex(TimeoutError, "startup timed out") as caught,
                    managed_service(self.service_args(timeout=1.5, request_timeout=120.0), output),
                ):
                    self.fail("A non-responsive server must not become ready")
                self.assertEqual(timeouts[0], 1.0)
                self.assertAlmostEqual(timeouts[1], 0.3)
                self.assertAlmostEqual(clock.now, 1.5)
                self.assertIsInstance(caught.exception.__cause__, VvlaHttpError)
                process.terminate.assert_called_once()

    def test_child_exit_during_health_probe_fails_without_waiting_for_deadline(self):
        with self.startup_probes() as (output, client, process, clock):

            def exit_during_probe():
                process.returncode = 3
                raise VvlaHttpError("connection reset")

            client.health.side_effect = exit_during_probe
            with (
                self.assertRaisesRegex(RuntimeError, r"Inference exited \(3\)"),
                managed_service(self.service_args(), output),
            ):
                self.fail("An exited service must not become ready")
            self.assertEqual(clock.now, 0.0)
            process.terminate.assert_not_called()

    def test_capability_mismatch_is_not_retried(self):
        with self.startup_probes() as (output, client, process, clock):
            client.capabilities.return_value = {"adapter": {"action_space": "wrong-binding"}}
            with (
                self.assertRaisesRegex(RuntimeError, "capabilities do not match"),
                managed_service(self.service_args(), output),
            ):
                self.fail("An incompatible service must not become ready")
            client.capabilities.assert_called_once()
            self.assertEqual(clock.now, 0.0)
            process.terminate.assert_called_once()


class PublicEntrypointTests(unittest.TestCase):
    def test_public_example_and_service_help_without_gpu_dependencies(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), str(REPO / "integrations/microduck_vln/src")])
        commands = [
            [str(REPO / "examples/microduck_vln/run_demo.py"), "--help"],
            ["-m", "embodirun_microduck.service", "--help"],
        ]
        for command in commands:
            with self.subTest(command=command):
                result = subprocess.run(
                    [sys.executable, *command], env=env, text=True, capture_output=True, timeout=15, check=False
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--checkpoint", result.stdout)

    def test_inference_path_is_explicit_and_ignores_old_environment_overrides(self):
        with patch.dict(os.environ, {"MICRODUCK_INFERENCE_ROOT": "public-checkout", "MICRODUCK_VVLA_ROOT": "legacy"}):
            with patch.object(sys, "argv", ["run_demo.py", "--project-root", "/tmp/assets"]):
                self.assertEqual(runner.parse_args().inference_root, REPO / "third_party/embodiinfer")
            with patch.object(
                sys, "argv", ["run_demo.py", "--project-root", "/tmp/assets", "--inference-root", "explicit-checkout"]
            ):
                self.assertEqual(runner.parse_args().inference_root, Path("explicit-checkout"))


if __name__ == "__main__":
    unittest.main()
