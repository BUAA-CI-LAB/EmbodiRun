"""Exercise the terminal, real policy/session and fake IO without robot hardware."""

import json
import signal

import pytest

from embodirun.application.navigation.chunk_session import ChunkNavigationSession
from embodirun.bindings.unitree.go2.activevln import cli
from embodirun.bindings.unitree.go2.activevln.policy import ACTION_SPACE
from embodirun.model_services.contracts import PolicyAction, PolicyResult, Session
from embodirun.robots.mobile_base.frames import EncodedRGBFrame
from embodirun.robots.mobile_base.interfaces import MobileBaseError
from embodirun.robots.mobile_base.motion import MobileBaseState
from embodirun.robots.mobile_base.observation import NavigationObservation
from embodirun.utils import Pose2D


class Clock:
    now = 1.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Base:
    def __init__(self, clock):
        self.clock = clock
        self.active = None
        self.last = None
        self.expires = 0
        self.ready = self.supports_release = True
        self.starts, self.releases, self.stops = [], [], []
        self.unknown_start = False

    def state(self):
        if self.clock() >= self.expires:
            self.active = None
        return MobileBaseState(Pose2D(0, 0, 0), 0, 0, 0, 1, None, self.active)

    def preflight(self, *, require_release=False, require_renewal=False):
        if require_release and not self.supports_release:
            raise MobileBaseError("update the Go2 owner")
        if not self.ready or self.state().active_velocity_lease_id:
            raise MobileBaseError("base unavailable")
        return self.state()

    def start_velocity_lease(self, command, *, duration_s):
        self.active = self.last = str(len(self.starts))
        self.starts.append(command)
        self.expires = self.clock() + duration_s
        if self.unknown_start:
            raise MobileBaseError("start response lost")
        return self.active

    def update_velocity_lease(self, lease, command):
        assert self.state().active_velocity_lease_id == lease

    def release_velocity_lease(self, lease):
        assert lease == self.last and self.ready
        self.releases.append(lease)
        self.active = None

    def stop(self):
        self.stops.append(True)
        self.active, self.ready = None, False


class Camera:
    def __init__(self, clock):
        self.clock, self.sequence = clock, 0

    def capture(self, *, episode_id, reset, robot_state):
        self.sequence += 1
        return NavigationObservation(
            episode_id,
            self.sequence,
            reset,
            (EncodedRGBFrame(self.sequence, self.clock(), b"\xff\xd8jpeg"),),
        )


class Inference:
    def __init__(self):
        self.opened, self.closed, self.steps = [], [], []
        self.error = None
        self.close_error = None

    def capabilities(self):
        return {"adapter": {"model": "activevln", "recurrent": True, "action_space": ACTION_SPACE, "max_batch_size": 1}}

    def open_session(self, **kwargs):
        assert len(self.opened) == len(self.closed), "previous recurrent history must be closed first"
        session = Session(str(len(self.opened)), 0)
        self.opened.append((session, kwargs))
        return session

    def close(self, session):
        if self.close_error:
            raise self.close_error
        self.closed.append(session)

    def step(self, observation):
        self.steps.append(observation)
        if observation.session_id == "0" and observation.step_id == 1 and self.error:
            if callable(self.error):
                self.error()
            raise self.error
        rows = [[1, 25]] * 3 if observation.step_id < 2 else [[0, 0], [-1, 0], [-1, 0]]
        return PolicyResult(
            observation.request_id,
            observation.session_id,
            observation.step_id,
            observation.step_id + 1,
            ACTION_SPACE,
            (PolicyAction("discrete_chunk", {"rows": rows, "parsed_valid": True}),),
        )


def test_idle_inspection_waits_for_fresh_rgb_without_acquiring_motion():
    clock = Clock()
    base, camera = Base(clock), Camera(clock)
    session = ChunkNavigationSession(None, camera, base, clock=clock, sleep=clock.sleep)
    observation = session.capture_idle("rpent-inspect")
    assert observation.sequence == 2 and not observation.reset
    assert not base.starts and not base.releases and not base.stops
    base.active, base.expires = "other-controller", clock() + 10
    with pytest.raises(MobileBaseError, match="idle"):
        session.capture_idle("rpent-inspect")
    assert base.active == "other-controller" and not base.stops


@pytest.fixture
def setup_cli(tmp_path, monkeypatch):
    clock = Clock()
    base, camera, client = Base(clock), Camera(clock), Inference()
    connections = []

    def connect(*args, **kwargs):
        connections.append(True)
        return client

    monkeypatch.setattr(cli, "connect_client", connect)
    monkeypatch.setattr(cli, "Go2CameraClient", lambda *a, **kw: camera)
    monkeypatch.setattr(cli, "Go2ControlClient", lambda *a, **kw: base)
    monkeypatch.setattr(
        cli,
        "ChunkNavigationSession",
        lambda *a, **kw: ChunkNavigationSession(*a, **kw, clock=clock, sleep=clock.sleep),
    )
    path = tmp_path / "first-motion.json"
    path.write_text(
        json.dumps(
            {
                "robot_url": "http://localhost:8080",
                "inference_url": "http://localhost:18767",
                "mode": "chunk",
                "session": {"max_chunks": 1},
            }
        )
    )

    def run(lines, *extra):
        iterator = iter(lines)

        def read(prompt):
            try:
                line = next(iterator)
            except StopIteration:
                raise EOFError from None
            if isinstance(line, BaseException):
                raise line
            return line

        monkeypatch.setattr("builtins.input", read)
        return cli.main(["interactive", "--config", str(path), "--max-chunks", "100", "--max-runtime-s", "600", *extra])

    return run, base, client, connections, path


def test_multiple_long_tasks_reuse_client_but_reset_history(setup_cli, capsys, tmp_path):
    run, base, client, connections, _ = setup_cli
    log = tmp_path / "events.jsonl"
    assert run(["Go right", "Go left", "/quit"], "--execute", "--output-jsonl", str(log)) == 0
    assert len(connections) == 1
    assert len(base.starts) == 4 and len(base.releases) == 2
    assert base.ready and not base.stops
    assert [s.step_id for s in client.steps] == [0, 1, 2, 0, 1, 2]
    assert client.closed == ["0", "1"]
    episodes = [kw["metadata"]["episode_id"] for _, kw in client.opened]
    assert len(set(episodes)) == 2
    events = [json.loads(line) for line in log.read_text().splitlines()]
    results = [e for e in events if e["event"] == "navigation_finished"]
    assert [r["chunks"] for r in results] == [3, 3]
    assert [r["executed_actions"] for r in results] == [7, 7]
    assert "600 s" in capsys.readouterr().out


def test_ctrl_c_stops_task_then_accepts_fresh_instruction(setup_cli):
    run, base, client, _, _ = setup_cli
    client.error = KeyboardInterrupt()
    assert run(["first", "next", "/quit"], "--execute") == 0
    assert client.closed == ["0", "1"]
    assert [s.step_id for s in client.steps] == [0, 1, 0, 1, 2]
    assert len(base.releases) == 2 and not base.stops and base.ready


@pytest.mark.parametrize("fault", [ValueError("invalid model output"), MobileBaseError("connection lost")])
def test_task_fault_disarms_and_exits_instead_of_retrying(setup_cli, fault):
    run, base, client, _, _ = setup_cli
    client.error = fault
    assert run(["first", "must not run"], "--execute") == 1
    assert len(client.opened) == len(client.closed) == 1
    assert len(base.stops) == 1 and not base.ready and not base.releases


def test_sigterm_disarms_and_exits_then_restores_signal_handler(setup_cli):
    run, base, client, _, _ = setup_cli
    original = signal.getsignal(signal.SIGTERM)
    client.error = lambda: signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    with pytest.raises(SystemExit) as error:
        run(["first", "must not run"], "--execute")
    assert error.value.code == 143
    assert not base.ready and len(base.stops) == 1 and not base.releases
    assert client.closed == ["0"] and signal.getsignal(signal.SIGTERM) == original


def test_dry_run_repeats_without_control_writes(setup_cli):
    run, base, client, _, _ = setup_cli
    assert run(["", "/help", "/unknown", "first", "second"]) == 0
    assert [s.step_id for s in client.steps] == [0, 0]
    assert client.closed == ["0", "1"]
    assert not (base.starts or base.releases or base.stops)


def test_ctrl_c_at_prompt_and_eof_do_not_write_control(setup_cli):
    run, base, client, _, _ = setup_cli
    assert run([KeyboardInterrupt()], "--execute") == 0
    assert run([], "--execute") == 0
    assert not (client.opened or base.starts or base.releases or base.stops)


def test_old_owner_rejected_before_motion(setup_cli):
    run, base, client, _, _ = setup_cli
    base.supports_release = False
    assert run(["go"], "--execute") == 1
    assert not (client.opened or base.starts)


def test_unknown_start_outcome_uses_emergency_stop(setup_cli):
    run, base, client, _, _ = setup_cli
    base.unknown_start = True
    assert run(["go"], "--execute") == 1
    assert base.stops and not base.ready and not base.releases
    assert client.closed == ["0"]


def test_failure_to_close_history_exits_without_opening_next_session(setup_cli):
    run, _, client, _, _ = setup_cli
    client.close_error = RuntimeError("session close failed")
    assert run(["first", "must not run"]) == 1
    assert len(client.opened) == 1


def test_single_run_retains_disarming_behavior(setup_cli):
    _, base, client, _, path = setup_cli
    assert cli.main(["run", "--config", str(path), "--instruction", "go", "--execute"]) == 0
    assert len(client.steps) == len(base.starts) == len(base.stops) == 1
    assert not base.ready and not base.releases


@pytest.mark.parametrize(
    "limits, reason, starts",
    [
        (["--max-chunks", "2"], "chunk_limit", 2),
        (["--max-runtime-s", "5"], "max_runtime", 1),
    ],
)
def test_task_limits_release_and_return_to_prompt(setup_cli, limits, reason, starts, capsys):
    run, base, _, _, _ = setup_cli
    assert run(["go", "/quit"], "--execute", *limits) == 0
    assert len(base.starts) == starts and len(base.releases) == 1 and not base.stops
    assert f"Task ended: {reason}" in capsys.readouterr().out


@pytest.mark.parametrize(
    "limits",
    [
        ["--max-chunks", "0"],
        ["--max-runtime-s", "nan"],
        ["--overlap-lead-s", "0"],
        ["--overlap-lead-s", "-1"],
        ["--overlap-lead-s", "nan"],
        ["--overlap-lead-s", "inf"],
    ],
)
def test_invalid_limits_fail_before_connecting(setup_cli, limits):
    run, _, client, connections, _ = setup_cli
    assert run(["go"], "--execute", *limits) == 1
    assert not connections and not client.opened


def test_plan_previews_overrides_without_changing_file_or_connecting(setup_cli, capsys):
    _, _, _, connections, path = setup_cli
    original = path.read_bytes()
    assert cli.main(["plan", "--config", str(path), "--mode", "overlap", "--overlap-lead-s", "2.5"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["configuration"]["mode"] == "overlap"
    assert result["configuration"]["overlap"]["lead_s"] == 2.5
    assert result["motion_enabled"] is False
    assert not connections and path.read_bytes() == original


def test_overlap_override_requires_prefix_support_before_motion(setup_cli, capsys):
    run, base, client, _, _ = setup_cli
    assert run(["go"], "--execute", "--mode", "overlap") == 1
    assert "execution-prefix-aware" in capsys.readouterr().err
    assert not (client.opened or base.starts or base.stops)


def test_chunk_override_uses_sync_execution_with_overlap_config(setup_cli, capsys):
    run, base, client, _, path = setup_cli
    config = json.loads(path.read_text())
    config["mode"] = "overlap"
    path.write_text(json.dumps(config))
    assert run(["go", "/quit"], "--execute", "--mode", "chunk") == 0
    assert len(base.starts) == 2 and len(base.releases) == 1
    assert client.closed == ["0"]
    output = capsys.readouterr().out
    assert "Mode: chunk." in output
    started = [json.loads(line) for line in output.splitlines() if line.startswith('{"event": "navigation_started"')]
    assert started[0]["mode"] == "chunk"
