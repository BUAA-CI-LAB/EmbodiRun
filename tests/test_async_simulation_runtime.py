"""Asynchronous simulator-episode runtime.

Everything here runs on CPU with fakes and an injected clock: a fake simulator, a
fake inference client, and a fake monotonic clock.  No LIBERO, no network, no GPU,
no inference engine.

The ``Δ`` (``inference_delay_steps``) exercised throughout is the *modeled* delay
the runtime advances the simulator by while the next chunk is computed, exactly as
the RTC and VLASH papers express it in simulation.
"""

from __future__ import annotations

import pytest

from embodirun.application.simulation.async_runtime import (
    ASYNC_SIMULATION_MODES,
    AsyncSimulationConfig,
    AsyncSimulationRuntime,
    latency_steps,
)
from embodirun.application.simulation.runtime import EpisodeOutcome, SimulationRuntime
from embodirun.model_services import (
    ImagePayload,
    PolicyAction,
    PolicyObservation,
    PolicyResult,
    Session,
)
from embodirun.robots import RobotAction, RobotObservation
from embodirun.robots.sensors.cameras import CameraFrame
from embodirun.simulators import SimulationStep, SimulatorAdapter, SimulatorObservation

FRAME_NAME = "observation.images.front"
INSTRUCTION = "pick up the black bowl"


class FakeClock:
    """Deterministic monotonic clock advanced by explicit, injected amounts."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeSimulator(SimulatorAdapter):
    """Deterministic LIBERO-shaped simulator: state is the executed step count."""

    simulator_id = "fake-libero-sim"

    def __init__(self, *, terminate_at: int | None = None, truncate_at: int | None = None, reward: float = 1.0):
        self.terminate_at = terminate_at
        self.truncate_at = truncate_at
        self.reward = reward
        self.step_index = 0
        self.executed: list[float] = []
        self.closed = False
        self.reset_count = 0

    def reset(self, *, task=None, seed=None, options=None) -> SimulatorObservation:
        self.reset_count += 1
        self.step_index = 0
        self.executed = []
        return self._observation()

    def step(self, action: RobotAction) -> SimulationStep:
        self.step_index += 1
        self.executed.append(float(action.values["vector"][0]))
        terminated = self.terminate_at is not None and self.step_index >= self.terminate_at
        truncated = self.truncate_at is not None and self.step_index >= self.truncate_at
        return SimulationStep(
            observation=self._observation(),
            reward=self.reward,
            terminated=terminated,
            truncated=truncated,
        )

    def close(self) -> None:
        self.closed = True

    def _observation(self) -> SimulatorObservation:
        step = self.step_index
        return SimulatorObservation(
            robot=RobotObservation(
                timestamp_s=float(step),
                values={"observation.state": [float(step)]},
                metadata={
                    "instruction": INSTRUCTION,
                    "step_index": step,
                    "observation_id": f"obs-{step}",
                },
            ),
            frames=(CameraFrame(FRAME_NAME, "image/jpeg", f"frame-{step}".encode()),),
        )


class FakeInferenceClient:
    """Blocking fake client that advances the injected clock while 'inferring'."""

    policy_action_space = "fake.action_chunk.v1"

    def __init__(self, *, chunk_steps: int, clock: FakeClock, latency_s: float) -> None:
        self.requests: list[PolicyObservation] = []
        self._chunk_steps = chunk_steps
        self._clock = clock
        self._latency_s = latency_s
        self._counter = 0
        self.closed = False

    def open_session(self, *, robot_id: str, action_space: str, metadata=None) -> Session:
        self.session_metadata = dict(metadata or {})
        return Session("session-1", 0)

    def step(self, observation: PolicyObservation) -> PolicyResult:
        self.requests.append(observation)
        self._clock.advance(self._latency_s)
        base = 100.0 * self._counter
        self._counter += 1
        data = [[base + float(row)] for row in range(self._chunk_steps)]
        return PolicyResult(
            request_id=observation.request_id,
            session_id=observation.session_id,
            step_id=observation.step_id,
            session_revision=self._counter,
            action_space=self.policy_action_space,
            actions=(PolicyAction("action_chunk", {"data": data}),),
        )

    def reset(self, session_id: str, *, request_id: str) -> Session:
        return Session(session_id, 0)

    def close(self, session_id: str) -> None:
        self.closed = True


class FakeMapper:
    """Maps the fake observation and the fake action chunk without any conversion."""

    policy_action_space = "fake.action_chunk.v1"

    def map_observation(
        self,
        observation: RobotObservation,
        *,
        session_id: str,
        request_id: str,
        step_id: int,
        instruction: str,
        frames,
    ) -> PolicyObservation:
        return PolicyObservation(
            session_id=session_id,
            request_id=request_id,
            step_id=step_id,
            instruction=instruction,
            state=dict(observation.values),
            images=tuple(ImagePayload(frame.name, frame.mime_type, frame.data) for frame in frames),
            metadata={
                "robot_timestamp_s": observation.timestamp_s,
                "observation_id": observation.metadata.get("observation_id"),
                "step_index": observation.metadata.get("step_index"),
            },
        )

    def map_result(self, result: PolicyResult) -> tuple[RobotAction, ...]:
        data = result.actions[0].values["data"]
        return tuple(
            RobotAction(timestamp_s=0.0, values={"type": "joint_position", "vector": [float(value) for value in row]})
            for row in data
        )


def _harness(
    *,
    mode: str,
    chunk_steps: int,
    overlap_steps: int = 0,
    inference_delay_steps: int = 0,
    latency_s: float = 0.0,
    control_hz: float = 10.0,
    execution_horizon: int | None = None,
    prefix_attention_schedule: str = "linear",
    max_guidance_weight: float = 10.0,
    hard_prefix: bool = False,
    terminate_at: int | None = None,
    truncate_at: int | None = None,
):
    config = AsyncSimulationConfig(
        mode=mode,
        chunk_steps=chunk_steps,
        overlap_steps=overlap_steps,
        inference_delay_steps=inference_delay_steps,
        control_hz=control_hz,
        execution_horizon=execution_horizon,
        prefix_attention_schedule=prefix_attention_schedule,
        max_guidance_weight=max_guidance_weight,
        hard_prefix=hard_prefix,
    )
    clock = FakeClock()
    simulator = FakeSimulator(terminate_at=terminate_at, truncate_at=truncate_at)
    client = FakeInferenceClient(chunk_steps=chunk_steps, clock=clock, latency_s=latency_s)
    runtime = AsyncSimulationRuntime(
        simulator,
        client,
        mapper=FakeMapper(),
        config=config,
        monotonic=clock.monotonic,
    )
    return runtime, simulator, client, clock


def _run(runtime, *, max_policy_steps: int, instruction: str | None = INSTRUCTION):
    return runtime.run(instruction=instruction, max_policy_steps=max_policy_steps, seed=7)


def _async_block(request: PolicyObservation):
    return request.metadata.get("async")


# ============================================================================
# sync
# ============================================================================
def test_sync_mode_has_no_overlap_no_delay_and_no_async_block() -> None:
    runtime, simulator, client, _clock = _harness(mode="sync", chunk_steps=4)

    outcome = _run(runtime, max_policy_steps=3)
    runtime.close()

    assert outcome.policy_steps == 3
    assert outcome.requests == 3
    assert outcome.environment_steps == 12
    assert outcome.terminated is False and outcome.truncated is False
    assert outcome.inference_delay_steps == (0, 0, 0)

    # A request is issued only once the previous chunk is fully consumed: the
    # simulator state at request time is 0, 4, 8.
    assert [request.state["observation.state"] for request in client.requests] == [[0.0], [4.0], [8.0]]
    assert all("async" not in request.metadata for request in client.requests)
    assert client.closed is True
    assert simulator.executed == [
        0.0,
        1.0,
        2.0,
        3.0,
        100.0,
        101.0,
        102.0,
        103.0,
        200.0,
        201.0,
        202.0,
        203.0,
    ]


def test_sync_mode_outcome_is_episode_outcome_compatible() -> None:
    runtime, _simulator, _client, _clock = _harness(mode="sync", chunk_steps=3)

    outcome = _run(runtime, max_policy_steps=2)

    assert outcome.to_episode_outcome() == EpisodeOutcome(2, 6, 6.0, False, False)
    assert isinstance(outcome.to_episode_outcome(), EpisodeOutcome)


# ============================================================================
# naive
# ============================================================================
def test_naive_mode_sends_no_conditioning_block_and_adopts_at_handover() -> None:
    runtime, _simulator, client, _clock = _harness(
        mode="naive", chunk_steps=5, overlap_steps=2, inference_delay_steps=2
    )

    outcome = _run(runtime, max_policy_steps=2)

    assert outcome.requests == 2
    assert all("async" not in request.metadata for request in client.requests)

    trigger = client.requests[1]
    assert trigger.metadata["observation_id"] == "obs-3"
    assert trigger.state["observation.state"] == [3.0]
    assert client.requests[1].images[0].data == b"frame-3"

    record = outcome.records[1]
    assert record.trigger is True
    assert record.trigger_chunk_index == 3
    assert record.environment_steps_at_capture == 3
    assert record.environment_steps_at_send == 5
    assert record.inference_delay_steps == 2

    # The first chunk was executed through to its end (handover) before the new
    # chunk's first action ran, so the naive baseline never replaces the tail.
    assert _simulator_actions(runtime) == [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,  # bootstrap chunk, fully executed
        100.0,
        101.0,
        102.0,  # new chunk taken over at handover
    ]


def _simulator_actions(runtime: AsyncSimulationRuntime) -> list[float]:
    return list(runtime.simulator.executed)


# ============================================================================
# rtc
# ============================================================================
def test_rtc_mode_emits_inference_delay_horizon_and_schedule() -> None:
    runtime, _simulator, client, _clock = _harness(
        mode="rtc",
        chunk_steps=5,
        overlap_steps=2,
        inference_delay_steps=2,
        execution_horizon=4,
        prefix_attention_schedule="exp",
        max_guidance_weight=7.5,
        hard_prefix=True,
    )

    outcome = _run(runtime, max_policy_steps=2)

    block = _async_block(client.requests[1])
    assert set(block) == {"schema", "rtc"}
    assert block["schema"] == "embodiinfer.async.v1"
    assert set(block["rtc"]) == {
        "prev_chunk_left_over",
        "inference_delay",
        "execution_horizon",
        "prefix_attention_schedule",
        "max_guidance_weight",
        "hard_prefix",
    }
    assert block["rtc"]["inference_delay"] == 2
    assert block["rtc"]["execution_horizon"] == 4
    assert block["rtc"]["prefix_attention_schedule"] == "exp"
    assert block["rtc"]["max_guidance_weight"] == 7.5
    assert block["rtc"]["hard_prefix"] is True
    # The client holds no model space, so it sends the delay and lets the server
    # slice its own cache.
    assert block["rtc"]["prev_chunk_left_over"] is None
    assert outcome.records[1].async_block == block

    # RTC adopts immediately, skipping the Δ actions that elapsed during the window.
    assert _simulator_actions(runtime) == [0.0, 1.0, 2.0, 3.0, 4.0, 102.0]


def test_rtc_bootstrap_request_carries_a_zero_delay_block() -> None:
    runtime, _simulator, client, _clock = _harness(
        mode="rtc", chunk_steps=5, overlap_steps=2, inference_delay_steps=1, execution_horizon=3
    )

    _run(runtime, max_policy_steps=1)

    bootstrap = _async_block(client.requests[0])["rtc"]
    assert bootstrap["inference_delay"] == 0
    assert bootstrap["prev_chunk_left_over"] is None
    assert bootstrap["execution_horizon"] == 3


# ============================================================================
# vlash
# ============================================================================
def test_vlash_mode_sends_the_exact_actions_consumed_in_the_window() -> None:
    runtime, _simulator, client, _clock = _harness(
        mode="vlash", chunk_steps=5, overlap_steps=3, inference_delay_steps=2
    )

    outcome = _run(runtime, max_policy_steps=2)

    block = _async_block(client.requests[1])
    assert set(block) == {"schema", "vlash"}
    assert block["vlash"]["delay"] == 2
    assert block["vlash"]["pending_actions"] == [[2.0], [3.0]]
    assert block["vlash"]["state_fields"] == ["observation.state"]
    assert block["vlash"]["action_space"] == "delta"
    assert outcome.records[1].pending_actions == ((2.0,), (3.0,))

    # The server rolled the captured state forward by Δ, so the new chunk starts at
    # its first action and the old chunk's unexecuted tail (row 4) is dropped.
    assert _simulator_actions(runtime) == [0.0, 1.0, 2.0, 3.0, 100.0, 101.0]

    # The queue is the faithful mirror of the executing chunk.
    assert runtime.queue.get_processed_left_over().tolist() == [[102.0], [103.0], [104.0]]
    assert runtime.queue.qsize() == 3


def test_vlash_bootstrap_request_has_no_conditioning_block() -> None:
    runtime, _simulator, client, _clock = _harness(
        mode="vlash", chunk_steps=5, overlap_steps=2, inference_delay_steps=1
    )

    _run(runtime, max_policy_steps=1)

    assert "async" not in client.requests[0].metadata


# ============================================================================
# stale observation capture
# ============================================================================
def test_request_uses_the_observation_captured_before_the_window() -> None:
    runtime, _simulator, client, _clock = _harness(mode="rtc", chunk_steps=6, overlap_steps=3, inference_delay_steps=3)

    outcome = _run(runtime, max_policy_steps=2)

    captured = client.requests[1]
    assert captured.metadata["observation_id"] == "obs-3"
    assert captured.metadata["step_index"] == 3
    assert captured.state["observation.state"] == [3.0]
    assert captured.images[0].data == b"frame-3"

    record = outcome.records[1]
    assert record.captured_observation_id == "obs-3"
    assert record.environment_steps_at_capture == 3
    assert record.environment_steps_at_send == 6
    assert record.trigger_chunk_index == 3


# ============================================================================
# Δ = 0 degenerates to the synchronous result
# ============================================================================
def test_zero_delay_degenerates_to_the_synchronous_result() -> None:
    sync_runtime, _sim, sync_client, _clock = _harness(mode="sync", chunk_steps=5)
    sync_outcome = _run(sync_runtime, max_policy_steps=2)

    naive_runtime, _sim2, naive_client, _clock2 = _harness(
        mode="naive", chunk_steps=5, overlap_steps=0, inference_delay_steps=0
    )
    naive_outcome = _run(naive_runtime, max_policy_steps=2)

    assert naive_outcome.to_episode_outcome() == sync_outcome.to_episode_outcome()
    assert [request.state for request in naive_client.requests] == [
        dict(request.state) for request in sync_client.requests
    ]
    assert all("async" not in request.metadata for request in naive_client.requests)

    # RTC/VLASH still attach their block, but the interaction is identical because
    # there is no overlap to trigger early and no delay to model.
    for mode in ("rtc", "vlash"):
        runtime, _simulator, client, _clock = _harness(
            mode=mode, chunk_steps=5, overlap_steps=0, inference_delay_steps=0
        )
        outcome = _run(runtime, max_policy_steps=2)
        assert outcome.to_episode_outcome() == sync_outcome.to_episode_outcome()
        assert [request.state for request in client.requests] == [
            dict(request.state) for request in sync_client.requests
        ]


def test_sync_runtime_and_async_sync_mode_execute_identically() -> None:
    runtime, _simulator, client, _clock = _harness(mode="sync", chunk_steps=4)
    async_outcome = _run(runtime, max_policy_steps=3)
    async_actions = list(runtime.simulator.executed)

    clock = FakeClock()
    simulator = FakeSimulator()
    reference_client = FakeInferenceClient(chunk_steps=4, clock=clock, latency_s=0.0)
    reference = SimulationRuntime(simulator, reference_client, mapper=FakeMapper(), chunk_steps=4)
    reference_outcome = reference.run(instruction=INSTRUCTION, max_policy_steps=3, seed=7)
    reference.close()

    assert async_outcome.to_episode_outcome() == reference_outcome
    assert async_actions == list(simulator.executed)
    assert [request.state for request in client.requests] == [
        dict(request.state) for request in reference_client.requests
    ]


# ============================================================================
# measured latency is reported, never substituted for Δ
# ============================================================================
def test_measured_latency_is_reported_but_not_used_as_the_modeled_delay() -> None:
    runtime, _simulator, client, _clock = _harness(
        mode="rtc",
        chunk_steps=6,
        overlap_steps=3,
        inference_delay_steps=2,
        latency_s=0.5,
        control_hz=10.0,
    )

    outcome = _run(runtime, max_policy_steps=2)

    record = outcome.records[1]
    assert record.measured_latency_s == pytest.approx(0.5)
    assert record.latency_steps == 5  # 0.5 s at 10 Hz
    assert record.inference_delay_steps == 2  # Δ, not the measurement
    assert record.environment_steps_at_send - record.environment_steps_at_capture == 2
    assert _async_block(client.requests[1])["rtc"]["inference_delay"] == 2
    assert outcome.latency_steps[1] == 5
    assert outcome.inference_delay_steps == (0, 2)

    assert runtime.latency.average() == pytest.approx(0.5)


def test_latency_steps_helper_and_validation() -> None:
    assert latency_steps(0.25, control_hz=10.0) == 2
    assert latency_steps(0.05, control_hz=10.0) == 0
    assert latency_steps(-1.0, control_hz=10.0) == 0
    with pytest.raises(ValueError):
        latency_steps(0.1, control_hz=0.0)
    with pytest.raises(TypeError):
        latency_steps(0.1, control_hz="fast")  # type: ignore[arg-type]


# ============================================================================
# termination
# ============================================================================
def test_episode_that_terminates_inside_the_inference_window() -> None:
    runtime, _simulator, client, _clock = _harness(
        mode="rtc",
        chunk_steps=5,
        overlap_steps=2,
        inference_delay_steps=2,
        terminate_at=4,
    )

    outcome = _run(runtime, max_policy_steps=5)

    assert outcome.terminated is True
    assert outcome.truncated is False
    assert outcome.policy_steps == 1
    assert outcome.requests == 1
    assert outcome.environment_steps == 4
    # Termination inside the window aborts the trigger request entirely.
    assert len(client.requests) == 1
    assert len(outcome.records) == 1


def test_episode_that_hits_max_policy_steps() -> None:
    runtime, _simulator, _client, _clock = _harness(mode="rtc", chunk_steps=5, overlap_steps=2, inference_delay_steps=2)

    outcome = _run(runtime, max_policy_steps=3)

    assert outcome.terminated is False and outcome.truncated is False
    assert outcome.policy_steps == 3
    assert outcome.requests == 3
    assert outcome.inference_delay_steps == (0, 2, 2)
    # Bootstrap (5 steps), then a request every H - O = 3 steps at the trigger.
    assert outcome.environment_steps == 9


def test_truncated_step_stops_the_episode() -> None:
    runtime, _simulator, _client, _clock = _harness(
        mode="vlash", chunk_steps=4, overlap_steps=2, inference_delay_steps=1, truncate_at=2
    )

    outcome = _run(runtime, max_policy_steps=4)

    assert outcome.truncated is True
    assert outcome.environment_steps == 2
    assert outcome.policy_steps == 1


# ============================================================================
# validation
# ============================================================================
def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="chunk_steps"):
        AsyncSimulationConfig(mode="sync", chunk_steps=0)
    with pytest.raises(ValueError, match="mode"):
        AsyncSimulationConfig(mode="teleport")
    with pytest.raises(ValueError, match="inference_delay_steps"):
        AsyncSimulationConfig(mode="rtc", chunk_steps=5, overlap_steps=2, inference_delay_steps=-1)
    with pytest.raises(ValueError, match="remaining at the trigger"):
        AsyncSimulationConfig(mode="rtc", chunk_steps=5, overlap_steps=2, inference_delay_steps=3)
    with pytest.raises(ValueError, match="adopted chunk"):
        # Δ <= overlap passes, but each RTC-adopted chunk is only H - Δ long.
        AsyncSimulationConfig(mode="rtc", chunk_steps=5, overlap_steps=3, inference_delay_steps=3)
    with pytest.raises(ValueError, match="overlap_steps"):
        AsyncSimulationConfig(mode="sync", chunk_steps=5, overlap_steps=1)
    with pytest.raises(ValueError, match="overlap_steps"):
        AsyncSimulationConfig(mode="rtc", chunk_steps=5, overlap_steps=6, inference_delay_steps=0)
    with pytest.raises(ValueError, match="execution_horizon"):
        AsyncSimulationConfig(mode="rtc", execution_horizon=-1)
    with pytest.raises(ValueError, match="prefix_attention_schedule"):
        AsyncSimulationConfig(mode="rtc", prefix_attention_schedule="cosine")
    with pytest.raises(ValueError, match="action_space"):
        AsyncSimulationConfig(mode="vlash", action_space="relative")
    with pytest.raises(ValueError, match="state_fields"):
        AsyncSimulationConfig(mode="vlash", state_fields=())
    with pytest.raises(ValueError, match="control_hz"):
        AsyncSimulationConfig(control_hz=0.0)


def test_modes_are_the_documented_four() -> None:
    assert ASYNC_SIMULATION_MODES == ("sync", "naive", "rtc", "vlash")


def test_run_rejects_invalid_arguments() -> None:
    runtime, _simulator, _client, _clock = _harness(mode="sync", chunk_steps=3)
    with pytest.raises(ValueError, match="instruction"):
        runtime.run(instruction="   ", max_policy_steps=1)
    with pytest.raises(ValueError, match="max_policy_steps"):
        runtime.run(instruction=INSTRUCTION, max_policy_steps=0)


def test_delay_is_a_modeled_parameter_not_a_wall_clock_sleep() -> None:
    """The episode advances by simulator steps only; the clock is never slept on."""

    runtime, simulator, _client, clock = _harness(
        mode="vlash", chunk_steps=4, overlap_steps=2, inference_delay_steps=2, latency_s=0.0
    )

    outcome = _run(runtime, max_policy_steps=2)

    assert outcome.environment_steps == 6
    assert clock.now == 0.0  # no latency configured, so no measured time elapsed
    assert simulator.step_index == 6
