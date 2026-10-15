"""Execution-side RTC algorithms.

The numeric core is pinned against the *shared* golden vectors in
``tests/fixtures/async_inference_golden.json``, which EmbodiInfer asserts with its
own inference-side implementation.  Both halves of the async loop therefore agree
on the same arithmetic across a process and a repository boundary.

Everything here runs on CPU with fakes and an injected clock; no robot hardware
and no inference engine are involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from embodirun.application.async_inference import (
    ASYNC_SCHEMA,
    ActionQueue,
    AsyncControlRuntime,
    AsyncRuntimeConfig,
    LatencyTracker,
    RTCAsyncRequest,
    build_async_metadata,
    prefix_weights,
)
from embodirun.model_services import (
    ImagePayload,
    PolicyAction,
    PolicyObservation,
    PolicyResult,
    Session,
)
from embodirun.robots import RobotAction, RobotObservation
from embodirun.robots.sensors.cameras import CameraFrame

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "async_inference_golden.json"


def _golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


# ============================================================================
# Shared golden vectors
# ============================================================================
@pytest.mark.parametrize("case", _golden()["rtc_prefix_weights"], ids=lambda case: case["name"])
def test_prefix_weights_match_golden_vectors(case: dict) -> None:
    weights = prefix_weights(case["start"], case["end"], case["total"], case["schedule"])
    expected = np.asarray(case["expected"], dtype=np.float64)
    assert weights.shape == (case["total"],), case["name"]
    assert np.allclose(weights, expected, atol=1e-6), (case["name"], weights.tolist())


def test_runtime_metadata_round_trips_as_json() -> None:
    """The async block must be JSON-serializable as the HTTP client sends it."""

    payload = build_async_metadata(
        rtc=RTCAsyncRequest(
            prev_chunk_left_over=((1.0, 2.0),),
            inference_delay=1,
            execution_horizon=2,
            model_space=True,
        ),
    )
    assert json.loads(json.dumps(payload)) == payload


# ============================================================================
# RTC prefix weights
# ============================================================================
def test_prefix_weights_documented_example() -> None:
    weights = prefix_weights(2, 6, 10, "linear")
    assert [round(float(value), 6) for value in weights] == [1, 1, 0.8, 0.6, 0.4, 0.2, 0, 0, 0, 0]


def test_prefix_weights_start_is_pushed_down_to_end() -> None:
    assert np.array_equal(prefix_weights(9, 4, 10, "zeros"), prefix_weights(4, 4, 10, "zeros"))


def test_prefix_weights_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        prefix_weights(0, 1, 0)
    with pytest.raises(ValueError):
        prefix_weights(-1, 1, 4)
    with pytest.raises(ValueError):
        prefix_weights(0, 1, 4, "cosine")
    with pytest.raises(ValueError):
        prefix_weights(0, 1, 4.0)  # type: ignore[arg-type]


# ============================================================================
# ActionQueue — replace (RTC) vs append (non-RTC)
# ============================================================================
def _chunks(rows: int, width: int = 2, offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    original = np.arange(rows * width, dtype=np.float64).reshape(rows, width) + offset
    processed = original + 100.0
    return original, processed


def test_action_queue_replace_drops_the_delay_prefix() -> None:
    original, processed = _chunks(5)
    queue = ActionQueue(rtc_enabled=True)
    queue.merge(original, processed, 2)

    assert queue.get_action_index() == 0
    assert queue.qsize() == 3
    assert not queue.empty()
    assert np.array_equal(queue.get_left_over(), original[2:])
    assert np.array_equal(queue.get_processed_left_over(), processed[2:])


def test_action_queue_get_returns_a_clone_and_advances_the_index() -> None:
    original, processed = _chunks(3)
    queue = ActionQueue(rtc_enabled=True)
    queue.merge(original, processed, 0, task="pick")

    action, task = queue.get_with_task()
    assert task == "pick"
    assert np.array_equal(action, processed[0])
    action[0] = -999.0
    assert queue.get_action_index() == 1
    assert np.array_equal(queue.get_left_over(), original[1:])
    assert np.array_equal(queue.get_processed_left_over(), processed[1:])
    assert np.array_equal(queue.get(), processed[1])
    assert np.array_equal(queue.get(), processed[2])
    assert queue.get() is None
    assert queue.empty()


def test_action_queue_leftover_after_partial_consumption() -> None:
    original, processed = _chunks(6)
    queue = ActionQueue(rtc_enabled=True)
    queue.merge(original, processed, 1)
    queue.get()
    queue.get()
    assert np.array_equal(queue.get_left_over(), original[3:])
    assert np.array_equal(queue.get_processed_left_over(), processed[3:])


def test_action_queue_append_preserves_continuity() -> None:
    original_a, processed_a = _chunks(5)
    original_b, processed_b = _chunks(3, offset=1000.0)
    queue = ActionQueue(rtc_enabled=False)
    queue.merge(original_a, processed_a, 0)
    queue.get()
    queue.get()
    queue.merge(original_b, processed_b, 0)

    assert queue.get_action_index() == 0
    assert queue.qsize() == 6
    assert np.array_equal(queue.get_processed_left_over(), np.concatenate([processed_a[2:], processed_b]))


def test_action_queue_append_from_empty_starts_the_queue() -> None:
    original, processed = _chunks(4)
    queue = ActionQueue(rtc_enabled=False)
    queue.merge(original, processed, 3)
    assert np.array_equal(queue.get_left_over(), original)
    assert np.array_equal(queue.get_processed_left_over(), processed)


def test_action_queue_clear_resets_everything() -> None:
    original, processed = _chunks(3)
    queue = ActionQueue(rtc_enabled=True)
    queue.merge(original, processed, 0)
    queue.get()
    queue.clear()
    assert queue.empty()
    assert queue.qsize() == 0
    assert queue.get_action_index() == 0
    assert queue.get_left_over() is None
    assert queue.get_processed_left_over() is None


def test_action_queue_delay_resolution_never_over_discards() -> None:
    original, processed = _chunks(5)
    queue = ActionQueue(rtc_enabled=True)
    queue.merge(original, processed, 0, task="pick")
    queue.get()
    queue.get()
    queue.get()

    # Inference claims a delay of 5 but only three actions were consumed.
    original_new, processed_new = _chunks(4, offset=2000.0)
    resolved = queue.merge(original_new, processed_new, 5, action_index_before_inference=0)
    assert resolved == 3
    assert np.array_equal(queue.get_left_over(), original_new[3:])
    assert np.array_equal(queue.get_processed_left_over(), processed_new[3:])


def test_action_queue_delay_resolution_keeps_the_smaller_delay() -> None:
    original, processed = _chunks(5)
    queue = ActionQueue(rtc_enabled=True)
    queue.merge(original, processed, 0)
    for _ in range(4):
        queue.get()

    original_new, processed_new = _chunks(4, offset=3000.0)
    # real_delay 2 is smaller than the 4 consumed and is what gets applied.
    resolved = queue.merge(original_new, processed_new, 2, action_index_before_inference=0)
    assert resolved == 2
    assert np.array_equal(queue.get_left_over(), original_new[2:])


def test_action_queue_delay_resolution_without_index_is_clamped_non_negative() -> None:
    original, processed = _chunks(3)
    queue = ActionQueue(rtc_enabled=True)
    assert queue.merge(original, processed, -4) == 0
    assert np.array_equal(queue.get_left_over(), original)


def test_action_queue_delay_clamps_to_the_shortest_chunk() -> None:
    original, processed = _chunks(3)
    queue = ActionQueue(rtc_enabled=True)
    assert queue.merge(original, processed, 99) == 3
    assert queue.qsize() == 0


def test_action_queue_task_lockstep_invariant_raises() -> None:
    original, processed = _chunks(3)
    queue = ActionQueue(rtc_enabled=True)
    queue.merge(original, processed, 0, task="pick")
    queue._task_queue.append("extra")  # deliberately break the invariant

    with pytest.raises(RuntimeError, match="lockstep"):
        queue.get()
    with pytest.raises(RuntimeError, match="lockstep"):
        queue.get_left_over()


def test_action_queue_rejects_non_numeric_and_non_2d_chunks() -> None:
    queue = ActionQueue(rtc_enabled=True)
    with pytest.raises(ValueError):
        queue.merge("not-a-grid", np.zeros((2, 2)), 0)
    with pytest.raises(ValueError):
        queue.merge([[1.0, 2.0]], np.zeros((2, 2, 2)), 0)
    with pytest.raises(TypeError):
        queue.merge(np.zeros((2, 2)), np.zeros((2, 2)), 1.5)  # type: ignore[arg-type]


# ============================================================================
# LatencyTracker
# ============================================================================
class FakeClock:
    """Deterministic monotonic clock with recorded sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_latency_tracker_measures_with_an_injected_clock() -> None:
    clock = FakeClock()
    tracker = LatencyTracker(window=3, monotonic=clock.monotonic)
    assert tracker.average() == 0.0
    assert tracker.median() == 0.0
    assert tracker.latest_delay(0.1) == 0

    tracker.start()
    clock.now += 0.3
    assert tracker.stop() == pytest.approx(0.3)
    tracker.record(0.1)
    tracker.record(0.2)

    assert len(tracker) == 3
    assert tracker.average() == pytest.approx(0.2)
    assert tracker.median() == pytest.approx(0.2)
    assert tracker.max() == pytest.approx(0.3)
    assert tracker.delay_in_steps(0.3, 0.1) == 3
    assert tracker.delay_in_steps(0.25, 0.1) == 2
    assert tracker.delay_in_steps(-1.0, 0.1) == 0
    assert tracker.latest_delay(0.1) == 2

    tracker.reset()
    assert len(tracker) == 0
    assert tracker.max() == 0.0


def test_latency_tracker_rejects_invalid_period_and_stop_before_start() -> None:
    tracker = LatencyTracker(monotonic=FakeClock().monotonic)
    with pytest.raises(RuntimeError):
        tracker.stop()
    with pytest.raises(ValueError):
        tracker.delay_in_steps(0.1, 0.0)
    with pytest.raises(ValueError):
        LatencyTracker(window=0)


# ============================================================================
# build_async_metadata
# ============================================================================
def test_build_async_metadata_matches_the_documented_schema() -> None:
    metadata = build_async_metadata(
        rtc=RTCAsyncRequest(
            prev_chunk_left_over=((1.0, 2.0), (3.0, 4.0)),
            inference_delay=3,
            execution_horizon=8,
            prefix_attention_schedule="linear",
            max_guidance_weight=10.0,
            hard_prefix=False,
            model_space=True,
            cache_start_index=6,
        ),
        base={"observation_id": "obs-1"},
    )

    assert metadata == {
        "observation_id": "obs-1",
        "async": {
            "schema": ASYNC_SCHEMA,
            "rtc": {
                "prev_chunk_left_over": [[1.0, 2.0], [3.0, 4.0]],
                "inference_delay": 3,
                "cache_start_index": 6,
                "execution_horizon": 8,
                "prefix_attention_schedule": "linear",
                "max_guidance_weight": 10.0,
                "hard_prefix": False,
                "model_space": True,
            },
        },
    }


def test_build_async_metadata_omits_disabled_algorithms() -> None:
    assert build_async_metadata() == {}
    assert build_async_metadata(base={"a": 1}) == {"a": 1}

    rtc_only = build_async_metadata(
        rtc=RTCAsyncRequest(prev_chunk_left_over=None, inference_delay=0, execution_horizon=2)
    )
    assert set(rtc_only["async"]) == {"schema", "rtc"}


def test_build_async_metadata_does_not_mutate_the_base() -> None:
    base = {"observation_id": "obs-1"}
    build_async_metadata(rtc=RTCAsyncRequest(None, 0, 1), base=base)
    assert base == {"observation_id": "obs-1"}


def test_rtc_request_rejects_malformed_input() -> None:
    with pytest.raises(TypeError):
        RTCAsyncRequest(None, 1.5, 4)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        RTCAsyncRequest(None, -1, 4)
    with pytest.raises(ValueError):
        RTCAsyncRequest(None, 1, -4)
    with pytest.raises(ValueError):
        RTCAsyncRequest(((1.0, float("inf")),), 1, 4)
    with pytest.raises(ValueError):
        RTCAsyncRequest(((1.0, 2.0), (3.0,)), 1, 4)
    with pytest.raises(ValueError):
        RTCAsyncRequest(None, 1, 4, "cosine")
    with pytest.raises(ValueError):
        RTCAsyncRequest(None, 1, 4, "linear", 0.0)
    with pytest.raises(TypeError):
        RTCAsyncRequest(None, 1, 4, "linear", 10.0, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cache_start_index"):
        RTCAsyncRequest(None, 1, 4, cache_start_index=-1)
    with pytest.raises(TypeError, match="cache_start_index"):
        RTCAsyncRequest(None, 1, 4, cache_start_index=True)  # type: ignore[arg-type]


# ============================================================================
# AsyncControlRuntime — fakes only, injected clock
# ============================================================================
class FakeRobot:
    robot_id = "fake-1"

    def __init__(self) -> None:
        self.actions: list[RobotAction] = []
        self.observations = 0

    def observe(self) -> RobotObservation:
        self.observations += 1
        return RobotObservation(timestamp_s=float(self.observations), values={"joint_0": 0.0, "joint_1": 0.0})

    def execute(self, action: RobotAction) -> None:
        self.actions.append(action)

    def stop(self) -> None:
        return None

    def close(self) -> None:
        return None


class FakeInferenceClient:
    """Blocking client that consumes fake-clock time while "inferring"."""

    policy_action_space = "pi05.action_chunk.v1"

    def __init__(
        self,
        *,
        chunk: list[list[float]],
        clock: FakeClock,
        period_s: float,
        latency_steps: int,
        robot: FakeRobot,
    ) -> None:
        self.requests: list[PolicyObservation] = []
        self.executed_at_request: list[int] = []
        self._base = np.asarray(chunk, dtype=np.float64)
        self._clock = clock
        self._period_s = period_s
        self._latency_steps = latency_steps
        self._robot = robot
        self._counter = 0

    def open_session(self, *, robot_id: str, action_space: str, metadata=None) -> Session:
        return Session("session-1", 0)

    def step(self, observation: PolicyObservation) -> PolicyResult:
        self.requests.append(observation)
        self.executed_at_request.append(len(self._robot.actions))
        self._counter += 1
        self._clock.now += self._latency_steps * self._period_s
        data = (self._base + 100.0 * (self._counter - 1)).tolist()
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
        return None


class FakeMapper:
    policy_action_space = "pi05.action_chunk.v1"

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
            metadata={"robot_timestamp_s": observation.timestamp_s},
        )

    def map_result(self, result: PolicyResult) -> tuple[RobotAction, ...]:
        data = result.actions[0].values["data"]
        return tuple(
            RobotAction(timestamp_s=0.0, values={"type": "joint_position", "vector": [float(v) for v in row]})
            for row in data
        )


FRAME = CameraFrame("observation.images.front", "image/jpeg", b"jpeg")


def _build_runtime(
    *,
    chunk: list[list[float]],
    config: AsyncRuntimeConfig,
    control_hz: float = 10.0,
    latency_steps: int = 2,
    chunk_steps: int | None = None,
):
    clock = FakeClock()
    robot = FakeRobot()
    client = FakeInferenceClient(
        chunk=chunk,
        clock=clock,
        period_s=0.0 if control_hz == 0 else 1.0 / control_hz,
        latency_steps=latency_steps,
        robot=robot,
    )
    runtime = AsyncControlRuntime(
        robot,
        client,
        instruction="pick up the block",
        mapper=FakeMapper(),
        chunk_steps=chunk_steps if chunk_steps is not None else len(chunk),
        control_hz=control_hz,
        config=config,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    return runtime, robot, client, clock


def test_async_rtc_sends_the_leftover_and_the_consumed_delay() -> None:
    chunk = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
    config = AsyncRuntimeConfig(
        enabled=True,
        rtc_enabled=True,
        overlap_steps=3,
        initial_inference_delay=2,
        execution_horizon=4,
        max_guidance_weight=7.5,
        send_model_space_prefix=True,
    )
    runtime, _robot, client, _clock = _build_runtime(chunk=chunk, config=config, latency_steps=2)

    for _ in range(4):
        runtime.step((FRAME,))
    runtime._collect_job(block=True)

    assert len(client.requests) == 2
    block = client.requests[1].metadata["async"]["rtc"]
    trigger = len(chunk) - config.overlap_steps
    assert block["prev_chunk_left_over"] == chunk[trigger:]
    assert block["inference_delay"] == 2
    assert block["execution_horizon"] == 4
    assert block["max_guidance_weight"] == 7.5
    assert block["hard_prefix"] is False

    event = runtime.events[-1]
    assert event.actions_consumed == 2
    assert event.inference_delay == 2
    assert event.resolved_delay == 2


def test_async_rtc_fast_response_waits_for_and_skips_the_promised_prefix() -> None:
    chunk = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
    config = AsyncRuntimeConfig(
        enabled=True,
        rtc_enabled=True,
        overlap_steps=3,
        initial_inference_delay=2,
    )
    runtime, _robot, _client, _clock = _build_runtime(chunk=chunk, config=config, latency_steps=0)

    for _ in range(4):
        runtime.step((FRAME,))
    runtime._collect_job(block=True)

    event = runtime.events[-1]
    assert event.latency_delay == 0
    assert event.actions_consumed == 2
    assert event.resolved_delay == 2


def test_async_rtc_session_cache_sends_chunk_offset_not_latency() -> None:
    chunk = [[float(index), float(index)] for index in range(8)]
    config = AsyncRuntimeConfig(
        enabled=True,
        rtc_enabled=True,
        overlap_steps=5,
        initial_inference_delay=2,
        execution_horizon=3,
        prefix_attention_schedule="exp",
        max_guidance_weight=5.0,
        send_model_space_prefix=False,
    )
    runtime, _robot, client, _clock = _build_runtime(
        chunk=chunk,
        config=config,
        latency_steps=2,
    )

    for _ in range(4):
        runtime.step((FRAME,))
    runtime._collect_job(block=True)

    assert len(client.requests) == 2
    bootstrap = client.requests[0].metadata["async"]["rtc"]
    replan = client.requests[1].metadata["async"]["rtc"]
    assert bootstrap["cache_start_index"] == 0
    assert replan["cache_start_index"] == 3  # H - overlap, not Δ=2
    assert replan["inference_delay"] == 2
    assert replan["prev_chunk_left_over"] is None


def test_async_disabled_is_identical_to_the_synchronous_runtime() -> None:
    chunk = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
    config = AsyncRuntimeConfig(enabled=False, rtc_enabled=True, overlap_steps=1)
    runtime, robot, client, clock = _build_runtime(
        chunk=chunk, config=config, control_hz=20.0, latency_steps=0, chunk_steps=3
    )

    result = runtime.step((FRAME,))

    assert result is not None
    assert [action.values["vector"] for action in robot.actions] == chunk
    assert clock.sleeps == [0.05, 0.05]
    assert len(client.requests) == 1
    assert "async" not in client.requests[0].metadata
    assert runtime.events == []


def test_async_runtime_paces_one_action_per_control_period() -> None:
    chunk = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
    config = AsyncRuntimeConfig(
        enabled=True,
        overlap_steps=1,
    )
    runtime, robot, _client, clock = _build_runtime(
        chunk=chunk,
        config=config,
        control_hz=20.0,
        latency_steps=0,
    )

    runtime.step((FRAME,))
    runtime.step((FRAME,))
    runtime.step((FRAME,))

    assert len(robot.actions) == 3
    assert clock.sleeps == [0.05, 0.05]


def test_async_runtime_zero_control_hz_disables_software_pacing() -> None:
    chunk = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
    config = AsyncRuntimeConfig(
        enabled=True,
        overlap_steps=1,
    )
    runtime, robot, _client, clock = _build_runtime(
        chunk=chunk,
        config=config,
        control_hz=0.0,
        latency_steps=0,
    )

    runtime.step((FRAME,))
    runtime.step((FRAME,))
    runtime.step((FRAME,))

    assert len(robot.actions) == 3
    assert clock.sleeps == []
    assert runtime.action_period_s == 0.0


def test_async_runtime_config_accepts_artificial_inference_delay() -> None:
    config = AsyncRuntimeConfig.from_options({"async": {"mode": "naive", "artificial_inference_delay_s": 0.2}})

    assert config.enabled is True
    assert config.artificial_inference_delay_s == pytest.approx(0.2)

    with pytest.raises(ValueError, match="artificial_inference_delay_s"):
        AsyncRuntimeConfig(artificial_inference_delay_s=-0.1)


def test_async_runtime_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        AsyncRuntimeConfig(enabled=True, overlap_steps=-1)
    with pytest.raises(ValueError):
        AsyncRuntimeConfig(enabled=True, max_guidance_weight=float("inf"))
