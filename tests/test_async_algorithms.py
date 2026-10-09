"""Execution-side RTC and VLASH algorithms.

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
    VlashAsyncRequest,
    VlashScheduler,
    build_async_metadata,
    estimate_future_state,
    prefix_weights,
    quantize_actions,
    roll_state_forward,
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
@pytest.mark.parametrize("case", _golden()["vlash_roll_forward"], ids=lambda case: case["name"])
def test_roll_state_forward_matches_golden_vectors(case: dict) -> None:
    rolled = roll_state_forward(case["state"], case["actions"], case["delay"], case["semantics"])
    expected = np.asarray(case["expected"], dtype=np.float64)
    assert rolled.shape == expected.shape, case["name"]
    assert np.allclose(rolled, expected, atol=1e-6), case["name"]


@pytest.mark.parametrize("case", _golden()["rtc_prefix_weights"], ids=lambda case: case["name"])
def test_prefix_weights_match_golden_vectors(case: dict) -> None:
    weights = prefix_weights(case["start"], case["end"], case["total"], case["schedule"])
    expected = np.asarray(case["expected"], dtype=np.float64)
    assert weights.shape == (case["total"],), case["name"]
    assert np.allclose(weights, expected, atol=1e-6), (case["name"], weights.tolist())


def test_runtime_metadata_round_trips_as_json() -> None:
    """The async block must be JSON-serializable as the HTTP client sends it."""

    payload = build_async_metadata(
        rtc=RTCAsyncRequest(prev_chunk_left_over=((1.0, 2.0),), inference_delay=1, execution_horizon=2),
        vlash=VlashAsyncRequest(
            state_fields=("joint_0", "joint_1"),
            pending_actions=((0.5, 0.25),),
            delay=1,
            action_space="delta",
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
# roll_state_forward — semantics and validation
# ============================================================================
def test_roll_state_forward_zero_delay_is_identity() -> None:
    state = np.array([0.5, -0.25])
    actions = np.array([[1.0, 2.0], [3.0, 4.0]])
    for semantics in ("absolute", "delta"):
        assert np.array_equal(roll_state_forward(state, actions, 0, semantics), state)


def test_roll_state_forward_full_consumption() -> None:
    state = np.array([0.0, 0.0])
    actions = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert np.array_equal(roll_state_forward(state, actions, 3, "absolute"), np.array([5.0, 6.0]))
    assert np.array_equal(roll_state_forward(state, actions, 3, "delta"), np.array([9.0, 12.0]))


def test_roll_state_forward_batched_and_rank_preserving() -> None:
    state = np.array([[0.0, 0.0], [1.0, 1.0]])
    actions = np.ones((2, 3, 2))
    rolled = roll_state_forward(state, actions, 2, "delta")
    assert rolled.shape == (2, 2)
    assert np.array_equal(rolled, np.array([[2.0, 2.0], [3.0, 3.0]]))
    single = roll_state_forward(np.zeros(2), np.ones((3, 2)), 1, "absolute")
    assert single.shape == (2,)


@pytest.mark.parametrize(
    ("state", "actions", "delay", "match"),
    [
        (np.zeros(2), np.zeros((3, 2)), -1, "non-negative"),
        (np.zeros(2), np.zeros((3, 2)), 4, "exceeds"),
        (np.zeros(2), np.zeros((3, 3)), 1, "width"),
        (np.zeros((2, 2)), np.zeros((1, 3, 2)), 1, "batch"),
    ],
)
def test_roll_state_forward_validation_errors(state, actions, delay, match) -> None:
    with pytest.raises(ValueError, match=match):
        roll_state_forward(state, actions, delay)


def test_roll_state_forward_rejects_bad_delay_and_shapes() -> None:
    with pytest.raises(ValueError, match="integer"):
        roll_state_forward(np.zeros(2), np.zeros((3, 2)), 1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        roll_state_forward(np.zeros((1, 1, 2)), np.zeros((3, 2)), 1)
    with pytest.raises(ValueError):
        roll_state_forward(np.zeros(2), np.zeros(3), 1)
    with pytest.raises(ValueError):
        roll_state_forward(np.zeros(2), np.zeros((3, 2)), 1, "relative")


# ============================================================================
# quantize_actions
# ============================================================================
def test_quantize_actions_sums_delta_groups() -> None:
    actions = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
    quantized = quantize_actions(actions, 2, "delta")
    assert np.array_equal(quantized, np.array([[4.0, 6.0], [12.0, 14.0], [9.0, 10.0]]))


def test_quantize_actions_keeps_last_absolute_of_each_group() -> None:
    actions = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
    quantized = quantize_actions(actions, 2, "absolute")
    assert np.array_equal(quantized, np.array([[3.0, 4.0], [7.0, 8.0], [9.0, 10.0]]))


@pytest.mark.parametrize("ratio", [0, 1, -3])
def test_quantize_actions_ratio_at_most_one_is_a_noop(ratio: int) -> None:
    actions = [[1.0, 2.0], [3.0, 4.0]]
    assert quantize_actions(actions, ratio, "delta") is actions


def test_quantize_actions_validation() -> None:
    with pytest.raises(TypeError):
        quantize_actions([[1.0]], 1.5, "delta")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        quantize_actions([[[1.0]]], 2, "delta")
    with pytest.raises(ValueError):
        quantize_actions([[1.0]], 2, "relative")


# ============================================================================
# VlashScheduler
# ============================================================================
def test_vlash_scheduler_trigger_points() -> None:
    scheduler = VlashScheduler(chunk_length=10, overlap_steps=3)
    assert not scheduler.should_launch_next_inference(6)
    assert scheduler.should_launch_next_inference(7)
    assert scheduler.should_switch_chunk(0)
    assert not scheduler.should_switch_chunk(1)
    assert scheduler.should_fetch_observation(0, running=False)
    assert not scheduler.should_fetch_observation(0, running=True)
    assert scheduler.should_fetch_observation(7, running=True)


def test_vlash_scheduler_scales_overlap_by_quantization_ratio() -> None:
    scheduler = VlashScheduler(chunk_length=10, overlap_steps=3, action_quant_ratio=2)
    assert scheduler.effective_overlap_steps == 6
    assert scheduler.should_launch_next_inference(4)
    assert not scheduler.should_launch_next_inference(7)


@pytest.mark.parametrize(
    ("chunk_length", "overlap_steps", "ratio"),
    [(0, 0, 1), (10, -1, 1), (10, 3, 0), (10, 6, 2)],
)
def test_vlash_scheduler_validation(chunk_length: int, overlap_steps: int, ratio: int) -> None:
    with pytest.raises(ValueError):
        VlashScheduler(chunk_length=chunk_length, overlap_steps=overlap_steps, action_quant_ratio=ratio)


# ============================================================================
# estimate_future_state
# ============================================================================
def test_estimate_future_state_reduces_to_the_last_action() -> None:
    state = np.array([0.5, -0.25])
    chunk = np.arange(10, dtype=np.float64).reshape(5, 2)
    remaining = len(chunk) - 2

    future = estimate_future_state(state, chunk, 2, remaining, "absolute")
    assert np.array_equal(future, chunk[-1])


def test_estimate_future_state_from_chunk_start_reduces_to_last_action() -> None:
    state = np.array([0.0, 0.0])
    chunk = np.arange(6, dtype=np.float64).reshape(3, 2)
    assert np.array_equal(estimate_future_state(state, chunk, 0, 3, "absolute"), chunk[-1])


def test_estimate_future_state_delta_accumulates_from_the_index() -> None:
    state = np.array([1.0, 1.0])
    chunk = np.arange(8, dtype=np.float64).reshape(4, 2)
    future = estimate_future_state(state, chunk, 1, 3, "delta")
    assert np.array_equal(future, state + chunk[1:4].sum(axis=0))


def test_estimate_future_state_validation() -> None:
    with pytest.raises(ValueError):
        estimate_future_state(np.zeros(2), np.zeros((3, 2)), 4, 1)
    with pytest.raises(TypeError):
        estimate_future_state(np.zeros(2), np.zeros((3, 2)), 1.5, 1)  # type: ignore[arg-type]


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
        ),
        vlash=VlashAsyncRequest(
            state_fields=("joint_0", "joint_1"),
            pending_actions=((5.0, 6.0), (7.0, 8.0), (9.0, 10.0), (11.0, 12.0)),
            delay=4,
            action_space="absolute",
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
                "execution_horizon": 8,
                "prefix_attention_schedule": "linear",
                "max_guidance_weight": 10.0,
                "hard_prefix": False,
            },
            "vlash": {
                "state_fields": ["joint_0", "joint_1"],
                "pending_actions": [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
                "delay": 4,
                "action_space": "absolute",
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

    vlash_only = build_async_metadata(
        vlash=VlashAsyncRequest(state_fields=("joint_0",), pending_actions=((1.0,),), delay=1)
    )
    assert set(vlash_only["async"]) == {"schema", "vlash"}


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


def test_vlash_request_rejects_malformed_input() -> None:
    with pytest.raises(ValueError):
        VlashAsyncRequest((), ((1.0,),), 1)
    with pytest.raises(ValueError):
        VlashAsyncRequest(("joint_0", "joint_0"), ((1.0, 1.0),), 1)
    with pytest.raises(ValueError):
        VlashAsyncRequest(("joint_0",), ((1.0,), (2.0, 3.0)), 1)  # rows differ in width
    with pytest.raises(ValueError):
        VlashAsyncRequest(("joint_0",), ((),), 0)  # an empty action vector
    with pytest.raises(ValueError):
        VlashAsyncRequest(("joint_0",), ((1.0,),), 2)
    with pytest.raises(ValueError):
        VlashAsyncRequest(("joint_0",), ((1.0,),), 1, "relative")
    with pytest.raises(ValueError):
        VlashAsyncRequest(("joint_0",), ((float("nan"),),), 1)


def test_vlash_request_accepts_a_vector_valued_state_field() -> None:
    """A named field may hold a whole vector, which is the deployment case.

    The server reads each field's width from the state itself, so the client must
    NOT require one scalar column per field name. Refusing this shape makes VLASH
    unusable with LIBERO/π0.5, whose proprioception is a single
    ``observation.state`` entry holding the entire joint vector.
    """
    request = VlashAsyncRequest(
        state_fields=("observation.state",),
        pending_actions=((0.1, 0.2, 0.3), (0.4, 0.5, 0.6)),
        delay=2,
        action_space="delta",
    )
    assert request.state_fields == ("observation.state",)
    assert len(request.pending_actions) == 2
    assert all(len(row) == 3 for row in request.pending_actions)

    # Several fields of mixed arity are also fine: widths are a server-side concern.
    mixed = VlashAsyncRequest(
        state_fields=("arm", "gripper"),
        pending_actions=((1.0, 2.0, 3.0),),
        delay=1,
        action_space="delta",
    )
    assert mixed.pending_actions == ((1.0, 2.0, 3.0),)


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
        period_s=1.0 / control_hz,
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


def test_async_vlash_launches_before_exhaustion_and_sends_the_consumed_delay() -> None:
    chunk = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
    config = AsyncRuntimeConfig(
        enabled=True,
        vlash_enabled=True,
        overlap_steps=2,
        state_fields=("joint_0", "joint_1"),
        action_space="absolute",
    )
    runtime, robot, client, _clock = _build_runtime(chunk=chunk, config=config)

    for _ in range(6):
        runtime.step((FRAME,))

    # Inference #0 is the bootstrap; inference #1 is the overlap launch.
    assert len(client.requests) == 2
    assert client.executed_at_request[1] == len(chunk) - config.overlap_steps
    assert client.executed_at_request[1] < len(chunk)

    block = client.requests[1].metadata["async"]
    assert block["schema"] == ASYNC_SCHEMA
    assert block["vlash"]["delay"] == config.overlap_steps
    assert block["vlash"]["pending_actions"] == chunk[len(chunk) - config.overlap_steps :]
    assert block["vlash"]["state_fields"] == ["joint_0", "joint_1"]

    event = runtime.events[-1]
    assert event.actions_consumed == block["vlash"]["delay"]
    assert event.resolved_delay == block["vlash"]["delay"]
    assert event.index_before_inference == len(chunk) - config.overlap_steps

    # The whole first chunk was executed, then the second chunk's first actions.
    assert [action.values["vector"] for action in robot.actions[:5]] == chunk
    assert robot.actions[5].values["vector"] == [100.0, 100.0]


def test_async_vlash_quantizes_pending_actions_at_send_time() -> None:
    chunk = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
    config = AsyncRuntimeConfig(
        enabled=True,
        vlash_enabled=True,
        overlap_steps=2,
        action_quant_ratio=2,
        action_space="delta",
        state_fields=("joint_0", "joint_1"),
    )
    runtime, _robot, client, _clock = _build_runtime(chunk=chunk, config=config)

    for _ in range(6):
        runtime.step((FRAME,))

    block = client.requests[1].metadata["async"]["vlash"]
    # The four remaining micro-actions collapse into two macro-actions.
    assert block["pending_actions"] == [[8.0, 10.0], [16.0, 18.0]]
    assert block["delay"] == 2
    assert block["action_space"] == "delta"


def test_async_rtc_sends_the_leftover_and_the_consumed_delay() -> None:
    chunk = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
    config = AsyncRuntimeConfig(
        enabled=True,
        rtc_enabled=True,
        overlap_steps=3,
        initial_inference_delay=2,
        execution_horizon=4,
        max_guidance_weight=7.5,
    )
    runtime, _robot, client, _clock = _build_runtime(chunk=chunk, config=config, latency_steps=2)

    for _ in range(3):
        runtime.step((FRAME,))

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


def test_async_runtime_requires_state_fields_for_vlash() -> None:
    with pytest.raises(ValueError, match="state_fields"):
        AsyncRuntimeConfig(enabled=True, vlash_enabled=True, state_fields=())


def test_async_runtime_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        AsyncRuntimeConfig(enabled=True, overlap_steps=-1)
    with pytest.raises(ValueError):
        AsyncRuntimeConfig(enabled=True, action_quant_ratio=0)
    with pytest.raises(ValueError):
        AsyncRuntimeConfig(enabled=True, action_space="relative")
    with pytest.raises(ValueError):
        AsyncRuntimeConfig(enabled=True, max_guidance_weight=float("inf"))
