"""Map typed ActiveVLN HTTP actions onto metric Go2 commands, without model imports."""

from __future__ import annotations

import hashlib
import json
import math
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence

from embodirun.application.navigation.chunk_session import NavigationChunk
from embodirun.model_services.backends.embodiinfer.http import EmbodiInferHttpClient
from embodirun.model_services.contracts import ImagePayload, PolicyObservation, PolicyResult, Session
from embodirun.robots.mobile_base.discrete import NavigationCommand, NavigationCommandKind
from embodirun.robots.mobile_base.request import NavigationRequest

ACTION_SPACE = "activevln.r2r.discrete.v1"
EXECUTION_PREFIX_SCHEMA = "activevln.execution_prefix.v1"


def check_capabilities(client: EmbodiInferHttpClient, *, overlap: bool = False) -> dict[str, object]:
    """Require the batch-one ActiveVLN R2R contract before opening a session."""
    adapter = client.capabilities().get("adapter")
    if not isinstance(adapter, Mapping) or adapter.get("action_space") != ACTION_SPACE:
        raise ValueError(f"inference service must advertise {ACTION_SPACE}")
    if adapter.get("model") != "activevln" or adapter.get("recurrent") is not True:
        raise ValueError("inference service must advertise recurrent ActiveVLN")
    if adapter.get("max_batch_size") != 1:
        raise ValueError("Go2 requires the batch=1 ActiveVLN serving profile")
    if overlap and adapter.get("execution_prefix") != EXECUTION_PREFIX_SCHEMA:
        raise ValueError("async navigation requires an execution-prefix-aware ActiveVLN service")
    return dict(adapter)


def navigation_chunk(result: PolicyResult, observation_sequence: int) -> NavigationChunk:
    """Validate already-parsed [3,2] R2R rows and convert cm/degrees to metric commands.

    This does not parse model text. STOP ends execution even if later rows are
    present; padding and action magnitudes must satisfy the published grammar.
    """
    if result.action_space != ACTION_SPACE or len(result.actions) != 1:
        raise ValueError("unexpected ActiveVLN action space or action count")
    action = result.actions[0]
    values = action.values
    if action.kind != "discrete_chunk" or values.get("parsed_valid") is not True:
        raise ValueError("ActiveVLN did not return a valid parsed discrete chunk")
    rows = values.get("rows")
    if not isinstance(rows, (list, tuple)) or len(rows) != 3:
        raise ValueError("ActiveVLN rows must have shape [3,2]")
    commands: list[NavigationCommand] = []
    padded = stopped = False
    for row in rows:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 2:
            raise ValueError("ActiveVLN rows must have shape [3,2]")
        kind, value = row
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in row):
            raise ValueError("action entries must be finite numbers")
        if kind not in (-1, 0, 1, 2, 3):
            raise ValueError("unknown discrete action ID")
        allowed = (0,) if kind in (-1, 0) else ((25, 50, 75) if kind == 1 else (15, 30, 45))
        if value not in allowed:
            raise ValueError("action magnitude exceeds the R2R grammar")
        if kind == -1:
            padded = True
            continue
        if padded:
            raise ValueError("padding must be trailing")
        if stopped:
            continue
        if kind == 0:
            commands.append(NavigationCommand(NavigationCommandKind.STOP))
            stopped = True
        elif kind == 1:
            commands.append(NavigationCommand(NavigationCommandKind.MOVE_FORWARD, distance_m=value / 100))
        else:
            direction = NavigationCommandKind.TURN_LEFT if kind == 2 else NavigationCommandKind.TURN_RIGHT
            commands.append(NavigationCommand(direction, angle_deg=value))
    return NavigationChunk(observation_sequence, tuple(commands))


class ActiveVLNPolicy:
    """Serialize one remote recurrent session, including optional continuation context."""

    def __init__(
        self,
        client: EmbodiInferHttpClient,
        *,
        robot_id: str = "go2",
        event_sink: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.client, self.robot_id, self.event_sink = client, robot_id, event_sink
        self.session: Session | None = None
        self._step = 0
        self._episode: str | None = None
        self._failed = False
        self._lock = threading.RLock()

    def plan(self, request: NavigationRequest) -> NavigationChunk:
        """Send one RGB observation and validate the committed response identity."""
        with self._lock:
            return self._plan(request)

    def plan_after(self, request: NavigationRequest, remaining: tuple[NavigationCommand, ...]) -> NavigationChunk:
        """Plan a suffix after the immutable previous chunk, without replaying its tail."""
        with self._lock:
            if self._step == 0:
                raise ValueError("overlap requires a committed previous model turn")
            rows = []
            for command in remaining:
                if command.kind is NavigationCommandKind.MOVE_FORWARD:
                    rows.append([1, float(command.distance_m * 100)])
                elif command.kind in (NavigationCommandKind.TURN_LEFT, NavigationCommandKind.TURN_RIGHT):
                    rows.append([2 if command.kind is NavigationCommandKind.TURN_LEFT else 3, float(command.angle_deg)])
                else:
                    raise ValueError("a committed STOP cannot have a continuation")
            prefix = {"schema": EXECUTION_PREFIX_SCHEMA, "previous_step_id": self._step - 1, "remaining_actions": rows}
            return self._plan(request, prefix)

    def _plan(self, request: NavigationRequest, prefix: dict[str, object] | None = None) -> NavigationChunk:
        if self._failed:
            raise RuntimeError("failed policy session cannot be reused")
        observation = request.observation
        try:
            if self.session is None:
                if not observation.reset:
                    raise ValueError("the first observation must reset episode history")
                self.session = self.client.open_session(
                    robot_id=self.robot_id,
                    action_space=ACTION_SPACE,
                    metadata={"episode_id": observation.episode_id},
                )
                self._episode = observation.episode_id
            elif observation.reset or observation.episode_id != self._episode:
                raise ValueError("episode changed within a recurrent session")
            image = observation.latest_rgb
            response = self.client.step(
                PolicyObservation(
                    session_id=self.session.session_id,
                    request_id=uuid.uuid4().hex,
                    step_id=self._step,
                    instruction=request.instruction,
                    # The frozen optimized NX adapter reserves slot zero for its
                    # single resident KV cache. This is not robot proprioception.
                    state={"benchmark_slot": 0},
                    images=(ImagePayload("observation.images.rgb", image.media_type, image.data),),
                    metadata={
                        "episode_id": self._episode,
                        "observation_sequence": observation.sequence,
                        **({"activevln_execution": prefix} if prefix is not None else {}),
                    },
                )
            )
            if response.session_revision != self.session.revision + 1:
                raise ValueError("inference session revision did not advance by one")
            chunk = navigation_chunk(response, observation.sequence)
            if prefix is not None:
                digest = hashlib.sha256(
                    json.dumps(prefix, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                ).hexdigest()
                if response.actions[0].values.get("execution_prefix_sha256") != digest:
                    raise ValueError("inference did not acknowledge the executed prefix context")
            self.session = Session(response.session_id, response.session_revision)
            self._step += 1
            if self.event_sink is not None:
                self.event_sink(
                    {
                        "event": "policy_result",
                        "step_id": response.step_id,
                        "session_revision": response.session_revision,
                        "policy_revision": response.policy_revision,
                        "timing": dict(response.timing),
                        "observation_sequence": observation.sequence,
                    }
                )
            return chunk
        except BaseException:
            self._failed = True
            raise

    def close(self) -> None:
        """Release remote recurrent state, including after a rejected action."""
        with self._lock:
            self._failed = True
            if self.session is not None:
                session, self.session = self.session, None
                self.client.close(session.session_id)
