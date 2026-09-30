"""Contract tests for the LIBERO + Pi0.5 binding mapper.

The tests use synthetic observations and encoded frames only: the mapper must
stay runnable in the plain development environment, without LIBERO, a GPU, or a
policy server.
"""

from __future__ import annotations

import io
import math
import sys

import pytest
from PIL import Image

from embodirun.bindings import binding_definition
from embodirun.bindings.simulated.libero.pi05 import (
    ACTION_FEATURE_NAMES,
    MAXIMUM_CHUNK_STEPS,
    POLICY_ACTION_SPACE,
    LiberoPi05Mapper,
    LiberoPi05MapperError,
)
from embodirun.bindings.simulated.libero.pi05.mapper import (
    STATE_FIELD,
    WIRE_IMAGE_FIELDS,
    libero_state_from_components,
    quaternion_xyzw_to_rotation_vector,
)
from embodirun.model_services import (
    ImagePayload,
    PolicyAction,
    PolicyObservation,
    PolicyResult,
)
from embodirun.robots import RobotAction, RobotObservation
from embodirun.robots.sensors.cameras import CameraFrame
from embodirun.simulators import simulator_definition

STATE = (0.125, -0.25, 0.5, 0.0, 0.0, math.pi / 3.0, 0.04, -0.04)
IMAGE_SIZE = (8, 6)  # (width, height)
QUARTER_TURN_Z = (0.0, 0.0, math.sin(math.pi / 6.0), math.cos(math.pi / 6.0))


def _encoded_frame(
    name: str,
    *,
    image_format: str = "PNG",
    size: tuple[int, int] = IMAGE_SIZE,
    mode: str = "RGB",
    color: object = (17, 34, 51),
) -> CameraFrame:
    image = Image.new(mode, size, color)
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    mime_type = {"PNG": "image/png", "JPEG": "image/jpeg"}[image_format]
    return CameraFrame(name=name, mime_type=mime_type, data=buffer.getvalue())


def _adapter_frames() -> tuple[CameraFrame, CameraFrame]:
    """Frames exactly as ``LiberoAdapter`` emits them: qualified names, JPEG."""

    return (
        _encoded_frame("observation.images.image", image_format="JPEG", color=(200, 30, 40)),
        _encoded_frame("observation.images.wrist_image", image_format="JPEG", color=(30, 200, 60)),
    )


def _observation(values: object | None = None, *, timestamp_s: float = 12.5) -> RobotObservation:
    if values is None:
        values = {STATE_FIELD: list(STATE)}
    return RobotObservation(timestamp_s=timestamp_s, values=values)


def _result(
    rows: object | None = None,
    *,
    feature_names: object = ACTION_FEATURE_NAMES,
    action_space: str = POLICY_ACTION_SPACE,
    kind: str = "action_chunk",
    actions: object | None = None,
) -> PolicyResult:
    if actions is None:
        data = [[float(index), 1.0, 2.0, 3.0, 4.0, 5.0, float(index + 6)] for index in range(3)]
        if rows is not None:
            data = rows
        values: dict[str, object] = {"data": data}
        if feature_names is not None:
            values["feature_names"] = feature_names
        actions = (PolicyAction(kind, values),)
    return PolicyResult(
        request_id="request-1",
        session_id="session-1",
        step_id=0,
        session_revision=1,
        action_space=action_space,
        actions=actions,
    )


def _map(observation: RobotObservation | None = None, mapper: LiberoPi05Mapper | None = None) -> PolicyObservation:
    return (mapper or LiberoPi05Mapper()).map_observation(
        observation if observation is not None else _observation(),
        session_id="session-1",
        request_id="request-1",
        step_id=0,
        instruction="pick up the black bowl",
        frames=_adapter_frames(),
    )


def test_binding_is_discoverable_and_matches_the_libero_embodiment() -> None:
    binding = binding_definition("simulated.libero.pi05")
    simulator = simulator_definition("libero")

    assert binding.kind == "simulated.libero.pi05"
    assert binding.model_kind == "pi05"
    assert binding.robot_kind == simulator.embodiment_kind == "franka.panda.eef"
    assert binding.maximum_chunk_steps == MAXIMUM_CHUNK_STEPS == 10
    assert binding.adapter_config == {
        "state_fields": ("observation.state",),
        "image_fields": ("image", "wrist_image"),
        "action_feature_names": ACTION_FEATURE_NAMES,
    }
    assert binding.mapper_factory().policy_action_space == POLICY_ACTION_SPACE

    # The mapper must consume the frame names the LIBERO adapter declares.
    aliases = {alias for _, frame_aliases in WIRE_IMAGE_FIELDS for alias in frame_aliases}
    assert set(simulator.image_fields) <= aliases


@pytest.mark.parametrize(
    ("quaternion", "expected"),
    [
        ((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),  # identity
        ((0.0, 0.0, 0.0, -1.0), (0.0, 0.0, 0.0)),  # a full 2*pi turn is the identity
        ((0.0, 0.0, 1.0, 0.0), (0.0, 0.0, math.pi)),  # 180 degrees about z
        ((1.0, 0.0, 0.0, 0.0), (math.pi, 0.0, 0.0)),  # 180 degrees about x
        (QUARTER_TURN_Z, (0.0, 0.0, math.pi / 3.0)),  # hand-computed 60 degrees about z
        (
            (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0),
            (math.pi * math.sqrt(0.5), math.pi * math.sqrt(0.5), 0.0),
        ),  # 180 degrees about (1, 1, 0)/sqrt(2)
    ],
)
def test_quaternion_conversion_matches_hand_computed_rotations(
    quaternion: tuple[float, float, float, float],
    expected: tuple[float, float, float],
) -> None:
    assert quaternion_xyzw_to_rotation_vector(quaternion) == pytest.approx(expected, abs=1e-12)


def test_quaternion_conversion_has_well_defined_limits() -> None:
    # Near-zero angle: w rounds to one, so the vector part carries no usable axis.
    assert quaternion_xyzw_to_rotation_vector((1e-13, -2e-13, 3e-13, 1.0)) == (0.0, 0.0, 0.0)
    # Non-unit w is clamped rather than producing NaN.
    assert quaternion_xyzw_to_rotation_vector((0.0, 0.0, 0.0, 1.0000001)) == (0.0, 0.0, 0.0)
    assert quaternion_xyzw_to_rotation_vector((0.0, 0.0, 0.0, -1.0000001)) == (0.0, 0.0, 0.0)
    # Near pi the axis stays conditioned and the angle approaches pi.
    near_pi = quaternion_xyzw_to_rotation_vector((0.0, 0.0, 1.0, 1e-12))
    assert near_pi[:2] == (0.0, 0.0)
    assert near_pi[2] == pytest.approx(math.pi, abs=1e-9)
    assert math.isfinite(sum(abs(value) for value in near_pi))


@pytest.mark.parametrize(
    ("quaternion", "message"),
    [
        ((0.0, 0.0, 1.0), "must contain 4 values"),
        ((0.0, 0.0, 0.0, 1.0, 2.0), "must contain 4 values"),
        ((0.0, 0.0, 0.0, math.nan), "must be finite"),
        ("wxyz", "must be a sequence"),
    ],
)
def test_quaternion_conversion_rejects_malformed_input(quaternion: object, message: str) -> None:
    with pytest.raises(LiberoPi05MapperError, match=message):
        quaternion_xyzw_to_rotation_vector(quaternion)  # type: ignore[arg-type]


def test_state_uses_exactly_one_eight_float_entry() -> None:
    request = _map()

    assert list(request.state) == [STATE_FIELD]
    assert request.state[STATE_FIELD] == list(STATE)
    assert len(request.state[STATE_FIELD]) == 8


def test_state_is_rebuilt_from_raw_libero_components_in_adapter_order() -> None:
    values = {
        "robot_state": {
            "eef": {"pos": [0.1, 0.2, 0.3], "quat": list(QUARTER_TURN_Z)},
            "gripper": {"qpos": [0.04, -0.04]},
        }
    }
    request = _map(_observation(values))

    expected = libero_state_from_components([0.1, 0.2, 0.3], list(QUARTER_TURN_Z), [0.04, -0.04])
    assert request.state[STATE_FIELD] == pytest.approx(list(expected))
    assert request.state[STATE_FIELD][:3] == pytest.approx([0.1, 0.2, 0.3])
    assert request.state[STATE_FIELD][3:6] == pytest.approx([0.0, 0.0, math.pi / 3.0])
    assert request.state[STATE_FIELD][6:] == pytest.approx([0.04, -0.04])


def test_state_is_rebuilt_from_flat_libero_field_names() -> None:
    values = {
        "robot0_eef_pos": [0.1, 0.2, 0.3],
        "robot0_eef_quat": list(QUARTER_TURN_Z),
        "robot0_gripper_qpos": [0.04, -0.04],
    }
    request = _map(_observation(values))

    assert request.state[STATE_FIELD] == pytest.approx(
        list(libero_state_from_components([0.1, 0.2, 0.3], list(QUARTER_TURN_Z), [0.04, -0.04]))
    )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({STATE_FIELD: [0.0] * 7}, "must contain 8 values"),
        ({STATE_FIELD: [0.0] * 9}, "must contain 8 values"),
        ({STATE_FIELD: [0.0, math.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}, "must be finite"),
        ({"unrelated": [1.0]}, "must provide"),
        ([0.0] * 8, "must be an object"),
    ],
)
def test_state_rejects_observations_outside_the_contract(values: object, message: str) -> None:
    with pytest.raises(LiberoPi05MapperError, match=message):
        _map(_observation(values))


def test_adapter_jpeg_frames_become_named_rgb_png_payloads() -> None:
    request = _map()

    assert [image.name for image in request.images] == ["image", "wrist_image"]
    assert [image.mime_type for image in request.images] == ["image/png", "image/png"]
    for image in request.images:
        assert isinstance(image, ImagePayload)
        with Image.open(io.BytesIO(image.data)) as decoded:
            assert decoded.format == "PNG"
            assert decoded.mode == "RGB"
            assert decoded.size == IMAGE_SIZE


def test_png_frames_are_normalized_to_rgb_png() -> None:
    frames = (
        CameraFrame("image", "image/png", _encoded_frame("image", mode="L", color=128).data),
        CameraFrame("wrist_image", "image/png", _encoded_frame("wrist_image", mode="L", color=64).data),
    )
    request = LiberoPi05Mapper().map_observation(
        _observation(),
        session_id="session-1",
        request_id="request-1",
        step_id=0,
        instruction="pick up the black bowl",
        frames=frames,
    )

    assert [image.name for image in request.images] == ["image", "wrist_image"]
    for image in request.images:
        assert image.mime_type == "image/png"
        with Image.open(io.BytesIO(image.data)) as decoded:
            assert decoded.format == "PNG"
            assert decoded.mode == "RGB"
            assert decoded.size == IMAGE_SIZE


def test_missing_or_ambiguous_camera_frames_are_rejected() -> None:
    with pytest.raises(LiberoPi05MapperError, match="missing camera frames: wrist_image"):
        LiberoPi05Mapper().map_observation(
            _observation(),
            session_id="session-1",
            request_id="request-1",
            step_id=0,
            instruction="pick up the black bowl",
            frames=(_adapter_frames()[0],),
        )

    duplicated = (*_adapter_frames(), _encoded_frame("image"))
    with pytest.raises(LiberoPi05MapperError, match="ambiguous camera frames"):
        LiberoPi05Mapper().map_observation(
            _observation(),
            session_id="session-1",
            request_id="request-1",
            step_id=0,
            instruction="pick up the black bowl",
            frames=duplicated,
        )


def test_items_that_are_not_camera_frames_are_rejected() -> None:
    with pytest.raises(LiberoPi05MapperError, match="CameraFrame values"):
        LiberoPi05Mapper().map_observation(
            _observation(),
            session_id="session-1",
            request_id="request-1",
            step_id=0,
            instruction="pick up the black bowl",
            frames=(object(),),  # type: ignore[arg-type]
        )


def test_missing_pillow_is_reported_instead_of_sending_raw_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    # Encode the synthetic frames first so only the mapper needs Pillow.
    frames = _adapter_frames()
    monkeypatch.setitem(sys.modules, "PIL", None)
    monkeypatch.setitem(sys.modules, "PIL.Image", None)

    with pytest.raises(LiberoPi05MapperError, match="requires Pillow"):
        LiberoPi05Mapper().map_observation(
            _observation(),
            session_id="session-1",
            request_id="request-1",
            step_id=0,
            instruction="pick up the black bowl",
            frames=frames,
        )


def test_undecodable_frame_bytes_are_rejected() -> None:
    frames = (
        CameraFrame("image", "image/png", b"not an encoded image"),
        _encoded_frame("wrist_image"),
    )
    with pytest.raises(LiberoPi05MapperError, match="not a decodable image"):
        LiberoPi05Mapper().map_observation(
            _observation(),
            session_id="session-1",
            request_id="request-1",
            step_id=0,
            instruction="pick up the black bowl",
            frames=frames,
        )


def test_actions_preserve_chunk_order_width_and_identity() -> None:
    actions = LiberoPi05Mapper().map_result(_result())

    assert len(actions) == 3
    assert all(isinstance(action, RobotAction) for action in actions)
    assert [action.values["action"] for action in actions] == [
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0],
        [2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0],
    ]
    assert [action.metadata["chunk_index"] for action in actions] == [0, 1, 2]
    assert all(action.metadata["chunk_size"] == 3 for action in actions)
    assert all(action.metadata["action_space"] == POLICY_ACTION_SPACE for action in actions)
    assert all(action.metadata["action_feature_names"] == list(ACTION_FEATURE_NAMES) for action in actions)
    assert all(action.metadata["request_id"] == "request-1" for action in actions)
    assert all(action.metadata["session_id"] == "session-1" for action in actions)
    assert all(math.isfinite(value) for action in actions for value in action.values["action"])


@pytest.mark.parametrize("length", [1, 7, 50])
def test_action_chunk_length_is_not_hardcoded(length: int) -> None:
    actions = LiberoPi05Mapper().map_result(_result(rows=[[0.0] * 7] * length))

    assert len(actions) == length
    assert all(action.metadata["chunk_size"] == length for action in actions)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ([0.0] * 6, "must contain 7 values"),
        ([0.0] * 8, "must contain 7 values"),
        ([0.0] * 6 + [math.nan], "must be finite"),
        ([0.0] * 6 + [math.inf], "must be finite"),
        ([0.0] * 6 + ["x"], "must be numeric"),
        ([0.0] * 6 + [None], "must be numeric"),
        ([0.0] * 6 + [True], "must be numeric"),
        ("not-a-row", "must be a sequence"),
    ],
)
def test_malformed_action_rows_are_rejected(row: object, message: str) -> None:
    with pytest.raises(LiberoPi05MapperError, match=message):
        LiberoPi05Mapper().map_result(_result(rows=[row]))


def test_declared_feature_names_must_match_the_wire_order() -> None:
    assert len(LiberoPi05Mapper().map_result(_result())) == 3
    # An absent declaration is tolerated so backends that omit it still work.
    assert len(LiberoPi05Mapper().map_result(_result(feature_names=None))) == 3

    with pytest.raises(LiberoPi05MapperError, match="feature_names"):
        LiberoPi05Mapper().map_result(_result(feature_names=tuple(reversed(ACTION_FEATURE_NAMES))))
    with pytest.raises(LiberoPi05MapperError, match="feature_names"):
        LiberoPi05Mapper().map_result(_result(feature_names=["dx", "dy"]))
    with pytest.raises(LiberoPi05MapperError, match="feature_names"):
        LiberoPi05Mapper().map_result(_result(feature_names=(name for name in ACTION_FEATURE_NAMES)))


def test_action_space_and_chunk_shape_are_enforced() -> None:
    with pytest.raises(LiberoPi05MapperError, match="action_space mismatch"):
        LiberoPi05Mapper().map_result(_result(action_space="pi05.state.v1"))
    with pytest.raises(LiberoPi05MapperError, match="unsupported action kind"):
        LiberoPi05Mapper().map_result(_result(kind="waypoints"))
    with pytest.raises(LiberoPi05MapperError, match="must not be empty"):
        LiberoPi05Mapper().map_result(_result(rows=[]))
    with pytest.raises(LiberoPi05MapperError, match="exactly one policy action"):
        LiberoPi05Mapper().map_result(
            _result(
                actions=(
                    PolicyAction("action_chunk", {"data": [[0.0] * 7]}),
                    PolicyAction("action_chunk", {"data": [[0.0] * 7]}),
                )
            )
        )


def test_clipping_is_opt_in_and_clamps_to_the_libero_range() -> None:
    row = [0.5, -3.0, 2.0, 0.0, -1.5, 0.25, -0.75]

    unclipped = LiberoPi05Mapper().map_result(_result(rows=[row]))[0]
    assert unclipped.values["action"] == row
    assert unclipped.metadata["clipped"] is False

    clipped = LiberoPi05Mapper(clip_action=True).map_result(_result(rows=[row]))[0]
    assert clipped.values["action"] == [0.5, -1.0, 1.0, 0.0, -1.0, 0.25, -0.75]
    assert clipped.metadata["clipped"] is True

    with pytest.raises(LiberoPi05MapperError, match="clip_action must be a boolean"):
        LiberoPi05Mapper(clip_action=1)  # type: ignore[arg-type]


def test_map_observation_propagates_identity_instruction_and_timestamp() -> None:
    request = LiberoPi05Mapper().map_observation(
        _observation(timestamp_s=12.5),
        session_id="session-1",
        request_id="request-1",
        step_id=4,
        instruction="pick up the black bowl",
        frames=_adapter_frames(),
    )

    assert request.session_id == "session-1"
    assert request.request_id == "request-1"
    assert request.step_id == 4
    assert request.instruction == "pick up the black bowl"
    assert request.metadata["robot_timestamp_s"] == 12.5
    assert request.reset is False


def test_mapper_observation_round_trips_through_the_policy_contract() -> None:
    request = _map()

    assert isinstance(request, PolicyObservation)
    rebuilt = PolicyObservation(
        session_id=request.session_id,
        request_id=request.request_id,
        step_id=request.step_id,
        instruction=request.instruction,
        state=request.state,
        images=request.images,
        reset=request.reset,
        metadata=request.metadata,
    )

    assert rebuilt == request
    assert list(rebuilt.state) == [STATE_FIELD]
    assert len(rebuilt.state[STATE_FIELD]) == 8
    assert all(math.isfinite(value) for value in rebuilt.state[STATE_FIELD])
    assert all(isinstance(image, ImagePayload) for image in rebuilt.images)
    assert all(image.name in {"image", "wrist_image"} for image in rebuilt.images)
