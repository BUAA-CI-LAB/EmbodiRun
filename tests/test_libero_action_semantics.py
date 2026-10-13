from __future__ import annotations

import numpy as np
import pytest

from embodirun.simulators.libero.adapter import _action_array, _encode_frame
from embodirun.simulators.libero.config import LiberoConfig


@pytest.mark.parametrize(
    ("gripper", "expected"),
    [
        (-0.001, -1.0),
        (0.0, 0.0),
        (0.001, 1.0),
        (-4.0, -1.0),
        (4.0, 1.0),
    ],
)
def test_libero_action_binarizes_gripper_like_official_eval(
    gripper: float,
    expected: float,
) -> None:
    source = np.asarray([2.0, -2.0, 0.25, -0.25, 0.5, -0.5, gripper], dtype=np.float32)

    action = _action_array({"action": source})

    np.testing.assert_array_equal(
        action,
        np.asarray([1.0, -1.0, 0.25, -0.25, 0.5, -0.5, expected], dtype=np.float32),
    )
    # Formatting an action must not mutate the policy result kept for RTC
    # prefix bookkeeping or diagnostics.
    assert source[-1] == pytest.approx(gripper)


def test_libero_png_transport_is_lossless() -> None:
    cv2 = pytest.importorskip("cv2")
    image = np.asarray(
        [
            [[0, 17, 255], [64, 91, 128]],
            [[33, 127, 231], [255, 3, 79]],
        ],
        dtype=np.uint8,
    )

    frame = _encode_frame("observation.images.image", image, encoding="png")
    decoded_bgr = cv2.imdecode(np.frombuffer(frame.data, dtype=np.uint8), cv2.IMREAD_COLOR)

    assert frame.mime_type == "image/png"
    np.testing.assert_array_equal(decoded_bgr[..., ::-1], image)


def test_libero_config_rejects_unknown_image_encoding() -> None:
    with pytest.raises(ValueError, match="image_encoding"):
        LiberoConfig("libero", "libero_spatial", 0, image_encoding="webp")
