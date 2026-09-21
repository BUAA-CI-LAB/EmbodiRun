"""Validation of decoded R2R actions at the execution boundary."""

ACTION_SPACE = "activevln.r2r.discrete.v1"
IMAGE_FIELD = "observation.images.rgb"


def decode_actions(rows) -> list[tuple[int, int]]:
    """Validate the framework's [3,2] grammar; PAD never becomes STOP."""
    import numpy as np

    array = np.asarray(rows, dtype=float)
    if array.shape != (3, 2) or not np.isfinite(array).all():
        raise ValueError(f"Expected finite action chunk [3,2], received {array}")
    actions = []
    for raw_id, value in array:
        if raw_id != int(raw_id):
            raise ValueError(f"Non-integer action id: {raw_id}")
        action_id = int(raw_id)
        if action_id == -1:
            continue
        if action_id == 0:
            actions.append((0, 0))
            break
        valid = (25, 50, 75) if action_id == 1 else (15, 30, 45)
        if action_id not in (1, 2, 3) or value not in valid:
            raise ValueError(f"Action outside the pinned R2R grammar: {(action_id, value)}")
        actions.append((action_id, int(value)))
    if not actions:
        raise ValueError("Model returned no usable actions")
    return actions


def action_text(action: tuple[int, int]) -> str:
    kind, value = action
    if kind == 0:
        return "stop"
    if kind == 1:
        return f"forward {value}cm"
    return f"turn {'left' if kind == 2 else 'right'} {value}deg"
