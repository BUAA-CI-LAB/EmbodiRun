"""Keyboard-only differential drive input; never fabricates XR tracking."""

from dataclasses import dataclass

from .control import MappingConfig, finite


@dataclass(frozen=True)
class KeyboardFrame:
    seq: int
    timestamp_ms: float
    keys: frozenset[str]
    focused: bool
    video_ready: bool

    @classmethod
    def parse(cls, data: dict) -> "KeyboardFrame":
        if data.get("type") != "keyboard_input":
            raise ValueError("keyboard_input required")
        seq = data.get("seq")
        stamp = finite(data.get("timestamp_ms"), "timestamp_ms")
        keys = data.get("keys")
        if type(seq) is not int or seq < 0 or stamp < 0:
            raise ValueError("invalid keyboard sequence or timestamp")
        if (
            not isinstance(keys, list)
            or len(keys) > 4
            or any(k not in ("KeyW", "KeyA", "KeyS", "KeyD") for k in keys)
            or len(set(keys)) != len(keys)
        ):
            raise ValueError("keys must be unique WASD codes")
        if type(data.get("focused")) is not bool or type(data.get("video_ready")) is not bool:
            raise ValueError("keyboard focus and video_ready must be booleans")
        return cls(seq, stamp, frozenset(keys), data["focused"], data["video_ready"])

    @property
    def ready(self) -> bool:
        return self.focused and self.video_ready

    @property
    def neutral(self) -> bool:
        return self.ready and not self.keys

    def action(
        self, config: MappingConfig, *, linear_m_s: float | None = None, angular_deg_s: float | None = None
    ) -> dict[str, float]:
        if not config.enable_base or not self.ready:
            raise ValueError("keyboard drive requires base, focus and live video")
        linear = config.max_linear_m_s if linear_m_s is None else finite(linear_m_s)
        angular = config.max_angular_deg_s if angular_deg_s is None else finite(angular_deg_s)
        if not 0 < linear <= config.max_linear_m_s or not 0 < angular <= config.max_angular_deg_s:
            raise ValueError("keyboard speed outside configured limits")
        return {
            "x.vel": (("KeyW" in self.keys) - ("KeyS" in self.keys)) * linear,
            "theta.vel": (("KeyA" in self.keys) - ("KeyD" in self.keys)) * angular,
        }
