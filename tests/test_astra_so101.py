from __future__ import annotations

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from agents.astra_pi05 import (
    BI_SO101_POSITION_FEATURES,
    BiSO101ActionEncoder,
    SO101ReviewPacketBuilder,
    extract_bi_so101_state,
)
from agents.astra_pi05.cooperative import Proposal
from agents.rpent.session import PublicCooperativeSession, normalize_public_proposal
from agents.rpent.so101_correction import SO101CorrectionError, SO101PlanarCorrectionMapper

from embodirun.application.api import ControlApplication
from embodirun.client import ControlClient, Observation
from embodirun.devices.execution.arbitration import (
    RobotAdapterCommandPort,
    RobotControlArbiter,
)
from embodirun.devices.observations.store import ObservationStore
from embodirun.devices.observations.values import ObservationSnapshot
from embodirun.robots.lerobot.bi_so101 import BiSO101Adapter, BiSO101Config
from embodirun.robots.lerobot.so101 import SO101_POSITION_FEATURES, SO101Adapter
from embodirun.services.control.http_api import ControlHTTPAPI


def _proposal(observation_id: str) -> Proposal:
    return Proposal.from_payload(
        {
            "proposal_id": "proposal-http",
            "observation_id": observation_id,
            "actions": [[0.0] * 12 for _ in range(50)],
            "metadata": {
                "action_semantics": "biso101_so101_v1",
                "action_layout": "left6_right6",
                "action_encoding": "absolute",
                "joint_position_unit": "degrees",
                "action_dim": 12,
                "horizon": 50,
                "feature_names": list(BI_SO101_POSITION_FEATURES),
            },
        },
        observation_id=observation_id,
    )


class _MediaHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args: object) -> None:
        return None

    def do_GET(self) -> None:  # noqa: N802 - stdlib HTTP hook
        query = parse_qs(urlsplit(self.path).query)
        role = query.get("frame", [""])[0]
        self.server.paths.append(self.path)  # type: ignore[attr-defined]
        payload = {
            "observation_id": "obs-http",
            "media": [
                {
                    "name": role,
                    "mime_type": "image/png",
                    "bytes": len(role),
                    "media_ref": f"observation://obs-http/{role}",
                    "data_base64": base64.b64encode(role.encode()).decode("ascii"),
                }
            ],
        }
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def test_packet_builder_decodes_the_real_public_media_envelope(tmp_path: Path) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MediaHandler)
    server.paths = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = ControlClient(
            f"http://127.0.0.1:{server.server_port}",
            caller_id="agent",
            session_id="session",
        )
        state = {name: float(index) for index, name in enumerate(BI_SO101_POSITION_FEATURES)}
        observation = Observation(
            "obs-http",
            {
                "observation_id": "obs-http",
                "robot": {"timestamp_s": 1.0, "values": state},
            },
        )
        assert extract_bi_so101_state(observation) == [float(index) for index in range(12)]
        builder = SO101ReviewPacketBuilder(
            client,
            media_dir=tmp_path,
            state_extractor=lambda value: [value.robot[name] for name in BI_SO101_POSITION_FEATURES],
            trajectory_preview=lambda rows: {
                "frame": "so101_shoulder_plane",
                "left": [[0.0, 0.0] for _ in rows],
                "right": [[0.0, 0.0] for _ in rows],
            },
        )
        packet = builder(observation, _proposal("obs-http"), {})
        images = packet["observation"]["images"]
        assert set(images) == {"front", "left_wrist", "right_wrist"}
        for role, filename in images.items():
            assert Path(filename).read_bytes() == role.encode()
        assert all("include_data=true" in path for path in server.paths)  # type: ignore[attr-defined]
    finally:
        server.shutdown()
        server.server_close()


def _calibration() -> dict[str, dict[str, object]]:
    return {name: {"range_min": 0, "range_max": 4095, "drive_mode": 0} for name in BI_SO101_POSITION_FEATURES}


def test_so101_mapper_requires_public_step_and_preserves_other_arm() -> None:
    mapper = SO101PlanarCorrectionMapper(
        BI_SO101_POSITION_FEATURES,
        calibration=_calibration(),
        control_hz=5.0,
        action_encoder=BiSO101ActionEncoder(),
    )
    state = {name: float(index) for index, name in enumerate(BI_SO101_POSITION_FEATURES)}
    state.update(
        {
            BI_SO101_POSITION_FEATURES[5]: 50.0,
            BI_SO101_POSITION_FEATURES[11]: 60.0,
        }
    )
    observation = Observation("obs-state", {"robot": {"values": state}})
    left_target = mapper.preview([[0.0] * 12])["left"][0]
    correction = {
        "left": {
            "reach_m": left_target[0],
            "height_m": left_target[1],
            "pan_deg": 0.0,
            "wrist_flex_deg": 0.0,
            "wrist_roll_deg": 0.0,
            "gripper": 50.0,
        },
        "right": None,
        "frame": "so101_shoulder_plane",
        "units": "m_deg",
        "duration_s": 0.2,
    }
    actions = mapper([correction], observation=observation)
    assert actions[0]["values"]["type"] == "joint_position"
    assert actions[0]["values"]["right"]["joint_positions_deg"] == [
        state[name] for name in BI_SO101_POSITION_FEATURES[6:11]
    ]
    assert actions[0]["metadata"]["control_hz"] == 5.0
    correction["duration_s"] = 0.3
    with pytest.raises(SO101CorrectionError, match="control_hz"):
        mapper([correction], observation=observation)


def test_structured_public_proposal_is_decoded_only_with_explicit_encoder() -> None:
    encoder = BiSO101ActionEncoder()
    target = encoder(
        [1, 2, 3, 4, 5, 55, 6, 7, 8, 9, 10, 65],
        timestamp_s=0.0,
        metadata={"joint_position_unit": "degrees"},
    )
    payload = {
        "proposal_id": "structured-proposal",
        "observation_id": "obs-structured",
        "actions": [{"timestamp_s": float(index), "values": target["values"]} for index in range(50)],
        "metadata": {
            "action_semantics": "biso101_so101_v1",
            "action_layout": "left6_right6",
            "action_encoding": "absolute",
            "joint_position_unit": "degrees",
            "action_dim": 12,
            "horizon": 50,
            "feature_names": list(BI_SO101_POSITION_FEATURES),
        },
    }
    normalized = normalize_public_proposal(
        payload,
        observation_id="obs-structured",
        feature_names=BI_SO101_POSITION_FEATURES,
        metadata=encoder.proposal_metadata,
        action_decoder=encoder.decode,
    )
    assert normalized["actions"][0] == [1.0, 2.0, 3.0, 4.0, 5.0, 55.0, 6.0, 7.0, 8.0, 9.0, 10.0, 65.0]


def test_session_reencodes_structured_proposal_for_public_biso101_execute() -> None:
    client = ControlClient("http://127.0.0.1:1", caller_id="agent", session_id="session")
    observations = iter(
        [
            Observation("obs-1", {"observation_id": "obs-1", "sequence": 1}),
            Observation("obs-2", {"observation_id": "obs-2", "sequence": 2}),
            Observation("obs-3", {"observation_id": "obs-3", "sequence": 3}),
        ]
    )
    client.observe = lambda **_kwargs: next(observations)  # type: ignore[method-assign]
    encoder = BiSO101ActionEncoder()
    action = encoder(
        [1, 2, 3, 4, 5, 55, 6, 7, 8, 9, 10, 65],
        timestamp_s=0.0,
        metadata={"joint_position_unit": "degrees"},
    )
    client.propose = lambda **_kwargs: {  # type: ignore[method-assign]
        "status": "proposed",
        "proposal_id": "structured-session",
        "observation_id": "obs-1",
        "actions": [{"timestamp_s": float(index), "values": action["values"]} for index in range(50)],
    }
    executed: list[list[dict[str, object]]] = []

    def execute(actions, **_kwargs):
        executed.append(actions)
        return {"status": "completed", "result": {"executed_steps": len(actions)}}

    client.execute = execute  # type: ignore[method-assign]
    session = PublicCooperativeSession(
        client,
        reviewer=lambda packet: {
            "proposal_id": packet["proposal"]["proposal_id"],
            "observation_id": packet["observation"]["observation_id"],
            "decision": "execute_prefix",
            "execute_steps": 1,
            "corrections": [],
            "reason": "structured fake review",
        },
        instruction="move",
        feature_names=BI_SO101_POSITION_FEATURES,
        action_encoder=encoder,
        control_hz=5.0,
        post_observation_timeout_s=0.01,
    )
    result = session.run()
    assert result["status"] == "completed"
    assert executed[0][0]["values"]["left"]["joint_positions_deg"] == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ]


class _Bus:
    def __init__(self, positions: list[float]) -> None:
        self.positions = dict(zip(SO101_POSITION_FEATURES, positions))
        self.is_connected = False
        self.is_calibrated = True
        self.actions: list[dict[str, float]] = []

    def connect(self, *, calibrate: bool, prepare: bool = True) -> None:
        assert calibrate is False
        self.is_connected = True

    def get_observation(self) -> dict[str, float]:
        return dict(self.positions)

    def send_action(self, action: dict[str, float]) -> None:
        self.actions.append(dict(action))
        self.positions.update(action)

    def disconnect(self) -> None:
        self.is_connected = False


class _AdapterService:
    def __init__(self, arbiter: RobotControlArbiter, store: ObservationStore) -> None:
        self.arbiter = arbiter
        self.store = store

    def describe(self) -> dict[str, object]:
        return {"status": "ok", "robot_id": "pair", "capabilities": {}}

    def observe(self, *, runtime_id: str | None, include_robot: bool) -> dict[str, object]:
        value = self.arbiter.observe()
        now = time.monotonic_ns()
        observation_id, generation, sequence = self.store.next_observation_id()
        self.store.publish(
            ObservationSnapshot(
                observation_id=observation_id,
                service_instance_id=self.store.service_instance_id,
                generation=generation,
                sequence=sequence,
                state=dict(value.values),
                metadata={
                    "state_metadata": {"timestamp_s": value.timestamp_s},
                    "clock_domain": "host_monotonic_ns",
                },
                captured_timestamp_ns=now,
                received_timestamp_ns=now,
                published_timestamp_ns=now,
                source_timestamps_ns={"state": now},
                source_received_timestamps_ns={"state": now},
                clock_domains={"state": "host_monotonic_ns"},
                skew_ns=0,
            )
        )
        return {
            "status": "ok",
            "observation_id": observation_id,
            "sequence": sequence,
            "robot": {
                "timestamp_s": value.timestamp_s,
                "values": dict(value.values),
                "metadata": {},
            },
        }


class _ControlHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args: object) -> None:
        return None

    def _call(self, method: str) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length)) if length else None
        response = self.server.api.dispatch(  # type: ignore[attr-defined]
            method,
            self.path,
            body=body,
            headers=dict(self.headers.items()),
        )
        encoded = json.dumps(response.payload).encode("utf-8")
        self.send_response(response.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802 - stdlib HTTP hook
        self._call("GET")

    def do_POST(self) -> None:  # noqa: N802 - stdlib HTTP hook
        self._call("POST")


def test_public_http_execute_reaches_real_biso101_adapter_with_fake_buses() -> None:
    left_bus = _Bus([0, 1, 2, 3, 4, 50])
    right_bus = _Bus([5, 6, 7, 8, 9, 60])
    config = BiSO101Config.from_mapping(
        "pair",
        {
            "left_port": "/dev/fake-left",
            "right_port": "/dev/fake-right",
            "max_joint_step_deg": 20.0,
            "max_gripper_step": 30.0,
        },
    )
    adapter = BiSO101Adapter(
        config,
        left=SO101Adapter(config.left, controller=left_bus),
        right=SO101Adapter(config.right, controller=right_bus),
    )
    adapter.connect()
    port = RobotAdapterCommandPort(adapter)
    arbiter = RobotControlArbiter(port)
    store = ObservationStore(service_instance_id="biso101-http")
    service = _AdapterService(arbiter, store)
    application = ControlApplication(
        service,
        observation_store=store,
        arbiter_provider=lambda: arbiter,
        default_control_hz=5.0,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ControlHandler)
    server.api = ControlHTTPAPI(application)  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = ControlClient(
            f"http://127.0.0.1:{server.server_port}",
            caller_id="agent",
            session_id="session",
        )
        before = client.observe()
        encoder = BiSO101ActionEncoder()
        row = [1, 2, 3, 4, 5, 55, 6, 7, 8, 9, 10, 65]
        action = encoder(
            row,
            timestamp_s=0.0,
            metadata={"joint_position_unit": "degrees"},
        )
        result = client.execute(
            [action],
            request_id="biso101-http-1",
            observation_id=before.observation_id,
            control_hz=5.0,
            wait=True,
        )
        assert result["status"] == "completed"
        assert result["result"]["executed_steps"] == 1
        assert left_bus.actions[-1] == {
            "shoulder_pan.pos": 1.0,
            "shoulder_lift.pos": 2.0,
            "elbow_flex.pos": 3.0,
            "wrist_flex.pos": 4.0,
            "wrist_roll.pos": 5.0,
            "gripper.pos": 55.0,
        }
        assert right_bus.actions[-1]["gripper.pos"] == 65.0
    finally:
        server.shutdown()
        server.server_close()
        application.close()
        arbiter.close(hold=False)
        adapter.close()
