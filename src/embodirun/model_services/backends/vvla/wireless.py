"""VVLA inference service integration over WirelessComm RPC."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...contracts import PolicyObservation, PolicyResult, Session
from ...protocols.wireless import (
    WirelessProtocolError,
    WirelessRpcTransport,
    WirelessTransport,
)

RPC_SCHEMA = "vvla.policy.rpc.v1"
REQUEST_TAG = 0x56564C41
RESPONSE_TAG = 0x56564C42


class VvlaWirelessError(RuntimeError):
    """A VVLA WirelessComm request failed locally or remotely."""

    def __init__(self, message: str, *, status: int | None = None, code: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


class VvlaWirelessClient:
    """Session-oriented VVLA client over the WirelessComm protocol."""

    def __init__(
        self,
        transport: WirelessTransport,
        *,
        timeout_s: float = 5.0,
        owns_transport: bool = True,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.transport = transport
        self.timeout_s = float(timeout_s)
        self.owns_transport = owns_transport

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        *,
        server_node_id: str,
        token: str | None = None,
        timeout_s: float = 5.0,
    ) -> VvlaWirelessClient:
        """Create a client backed by a process-owned WirelessComm runtime."""

        transport = WirelessRpcTransport.from_config(
            config_path,
            server_node_id=server_node_id,
            rpc_schema=RPC_SCHEMA,
            request_tag=REQUEST_TAG,
            response_tag=RESPONSE_TAG,
            token=token,
        )
        return cls(transport, timeout_s=timeout_s)

    def health(self) -> dict[str, Any]:
        return self._request("health", None)

    def capabilities(self) -> dict[str, Any]:
        return self._request("capabilities", None)

    def open_session(
        self,
        *,
        robot_id: str,
        action_space: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Session:
        result = self._request(
            "open_session",
            {
                "schema": "vvla.policy.session.v1",
                "robot_id": robot_id,
                "action_space": action_space,
                "metadata": dict(metadata or {}),
            },
        )
        return Session(
            session_id=str(result["session_id"]),
            revision=int(result.get("session_revision", 0)),
        )

    def step(self, observation: PolicyObservation) -> PolicyResult:
        result = PolicyResult.from_payload(
            self._request(
                "step",
                {
                    "schema": "vvla.policy.step.v1",
                    "session_id": observation.session_id,
                    "request_id": observation.request_id,
                    "step_id": observation.step_id,
                    "instruction": observation.instruction,
                    "state": dict(observation.state),
                    "reset": observation.reset,
                    "metadata": dict(observation.metadata),
                    "images": [
                        {
                            "name": image.name,
                            "mime_type": image.mime_type,
                            "data": image.data,
                        }
                        for image in observation.images
                    ],
                },
            )
        )
        if result.request_id != observation.request_id:
            raise VvlaWirelessError("step response request_id does not match the request")
        if result.session_id != observation.session_id:
            raise VvlaWirelessError("step response session_id does not match the request")
        if result.step_id != observation.step_id:
            raise VvlaWirelessError("step response step_id does not match the request")
        return result

    def reset(self, session_id: str, *, request_id: str) -> Session:
        result = self._request(
            "reset",
            {"session_id": session_id, "body": {"request_id": request_id}},
        )
        return Session(
            session_id=str(result["session_id"]),
            revision=int(result["session_revision"]),
        )

    def close(self, session_id: str) -> None:
        self._request("close", {"session_id": session_id})

    def shutdown(self) -> None:
        """Release the process-owned WirelessComm runtime."""

        if self.owns_transport:
            self.transport.shutdown()

    def with_timeout(self, timeout_s: float) -> VvlaWirelessClient:
        """Share this connection through a client with a request-local timeout."""

        return VvlaWirelessClient(
            self.transport,
            timeout_s=timeout_s,
            owns_transport=False,
        )

    def _request(
        self,
        method: str,
        payload: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        try:
            return self.transport.request(method, payload, timeout_s=self.timeout_s)
        except WirelessProtocolError as error:
            raise VvlaWirelessError(
                str(error),
                status=error.status,
                code=error.code,
            ) from error


def vvla_wireless_server_command(
    *,
    policy: str,
    checkpoint: str,
    comm_config: str,
    device: str | None = None,
    adapter_config: str | None = None,
) -> tuple[str, ...]:
    """Build the documented ``vvla-wireless-serve`` command-line contract."""

    argv = [
        "vvla-wireless-serve",
        "--policy",
        policy,
        "--checkpoint",
        checkpoint,
    ]
    if adapter_config is not None:
        argv.extend(("--adapter-config", adapter_config))
    if device is not None:
        argv.extend(("--device", device))
    argv.extend(("--comm-config", comm_config))
    return tuple(argv)


__all__ = [
    "VvlaWirelessClient",
    "VvlaWirelessError",
    "vvla_wireless_server_command",
]
