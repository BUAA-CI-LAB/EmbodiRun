"""Resolve service addresses without confusing SSH access with data traffic."""

from __future__ import annotations

from ipaddress import ip_address

from .config import ConfigError, NodeConfig


def service_host(node: NodeConfig, bind: str, *, remote: bool) -> str:
    """Return an advertised host reachable by the configured consumers."""

    if bind in {"0.0.0.0", "::"}:
        if not remote:
            return "::1" if bind == "::" else "127.0.0.1"
        host = node.address or node.connection.host
        if host is None:
            raise ConfigError(
                f"node {node.node_id!r} has no advertised address; set nodes."
                f"{node.node_id}.address for cross-node communication"
            )
    else:
        host = bind
    if any(char.isspace() for char in host) or any(char in host for char in "/@[]"):
        raise ConfigError(f"node {node.node_id!r} requires a host without URL or port")
    try:
        address = ip_address(host)
    except ValueError:
        if ":" in host:
            raise ConfigError(f"node {node.node_id!r} requires a host without port") from None
        loopback = host.rstrip(".").lower() == "localhost"
        unspecified = False
    else:
        loopback = address.is_loopback
        unspecified = address.is_unspecified
    if unspecified or (remote and loopback):
        raise ConfigError(
            f"node {node.node_id!r} cannot advertise {host!r} across nodes; "
            "use a non-loopback bind/address (nodes.<id>.address)"
        )
    return host
