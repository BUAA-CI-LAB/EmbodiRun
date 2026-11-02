"""Compatibility imports; deployment now lives in embodirun.deployment.go2."""

from embodirun.deployment.go2.transport import (  # noqa: F401
    CommandResult,
    ParamikoTransport,
    RemoteCommandError,
    RemoteTransport,
    SshConnection,
)
