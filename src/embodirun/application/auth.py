"""Small application-layer authorization boundary for Control routes.

The default deployment is reached through the already authenticated SSH and
loopback path, so an empty token map keeps that trusted mode.  Supplying any
token-to-principal map switches every application route to explicit token
authentication.  Each configured token binds a server-defined ``caller_id``
and may bind a ``session_id``; request payloads cannot choose or elevate a
role or ownership scope.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum


class AuthError(PermissionError):
    """Base error for missing, invalid, or insufficient credentials."""


class AuthenticationError(AuthError):
    """The configured token is absent or does not match."""


class AuthorizationError(AuthError):
    """The authenticated principal lacks the required route role."""


class Role(str, Enum):
    OBSERVER = "observer"
    CONTROLLER = "controller"
    ADMIN = "admin"


@dataclass(frozen=True, slots=True)
class Principal:
    """Detached route identity; no raw token is retained in the result."""

    role: Role
    trusted: bool = False
    caller_id: str | None = None
    session_id: str | None = None


_ROLE_RANK = {
    Role.OBSERVER: 0,
    Role.CONTROLLER: 1,
    Role.ADMIN: 2,
}


class AuthPolicy:
    """Authorize operations using configured roles and ownership scopes."""

    def __init__(
        self,
        token_roles: Mapping[str, str | Role | Principal | Mapping[str, object]] | None = None,
    ) -> None:
        values = {} if token_roles is None else dict(token_roles)
        principals: dict[str, Principal] = {}
        for token, configured in values.items():
            if not isinstance(token, str) or not token:
                raise ValueError("auth token keys must be non-empty strings")
            try:
                if isinstance(configured, Principal):
                    principal = configured
                elif isinstance(configured, Mapping):
                    role = configured.get("role")
                    caller_id = configured.get("caller_id")
                    session_id = configured.get("session_id")
                    if not isinstance(role, (str, Role)):
                        raise ValueError("principal role is required")
                    principal = Principal(
                        role if isinstance(role, Role) else Role(role),
                        caller_id=_identity_value(caller_id, "caller_id"),
                        session_id=_identity_value(session_id, "session_id"),
                    )
                else:
                    principal = Principal(configured if isinstance(configured, Role) else Role(configured))
            except (TypeError, ValueError) as error:
                # Do not include the configured secret in a startup or HTTP
                # error.  The caller can identify the configuration entry.
                raise ValueError("unsupported auth principal configuration") from error
            if principal.caller_id is None:
                raise ValueError(
                    "token principals must bind a caller_id; use trusted mode when caller binding is unavailable"
                )
            principals[token] = principal
        self._token_principals = principals

    @property
    def token_authentication_enabled(self) -> bool:
        return bool(self._token_principals)

    def authorize(self, token: str | None, required: Role) -> Principal:
        """Return a principal or fail closed when token authentication is on."""

        if not isinstance(required, Role):
            raise TypeError("required role must be a Role")
        if not self._token_principals:
            return Principal(Role.ADMIN, trusted=True)
        if not isinstance(token, str) or token not in self._token_principals:
            raise AuthenticationError("a configured control token is required")
        principal = self._token_principals[token]
        if _ROLE_RANK[principal.role] < _ROLE_RANK[required]:
            raise AuthorizationError(f"role {principal.role.value!r} cannot perform {required.value!r} operation")
        return principal

    def authorize_scope(
        self,
        token: str | None,
        required: Role,
        *,
        caller_id: str,
        session_id: str,
    ) -> Principal:
        """Authorize a route and bind its ownership scope to the token."""

        principal = self.authorize(token, required)
        if principal.trusted:
            return principal
        if principal.caller_id != caller_id:
            raise AuthorizationError("token principal is not authorized for caller_id")
        if principal.session_id is not None and principal.session_id != session_id:
            raise AuthorizationError("token principal is not authorized for session_id")
        return principal


def _identity_value(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"principal {name} must be a non-empty string")
    return value


__all__ = [
    "AuthError",
    "AuthPolicy",
    "AuthenticationError",
    "AuthorizationError",
    "Principal",
    "Role",
]
