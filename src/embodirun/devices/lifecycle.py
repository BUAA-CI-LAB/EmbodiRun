"""Shared physical-device ownership and lifecycle primitives.

The control service owns *leases*, while this module owns the underlying
connection.  A lease is intentionally small: readers and runtimes can keep a
reference without receiving a driver object that they could close behind
another consumer's back.  The process lock is acquired only while a resource
is opened and held for the lifetime of the owner, so a second control process
gets a useful ``busy`` error instead of racing a Python lock.

This is a local-node facility.  An external robot agent can be declared as the
owner and supplied with an attach function; Deploy then keeps a proxy/transport
reference and never opens the agent's SDK or device itself.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from embodirun.robots.adapter import RobotPreparationRefused

try:  # pragma: no cover - the supported deployment platforms are POSIX.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


class DeviceError(RuntimeError):
    """Base class for resource ownership and lifecycle failures."""


class DeviceBusyError(DeviceError):
    """A different process already owns a physical resource."""

    def __init__(self, identity: ResourceIdentity, holder: Mapping[str, Any] | None):
        self.identity = identity
        self.holder = dict(holder or {})
        owner = self.holder.get("owner_id", "unknown")
        pid = self.holder.get("pid", "unknown")
        state = self.holder.get("state")
        state_detail = f", state={state!r}" if state is not None else ""
        super().__init__(f"resource {identity.key!r} is busy (owner={owner!r}, pid={pid!r}{state_detail})")


class DeviceOpenError(DeviceError):
    """A resource could not be opened and was rolled back."""


class DeviceCloseError(DeviceError):
    """A resource close or stop operation failed and remains observable."""


class DeviceUncertainError(DeviceError):
    """A previous owner left a stop/torque state that is not verified."""


class DeviceStateError(DeviceError):
    """The lifecycle state file is unreadable and cannot be rewritten safely."""


@dataclass(frozen=True, slots=True)
class ResourceIdentity:
    """Canonical physical identity used by both in-process and OS locks.

    ``value`` is a resolved path for path-like resources and an explicit
    serial/bus identifier otherwise.  The node and resource kind are part of
    the key because the same path text on two deployment nodes is unrelated.
    """

    node: str
    kind: str
    value: str

    def __post_init__(self) -> None:
        for name in ("node", "kind", "value"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"resource {name} must not be empty")

    @property
    def key(self) -> str:
        return f"{self.node}:{self.kind}:{self.value}"

    @property
    def physical_key(self) -> str:
        """Key shared by local lock and lifecycle state across node aliases."""

        return f"{self.kind}:{self.value}"

    def __str__(self) -> str:
        return self.key


def canonical_resource_identity(
    node: str,
    kind: str,
    value: str | os.PathLike[str],
    *,
    resolve_path: bool = True,
) -> ResourceIdentity:
    """Resolve a path alias and return the stable owner identity.

    ``realpath`` is deliberately used even when the path does not currently
    exist.  This makes ``/dev/serial/by-id/...`` and its target converge when
    a test or a managed host exposes the symlink, while an unknown mapping
    remains an explicit, diagnosable identity rather than being guessed.
    """

    if not isinstance(node, str) or not node.strip():
        raise ValueError("resource node must not be empty")
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("resource kind must not be empty")
    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("resource value must not be empty")
    raw = os.path.expanduser(raw.strip())
    if resolve_path and (raw.startswith("/") or raw.startswith(".")):
        normalized = os.path.realpath(os.path.abspath(raw))
    else:
        normalized = raw
    return ResourceIdentity(node.strip(), kind.strip(), normalized)


def _physical_key(value: str) -> str:
    """Extract the node-independent part of a persisted identity key."""

    pieces = value.split(":", 2)
    return f"{pieces[1]}:{pieces[2]}" if len(pieces) == 3 else value


@dataclass(frozen=True, slots=True)
class DeviceResource:
    """Open/close behavior for one canonical connection and its components."""

    identity: ResourceIdentity
    opener: Callable[[], Any]
    closer: Callable[[Any], None] | None = None
    preparer: Callable[[Any], None] | None = None
    external_owner: bool = False
    component_identities: tuple[ResourceIdentity, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ResourceIdentity):
            raise TypeError("identity must be a ResourceIdentity")
        if not callable(self.opener):
            raise TypeError("opener must be callable")
        if self.closer is not None and not callable(self.closer):
            raise TypeError("closer must be callable")
        if self.preparer is not None and not callable(self.preparer):
            raise TypeError("preparer must be callable")
        if not isinstance(self.external_owner, bool):
            raise TypeError("external_owner must be a boolean")
        if not isinstance(self.component_identities, tuple) or any(
            not isinstance(identity, ResourceIdentity) for identity in self.component_identities
        ):
            raise TypeError("component_identities must contain ResourceIdentity values")
        identities = {self.identity.key}
        for identity in self.component_identities:
            if identity.key in identities:
                raise ValueError("component_identities must be distinct from identity")
            identities.add(identity.key)


@dataclass(frozen=True, slots=True)
class DeviceStatus:
    """A cheap in-memory lifecycle snapshot suitable for describe/health."""

    identity: ResourceIdentity
    state: Literal["closed", "open", "external", "uncertain"]
    owner_id: str | None = None
    pid: int | None = None
    references: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.key,
            "node": self.identity.node,
            "kind": self.identity.kind,
            "value": self.identity.value,
            "state": self.state,
            "owner_id": self.owner_id,
            "pid": self.pid,
            "references": self.references,
            "error": self.error,
        }


@dataclass(slots=True)
class _Record:
    resource: DeviceResource
    value: Any
    locks: tuple[_PosixResourceLock, ...]
    references: int = 0
    prepared: bool = False
    state: Literal["open", "external", "uncertain"] = "open"
    error: str | None = None
    uncertain_before: bool = False
    leases: set[int] = field(default_factory=set)
    closing: bool = False
    close_error: BaseException | None = None


class _PosixResourceLock:
    """One process-held flock and a small diagnostic owner record."""

    def __init__(self, identity: ResourceIdentity, directory: Path, owner_id: str):
        self.identity = identity
        self.directory = directory
        self.owner_id = owner_id
        # The directory is local to one control node.  Excluding the YAML
        # node label prevents two aliases for the same host from bypassing
        # the physical lock; the node remains in ResourceIdentity for logs.
        lock_key = f"{identity.kind}:{identity.value}"
        self.path = directory / f"{quote(lock_key, safe='')}.lock"
        self._fd: int | None = None

    def acquire(self) -> None:
        if fcntl is None:  # pragma: no cover
            raise DeviceError("POSIX file locking is unavailable on this platform")
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                holder = self._read_holder(fd)
                raise DeviceBusyError(self.identity, holder) from None
            payload = {
                "owner_id": self.owner_id,
                "pid": os.getpid(),
                "identity": self.identity.key,
            }
            encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
            os.ftruncate(fd, 0)
            os.write(fd, encoded)
            os.fsync(fd)
            self._fd = fd
        except BaseException:
            os.close(fd)
            raise

    def _read_holder(self, fd: int) -> dict[str, Any]:
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            raw = os.read(fd, 64 * 1024)
            value = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def release(self) -> None:
        fd = self._fd
        self._fd = None
        if fd is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


class DeviceLease:
    """Reference to a managed resource; close releases only this reference."""

    def __init__(self, manager: DeviceManager, key: str, token: int, value: Any):
        self._manager = manager
        self._key = key
        self._token = token
        self.value = value
        self._released = False

    @property
    def identity(self) -> ResourceIdentity:
        return self._manager._records[self._key].resource.identity

    @property
    def released(self) -> bool:
        return self._released

    def close(self) -> None:
        if self._released:
            return
        self._released = True
        self._manager._release(self._key, self._token)

    def __enter__(self) -> DeviceLease:
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


class DeviceStateStore:
    """Persist only unresolved lifecycle failures, using ordinary JSON state."""

    def __init__(self, path: str | os.PathLike[str] | None):
        self.path = Path(path).expanduser() if path is not None else None
        self._lock = threading.RLock()
        self.error: str | None = None
        self._process_lock_path = self.path.with_name(f".{self.path.name}.lock") if self.path is not None else None

    def load(self) -> dict[str, str]:
        self.error = None
        if self.path is None or not self.path.exists():
            return {}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            self.error = f"lifecycle state could not be read: {error}"
            return {}
        if not isinstance(value, dict):
            self.error = "lifecycle state is malformed: root must be an object"
            return {}
        unresolved = value.get("unresolved", {})
        if not isinstance(unresolved, dict):
            self.error = "lifecycle state is malformed"
            return {}
        if any(
            not isinstance(key, str) or not key.strip() or not isinstance(reason, str)
            for key, reason in unresolved.items()
        ):
            self.error = "lifecycle state is malformed: unresolved entries must be strings"
            return {}
        return dict(unresolved)

    def mark(self, identity: ResourceIdentity, reason: str) -> None:
        if self.path is None:
            return
        with self._lock, self._file_lock():
            unresolved = self.load()
            if self.error is not None:
                # A malformed file represents unknown history.  Replacing
                # it with the new error would silently discard faults
                # from another process, so require explicit recovery.
                raise DeviceStateError(self.error)
            physical_key = identity.physical_key
            unresolved = {key: value for key, value in unresolved.items() if _physical_key(key) != physical_key}
            unresolved[identity.key] = reason
            self._save(unresolved)

    def clear(self, identity: ResourceIdentity) -> None:
        if self.path is None:
            return
        with self._lock, self._file_lock():
            unresolved = self.load()
            if self.error is not None:
                raise DeviceStateError(self.error)
            physical_key = identity.physical_key
            filtered = {key: value for key, value in unresolved.items() if _physical_key(key) != physical_key}
            if len(filtered) != len(unresolved):
                unresolved = filtered
                self._save(unresolved)

    class _FileLock:
        def __init__(self, path: Path):
            self.path = path
            self.fd: int | None = None

        def __enter__(self):
            if fcntl is None:  # pragma: no cover
                return self
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(self.fd, fcntl.LOCK_EX)
            return self

        def __exit__(self, _type, _value, _traceback):
            if self.fd is not None:
                if fcntl is not None:
                    fcntl.flock(self.fd, fcntl.LOCK_UN)
                os.close(self.fd)

    def _file_lock(self) -> _FileLock:
        assert self._process_lock_path is not None
        return self._FileLock(self._process_lock_path)

    def _save(self, unresolved: Mapping[str, str]) -> None:
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump({"unresolved": dict(sorted(unresolved.items()))}, handle)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


class DeviceManager:
    """Own physical resources for one service process.

    The manager opens a resource once, tracks all leases in memory, and closes
    it only after the final lease is released or the service shuts down.  A
    control lease can therefore outlive a task and an observer can disconnect
    without stopping the active controller.  ``close`` attempts every resource
    and raises a combined error after all cleanup has been attempted.
    """

    def __init__(
        self,
        node_id: str = "local",
        owner_id: str | None = None,
        *,
        lock_dir: str | os.PathLike[str] | None = None,
        state_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError("node_id must not be empty")
        self.node_id = node_id
        self.owner_id = owner_id or f"control:{os.getpid()}"
        explicit_lock_dir = lock_dir is not None
        self.lock_dir = Path(
            lock_dir or os.environ.get("RLINF_DEPLOY_DEVICE_LOCK_DIR", "/tmp/rlinf-deploy-device-locks")
        ).expanduser()
        if state_path is None:
            if explicit_lock_dir:
                # Tests and explicitly isolated service instances keep their
                # lifecycle ledger beside their lock directory.  The normal
                # process-wide default uses a durable XDG state path below.
                state_path = self.lock_dir / "state.json"
            else:
                state_root = Path(
                    os.environ.get(
                        "XDG_STATE_HOME",
                        Path.home() / ".local" / "state",
                    )
                ).expanduser()
                state_path = state_root / "rlinf-deploy" / "device-state.json"
        self.state_store = DeviceStateStore(state_path)
        self._records: dict[str, _Record] = {}
        self._unresolved = self.state_store.load()
        self._lock = threading.RLock()
        # Close callbacks run outside this lock because SDK shutdown can
        # block.  The condition makes a lease release and service shutdown
        # single-flight for one record while still allowing independent
        # resources to close concurrently.
        self._condition = threading.Condition(self._lock)
        self._next_token = 1
        self._closed = False

    def acquire(
        self,
        resource: DeviceResource | ResourceIdentity,
        *,
        opener: Callable[[], Any] | None = None,
        closer: Callable[[Any], None] | None = None,
        preparer: Callable[[Any], None] | None = None,
        role: str = "observer",
        prepare: bool = False,
        external_owner: bool | None = None,
    ) -> DeviceLease:
        """Acquire a reference, opening the resource on its first consumer."""

        del role  # Role is retained in callers for audit/readability.
        if isinstance(resource, ResourceIdentity):
            if opener is None:
                raise ValueError("opener is required for a ResourceIdentity")
            resource = DeviceResource(
                resource,
                opener,
                closer=closer,
                preparer=preparer,
                external_owner=bool(external_owner),
            )
        elif not isinstance(resource, DeviceResource):
            raise TypeError("resource must be a DeviceResource or ResourceIdentity")
        if external_owner is not None and external_owner != resource.external_owner:
            resource = DeviceResource(
                resource.identity,
                resource.opener,
                resource.closer,
                resource.preparer,
                external_owner,
                resource.component_identities,
            )
        identity = resource.identity
        key = identity.key
        with self._condition:
            if self._closed:
                raise DeviceError("device manager is closed")
            self._refresh_unresolved()
            record = self._records.get(key)
            if record is not None and record.closing:
                # A close callback may run for an unbounded SDK interval. Do
                # not hand its value to a new lease while shutdown is in
                # flight, and never make an HTTP/observation caller wait for
                # an SDK close that may not return.  The same manager can
                # explicitly retry cleanup after the close callback exits.
                raise DeviceBusyError(
                    identity,
                    {
                        "owner_id": self.owner_id,
                        "pid": os.getpid(),
                        "state": "closing",
                    },
                )
            if record is not None:
                if record.resource.external_owner != resource.external_owner:
                    raise DeviceError(f"resource {key!r} has conflicting external-owner declarations")
                existing_components = tuple(item.key for item in record.resource.component_identities)
                requested_components = tuple(item.key for item in resource.component_identities)
                if existing_components != requested_components:
                    raise DeviceError(f"resource {key!r} has conflicting component identities")
                if record.references == 0 and record.state == "uncertain":
                    raise DeviceUncertainError(f"resource {key!r} remains held after an unresolved close failure")
                if prepare and (
                    not record.prepared
                    or (record.resource.external_owner and getattr(record.value, "prepared", True) is False)
                ):
                    if self.state_store.error is not None:
                        raise DeviceUncertainError(self.state_store.error)
                    reason = next(
                        (
                            self._unresolved.get(item)
                            for item in (key, *existing_components)
                            if self._unresolved.get(item) is not None
                        ),
                        None,
                    )
                    if reason is not None:
                        raise DeviceUncertainError(f"resource {key!r} has unresolved lifecycle state: {reason}")
                    self._prepare_record(record, preparer=resource.preparer)
                token = self._next_token
                self._next_token += 1
                record.references += 1
                record.leases.add(token)
                return DeviceLease(self, key, token, record.value)

            identities = (identity, *resource.component_identities)
            identity_keys = tuple(item.key for item in identities)
            reason = next(
                (self._unresolved.get(item) for item in identity_keys if self._unresolved.get(item) is not None),
                None,
            )
            if prepare:
                if self.state_store.error is not None:
                    raise DeviceUncertainError(self.state_store.error)
                if reason is not None:
                    raise DeviceUncertainError(f"resource {key!r} has unresolved lifecycle state: {reason}")
            process_locks: list[_PosixResourceLock] = []
            if not resource.external_owner:
                try:
                    for physical_identity in sorted(identities, key=lambda item: item.key):
                        process_lock = _PosixResourceLock(physical_identity, self.lock_dir, self.owner_id)
                        process_lock.acquire()
                        process_locks.append(process_lock)
                except BaseException:
                    for process_lock in reversed(process_locks):
                        process_lock.release()
                    raise
            value: Any | None = None
            try:
                value = resource.opener()
                if prepare:
                    self._prepare_value(resource, value)
            except BaseException as error:
                cleanup_error: BaseException | None = None
                if value is not None:
                    try:
                        if resource.closer is not None:
                            resource.closer(value)
                        else:
                            close = getattr(value, "close", None)
                            if callable(close):
                                close()
                    except BaseException as close_error:
                        cleanup_error = close_error
                if cleanup_error is not None:
                    reason = str(cleanup_error)
                    for identity_key in identity_keys:
                        self._unresolved[identity_key] = reason
                    try:
                        for physical_identity in identities:
                            self.state_store.mark(physical_identity, reason)
                    except DeviceStateError:
                        # Preserve a malformed state file and retain the live
                        # handle/lock as the only reliable exclusion marker.
                        pass
                    self._records[key] = _Record(
                        resource=resource,
                        value=value,
                        locks=tuple(process_locks),
                        references=0,
                        prepared=False,
                        state="uncertain",
                        error=reason,
                        uncertain_before=True,
                    )
                else:
                    for process_lock in reversed(process_locks):
                        process_lock.release()
                raise DeviceOpenError(f"failed to open {key!r}: {error}") from error
            record = _Record(
                resource=resource,
                value=value,
                locks=tuple(process_locks),
                references=1,
                prepared=prepare,
                state=(
                    "uncertain"
                    if any(item in self._unresolved for item in identity_keys) or self.state_store.error is not None
                    else "external"
                    if resource.external_owner
                    else "open"
                ),
                uncertain_before=any(item in self._unresolved for item in identity_keys)
                or self.state_store.error is not None,
            )
            token = self._next_token
            self._next_token += 1
            record.leases.add(token)
            self._records[key] = record
            return DeviceLease(self, key, token, value)

    def _prepare_value(self, resource: DeviceResource, value: Any) -> None:
        preparer = resource.preparer
        if preparer is None:
            method = getattr(value, "prepare", None)
            if not callable(method):
                raise DeviceError(f"resource {resource.identity.key!r} does not support explicit prepare")
            preparer = method
        preparer(value) if resource.preparer is not None else preparer()

    def _prepare_record(
        self,
        record: _Record,
        *,
        preparer: Callable[[Any], None] | None = None,
    ) -> None:
        try:
            resource = record.resource
            if preparer is None:
                self._prepare_value(resource, record.value)
            else:
                preparer(record.value)
        except BaseException as error:
            if isinstance(error, RobotPreparationRefused):
                raise
            record.state = "uncertain"
            record.error = str(error)
            record.uncertain_before = True
            for identity in (
                record.resource.identity,
                *record.resource.component_identities,
            ):
                self.state_store.mark(identity, str(error))
            raise DeviceError(f"resource {record.resource.identity.key!r} prepare failed: {error}") from error
        record.prepared = True
        record.error = None
        record.state = "external" if record.resource.external_owner else "open"

    def _release(self, key: str, token: int) -> None:
        with self._condition:
            record = self._records.get(key)
            if record is None or token not in record.leases:
                return
            record.leases.remove(token)
            record.references -= 1
            if record.references:
                return
        # Do not hold the manager lock while an SDK close runs.  The helper
        # coordinates a concurrent manager.close() call on this record.
        error = self._close_record_once(key, record)
        if error is not None:
            raise error

    def _close_record_once(self, key: str, record: _Record) -> BaseException | None:
        """Close one record exactly once per in-flight transaction.

        A waiter observes the first callback's result instead of invoking the
        closer a second time.  A later explicit call after a failed close is a
        new transaction and may retry cleanup.
        """

        with self._condition:
            if self._records.get(key) is not record:
                return None
            if record.closing:
                while record.closing:
                    self._condition.wait()
                if self._records.get(key) is not record:
                    return None
                return record.close_error
            record.closing = True
            record.close_error = None

        error: BaseException | None = None
        try:
            self._close_record(record)
            for process_lock in reversed(record.locks):
                process_lock.release()
        except BaseException as caught:
            error = caught

        with self._condition:
            record.closing = False
            if error is None:
                self._records.pop(key, None)
                record.close_error = None
            else:
                # Keep the OS lock while the SDK/thread may still own the
                # connection. Releasing it here could allow another process
                # to reopen the same hardware during unresolved cleanup.
                record.references = 0
                record.leases.clear()
                record.state = "uncertain"
                record.close_error = error
                self._records[key] = record
            self._condition.notify_all()
        return error

    def _close_record(self, record: _Record) -> None:
        closer = record.resource.closer
        try:
            if closer is not None:
                closer(record.value)
            else:
                method = getattr(record.value, "close", None)
                if callable(method):
                    method()
        except BaseException as error:
            record.state = "uncertain"
            record.error = str(error)
            identities = (
                record.resource.identity,
                *record.resource.component_identities,
            )
            for identity in identities:
                self._unresolved[identity.key] = str(error)
            try:
                for identity in identities:
                    self.state_store.mark(identity, str(error))
            except DeviceStateError as state_error:
                # Keep the original close failure visible while retaining the
                # unreadable-state diagnostic for fail-closed preparation.
                record.error = f"{error}; lifecycle state unavailable: {state_error}"
            raise DeviceCloseError(f"close failed for {record.resource.identity.key!r}: {record.error}") from error
        # A successful close of a later read-only attach cannot prove that a
        # previous stop/torque failure was repaired.  Keep that marker until
        # an explicit operator verification clears it.
        if not record.uncertain_before:
            identities = (
                record.resource.identity,
                *record.resource.component_identities,
            )
            for identity in identities:
                self._unresolved.pop(identity.key, None)
            try:
                for identity in identities:
                    self.state_store.clear(identity)
            except DeviceStateError as error:
                record.state = "uncertain"
                record.error = str(error)
                record.uncertain_before = True
                raise DeviceCloseError(
                    f"closed {record.resource.identity.key!r}, but lifecycle state could not be updated: {error}"
                ) from error

    def status(self, identity: ResourceIdentity | None = None) -> tuple[DeviceStatus, ...] | DeviceStatus:
        """Return lifecycle state without touching or reopening hardware."""

        with self._lock:
            self._refresh_unresolved()
            if identity is not None:
                return self._status_one(identity)
            keys = set(self._records) | set(self._unresolved)
            statuses = [self._status_one(record.resource.identity) for record in self._records.values()]
            known = {item.identity.key for item in statuses}
            for key in sorted(keys - known):
                pieces = key.split(":", 2)
                if len(pieces) != 3:
                    continue
                node, kind, value = pieces
                statuses.append(
                    DeviceStatus(
                        canonical_resource_identity(node, kind, value),
                        "uncertain",
                        error=self._unresolved.get(key),
                    )
                )
            return tuple(sorted(statuses, key=lambda item: item.identity.key))

    def _status_one(self, identity: ResourceIdentity) -> DeviceStatus:
        self._refresh_unresolved()
        record = self._records.get(identity.key)
        if record is not None:
            owner = self.owner_id
            return DeviceStatus(
                identity,
                record.state,
                owner_id=owner,
                pid=os.getpid(),
                references=record.references,
                error=record.error,
            )
        reason = self._unresolved.get(identity.key)
        return DeviceStatus(identity, "uncertain" if reason else "closed", error=reason)

    def _refresh_unresolved(self) -> None:
        current = self.state_store.load()
        if self.state_store.error is not None:
            self._unresolved = {}
        else:
            # Node labels identify a deployment in diagnostics, while this
            # manager's lock directory is local to one host.  Normalize old
            # state written under an alias to the current label so a restart
            # cannot hide an unresolved stop/torque fault.
            self._unresolved = {}
            for key, reason in current.items():
                pieces = key.split(":", 2)
                if len(pieces) == 3:
                    _, kind, value = pieces
                    key = f"{self.node_id}:{kind}:{value}"
                self._unresolved[key] = reason

    def clear_uncertainty(self, identity: ResourceIdentity) -> None:
        """Clear a recorded fault only after an operator verified recovery."""

        with self._lock:
            self._refresh_unresolved()
            if self.state_store.error is not None:
                raise DeviceUncertainError(self.state_store.error)
            self._unresolved.pop(identity.key, None)
            self.state_store.clear(identity)

    def mark_uncertain(self, identity: ResourceIdentity, reason: str) -> None:
        """Persist an unresolved stop/torque error from a control owner."""

        with self._lock:
            record = self._records.get(identity.key)
            identities = (
                (record.resource.identity, *record.resource.component_identities) if record is not None else (identity,)
            )
            for physical_identity in identities:
                self._unresolved[physical_identity.key] = reason
                self.state_store.mark(physical_identity, reason)
            if record is not None:
                record.state = "uncertain"
                record.error = reason
                record.uncertain_before = True

    def close(self) -> None:
        """Close all resources, attempting cleanup even when one close fails."""

        with self._condition:
            self._closed = True
            records = tuple(self._records.items())
        if not records:
            return
        errors: list[BaseException] = []
        for key, record in records:
            close_error = self._close_record_once(key, record)
            if close_error is not None:
                errors.append(close_error)
        if errors:
            message = "; ".join(str(error) for error in errors)
            raise DeviceCloseError(message) from errors[0]


__all__ = [
    "DeviceBusyError",
    "DeviceCloseError",
    "DeviceError",
    "DeviceStateError",
    "DeviceLease",
    "DeviceManager",
    "DeviceOpenError",
    "DeviceResource",
    "DeviceStateStore",
    "DeviceStatus",
    "DeviceUncertainError",
    "ResourceIdentity",
    "canonical_resource_identity",
]
