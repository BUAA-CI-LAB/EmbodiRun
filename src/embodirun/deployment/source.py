"""Prepare pinned checkouts and install deterministic local source releases."""

from __future__ import annotations

import gzip
import hashlib
import io
import posixpath
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .executor import Command, Executor

_DEPENDENCY_FILES = ("pyproject.toml", "uv.lock")
_ROOT_FILES = (*_DEPENDENCY_FILES, "README.md")
_SOURCE_DIRS = ("src/embodirun", "integrations")

# Historical and internal repository names that refer to the same public
# project.  Managed checkouts cloned before the rename keep working instead of
# forcing a fresh clone.
_REPOSITORY_ALIASES = {
    "embodirun": "EmbodiRun",
    "embodirun-inference": "EmbodiInfer",
    "embodiinfer": "EmbodiInfer",
}


def repository_identity(repository: str) -> str:
    """Normalize a GitHub repository reference for rename-tolerant comparison."""

    normalized = repository.strip().removesuffix(".git")
    for prefix in (
        "git@github.com:buaa-ci-lab/",
        "https://github.com/buaa-ci-lab/",
        "http://github.com/buaa-ci-lab/",
    ):
        lowered = normalized.lower()
        if lowered.startswith(prefix):
            name = lowered.removeprefix(prefix)
            return _REPOSITORY_ALIASES.get(name, name)
    return normalized.lower()


class SourceError(ValueError):
    """Deployment source cannot be resolved or installed safely."""


@dataclass(frozen=True, slots=True)
class SourceArchive:
    """Deterministic local source payload and its dependency identity."""

    content: bytes
    digest: str
    dependency_digest: str
    dependency_files: dict[str, bytes]


@dataclass(frozen=True, slots=True)
class SourceRelease:
    """One content-addressed source release installed on a node."""

    path: str
    uploaded: bool


class ProjectManager:
    """Prepare a managed Git checkout at an exact configured revision."""

    def __init__(self, executor: Executor, *, git_executable: str = "git") -> None:
        self.executor = executor
        self.git_executable = git_executable

    def prepare(self, *, repository: str, revision: str, project_dir: str) -> None:
        if not repository or not revision or not project_dir:
            raise SourceError("repository, revision, and project_dir are required")
        parent = posixpath.dirname(project_dir)
        self.executor.run(Command(("mkdir", "-p", parent)))
        checkout = self.executor.run(
            Command(("test", "-d", posixpath.join(project_dir, ".git"))),
            check=False,
        )
        if checkout.exit_code != 0:
            self.executor.run(
                Command(
                    (
                        self.git_executable,
                        "clone",
                        "--no-checkout",
                        repository,
                        project_dir,
                    ),
                    timeout_s=600.0,
                )
            )
        origin = self.executor.run(
            Command(
                (
                    self.git_executable,
                    "-C",
                    project_dir,
                    "remote",
                    "get-url",
                    "origin",
                )
            )
        ).stdout.strip()
        if repository_identity(origin) != repository_identity(repository):
            raise SourceError(f"managed checkout {project_dir!r} has unexpected origin {origin!r}")
        object_name = f"{revision}^{{commit}}"
        present = self.executor.run(
            Command(
                (
                    self.git_executable,
                    "-C",
                    project_dir,
                    "cat-file",
                    "-e",
                    object_name,
                )
            ),
            check=False,
        )
        if present.exit_code != 0:
            self.executor.run(
                Command(
                    (
                        self.git_executable,
                        "-C",
                        project_dir,
                        "fetch",
                        "origin",
                        revision,
                    ),
                    timeout_s=600.0,
                )
            )
        self.executor.run(
            Command(
                (
                    self.git_executable,
                    "-C",
                    project_dir,
                    "checkout",
                    "--detach",
                    revision,
                ),
                timeout_s=600.0,
            )
        )
        resolved = self.executor.run(
            Command(
                (
                    self.git_executable,
                    "-C",
                    project_dir,
                    "rev-parse",
                    "HEAD",
                    object_name,
                )
            )
        ).stdout.splitlines()
        if len(resolved) != 2 or resolved[0] != resolved[1]:
            raise SourceError(f"managed checkout {project_dir!r} did not resolve to {revision!r}")

    def submodule_revision(self, *, project_dir: str, revision: str, path: str) -> str:
        """Read a pinned gitlink from a commit without checking out the submodule."""

        result = self.executor.run(
            Command(
                (
                    self.git_executable,
                    "-C",
                    project_dir,
                    "ls-tree",
                    revision,
                    "--",
                    path,
                ),
                timeout_s=20.0,
            )
        )
        fields = result.stdout.strip().split()
        if (
            len(fields) != 4
            or fields[:2] != ["160000", "commit"]
            or fields[3] != path
            or len(fields[2]) not in {40, 64}
            or any(character not in "0123456789abcdef" for character in fields[2])
        ):
            raise SourceError(f"Deploy revision {revision!r} does not pin a valid submodule at {path!r}")
        return fields[2]


def managed_root(home: str, base: str, deployment_name: str) -> str:
    """Resolve the deployment root against the probed node home directory."""

    if not home.startswith("/") or not base or "\x00" in base:
        raise SourceError("node home must be absolute and managed root must be valid")
    if base == "~":
        resolved = home
    elif base.startswith("~/"):
        resolved = posixpath.join(home, base[2:])
    elif base.startswith("/"):
        resolved = base
    else:
        if ".." in PurePosixPath(base).parts:
            raise SourceError("relative managed root cannot contain '..'")
        resolved = posixpath.join(home, base)
    root = posixpath.normpath(posixpath.join(resolved, deployment_name))
    if root == "/":
        raise SourceError("managed deployment root cannot be filesystem root")
    return root


def active_deploy_project(deployment_root: str) -> str:
    """Return the stable path to the currently active Deploy source tree."""

    return posixpath.join(deployment_root, "overlays", "deploy", "current")


def build_source_archive(value: Path) -> SourceArchive:
    """Validate and archive one local EmbodiRun source tree."""

    source = value.expanduser().resolve()
    missing = [name for name in (*_ROOT_FILES, "src/embodirun") if not (source / name).exists()]
    if missing:
        raise SourceError(f"{source} is not an EmbodiRun source tree; missing: " + ", ".join(missing))
    package_root = source / "src" / "embodirun"
    if not package_root.is_dir():
        raise SourceError(f"{package_root} is not a directory")

    files = {name: (source / name).read_bytes() for name in _ROOT_FILES}
    for directory in _SOURCE_DIRS:
        root = source / directory
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            relative_parts = path.relative_to(source).parts
            if path.is_symlink() or not path.is_file() or _excluded(relative_parts):
                continue
            files[PurePosixPath(*relative_parts).as_posix()] = path.read_bytes()

    digest = _content_digest(files)
    dependency_files = {name: files[name] for name in _DEPENDENCY_FILES}
    dependency_digest = _content_digest(dependency_files)
    output = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed,
        tarfile.open(
            fileobj=compressed,
            mode="w",
            format=tarfile.PAX_FORMAT,
        ) as archive,
    ):
        directories = {
            parent.as_posix()
            for name in files
            for parent in PurePosixPath(name).parents
            if parent != PurePosixPath(".")
        }
        for name in sorted(directories, key=lambda item: (item.count("/"), item)):
            info = tarfile.TarInfo(name)
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            _normalize_tar_info(info)
            archive.addfile(info)
        for name, content in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o644
            _normalize_tar_info(info)
            archive.addfile(info, io.BytesIO(content))
    return SourceArchive(output.getvalue(), digest, dependency_digest, dependency_files)


def install_source_release(
    executor: Executor,
    *,
    overlay_root: str,
    archive: SourceArchive,
) -> SourceRelease:
    """Install one content-addressed source archive if it is not already present."""

    release = posixpath.join(overlay_root, "releases", archive.digest)
    complete_marker = posixpath.join(release, ".complete")
    existing = executor.run(Command(("test", "-f", complete_marker)), check=False)
    if existing.exit_code == 0:
        return SourceRelease(release, uploaded=False)
    if existing.exit_code != 1:
        detail = existing.stderr.strip() or f"exit code {existing.exit_code}"
        raise SourceError(f"could not inspect Deploy overlay: {detail}")

    archive_path = posixpath.join(
        overlay_root,
        "archives",
        f"{archive.digest}.tar.gz",
    )
    executor.run(
        Command(
            (
                "mkdir",
                "-p",
                posixpath.dirname(archive_path),
                release,
            )
        )
    )
    executor.write_bytes(archive_path, archive.content, mode=0o600)
    executor.run(Command(("tar", "-xzf", archive_path, "-C", release)))
    executor.write_text(complete_marker, f"{archive.digest}\n", mode=0o600)
    return SourceRelease(release, uploaded=True)


def _excluded(parts: tuple[str, ...]) -> bool:
    return "__pycache__" in parts or any(part.endswith((".pyc", ".pyo")) for part in parts)


def _normalize_tar_info(info: tarfile.TarInfo) -> None:
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""


def _content_digest(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name, content in sorted(files.items()):
        encoded_name = name.encode()
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


__all__ = [
    "ProjectManager",
    "SourceArchive",
    "SourceError",
    "SourceRelease",
    "active_deploy_project",
    "build_source_archive",
    "install_source_release",
    "managed_root",
]
