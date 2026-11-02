"""Deploy one optional Go2 owner, independently of any navigation policy."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import posixpath
from dataclasses import asdict
from pathlib import Path

from embodirun.deployment import supervisor as supervisor_module
from embodirun.deployment.executor import Command
from embodirun.deployment.plan import ServiceSpec
from embodirun.deployment.supervisor import ServiceSupervisor

from .transport import RemoteTransport

OWNER_MODULE = "embodirun_go2_owner"


class Go2OwnerDeployer:
    """Upload hardware code only and supervise one process with one lifecycle."""

    def __init__(self, transport: RemoteTransport, remote_root: str) -> None:
        self.transport = transport
        self.remote_root = transport.resolve_path(remote_root)
        if self.remote_root == "/":
            raise ValueError("remote root cannot be the filesystem root")
        self.source_root = posixpath.join(self.remote_root, "src")
        self.run_root = posixpath.join(self.remote_root, "run")
        self.log_root = posixpath.join(self.remote_root, "logs")
        self.supervisor_path = posixpath.join(self.remote_root, "supervisor.py")

    def probe(self, python: str = "python3") -> dict[str, object]:
        version = self.transport.run(Command((python, "--version")))
        output = (version.stdout or version.stderr).strip()
        if not output.startswith("Python "):
            raise RuntimeError("remote Python probe returned an invalid version")
        return {
            "python": python,
            "python_version": output[7:],
            "platform": self.transport.run(Command(("uname", "-s"))).stdout.strip(),
            "machine": self.transport.run(Command(("uname", "-m"))).stdout.strip(),
        }

    def install(self, package_root: Path | None = None) -> dict[str, object]:
        root = (package_root or _default_package_root()).resolve()
        if root.name != OWNER_MODULE or not (root / "__main__.py").is_file():
            raise ValueError("package root must be the embodirun_go2_owner source directory")
        self.transport.make_dirs(self.run_root, mode=0o700)
        self.transport.make_dirs(self.log_root, mode=0o700)
        digest = hashlib.sha256()
        files = [
            path for path in sorted(root.rglob("*.py")) if not path.is_symlink() and "__pycache__" not in path.parts
        ]
        for path in files:
            relative = path.relative_to(root).as_posix()
            digest.update(relative.encode() + b"\0" + path.read_bytes())
            self.transport.put_file(path, posixpath.join(self.source_root, OWNER_MODULE, relative))
        # The shared supervisor is standalone stdlib code; no Host package is uploaded.
        self.transport.put_file(Path(supervisor_module.__file__), self.supervisor_path)
        manifest = {
            "format": 2,
            "package": OWNER_MODULE,
            "file_count": len(files),
            "sha256": digest.hexdigest(),
            "source_root": self.source_root,
        }
        self.transport.put_bytes(
            json.dumps(manifest, indent=2).encode(), posixpath.join(self.remote_root, "deployment-manifest.json")
        )
        return manifest

    def _supervisor(self, python: str) -> ServiceSupervisor:
        return ServiceSupervisor(
            self.transport,
            python=python,
            agent_path=self.supervisor_path,
            run_root=self.run_root,
            log_root=self.log_root,
        )

    def start(
        self,
        config: Path,
        *,
        python: str = "python3",
        supervisor_python: str = "python3.10",
        operator_ready: bool = False,
        sdk_pythonpath: str | None = None,
        token: str | None = None,
    ) -> dict[str, object]:
        value = json.loads(config.read_text())
        if not isinstance(value, dict) or set(value) - {"server", "control", "camera"}:
            raise ValueError("expected an owner configuration with server, control and camera")
        server = value.get("server", {})
        if not isinstance(server, dict):
            raise ValueError("server must be an object")
        host, port = server.get("host", "127.0.0.1"), server.get("port", 8080)
        token_name = server.get("token_env", "GO2_API_TOKEN")
        if (
            not isinstance(token_name, str)
            or not token_name
            or token_name
            in {
                "PYTHONPATH",
                "PATH",
                "LD_LIBRARY_PATH",
                "EMBODIRUN_SERVICE_ID",
            }
        ):
            raise ValueError("server.token_env must name a dedicated credential environment variable")
        remote_config = posixpath.join(self.remote_root, "owner.json")
        environment = {"PYTHONPATH": self.source_root + (":" + sdk_pythonpath if sdk_pythonpath else "")}
        if token:
            environment[token_name] = token
        self.transport.put_bytes(json.dumps(value).encode(), remote_config, mode=0o600)
        argv = (python, "-m", OWNER_MODULE, "--config", remote_config)
        # Validate on the vendor interpreter without constructing SDK/camera resources.
        self.transport.run(Command((*argv, "--validate"), environment=environment))
        endpoint = f"http://{host}:{port}"
        service = ServiceSpec(
            service_id="go2",
            kind="control",
            node="go2",
            environment_id="go2-owner",
            endpoint=endpoint,
            health_endpoint=endpoint + "/health",
            command=Command((*argv, *(("--operator-ready",) if operator_ready else ())), environment=environment),
        )
        status = self._supervisor(supervisor_python).start(service)
        return {"services": {"go2": asdict(status)}}

    def status(self, supervisor_python: str = "python3.10") -> dict[str, object]:
        return {"services": {"go2": asdict(self._supervisor(supervisor_python).status("go2"))}}

    def stop(self, supervisor_python: str = "python3.10") -> dict[str, object]:
        return {"services": {"go2": asdict(self._supervisor(supervisor_python).stop("go2"))}}


def _default_package_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "integrations/go2_owner/src" / OWNER_MODULE
        if (candidate / "__main__.py").is_file():
            return candidate
    spec = importlib.util.find_spec(OWNER_MODULE)
    if spec is not None and spec.origin is not None:
        return Path(spec.origin).parent
    raise ValueError("install integrations/go2_owner locally or pass --package-root to its source directory")
