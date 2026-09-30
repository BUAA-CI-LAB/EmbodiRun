"""Run the repository's examples from a small, versioned YAML manifest.

This is orchestration, not a second deployment engine. Host owns distributed
services; the MicroDuck and XLeRobot examples retain their own execution loops.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from embodirun.bindings import binding_definition
from embodirun.deployment.config import load_config
from embodirun.deployment.config.loader import _load_yaml
from embodirun.deployment.plan import build_plan
from examples.configuration import check_report, init_main
from examples.lifecycle import Supervisor, stop_owned
from examples.setup_environment import environment_for_python, environment_path

ROOT = Path(__file__).resolve().parents[1]
HOST = [sys.executable, "-c", "from embodirun.services.host.cli.cli import main; raise SystemExit(main())"]
COMMANDS = ("validate", "plan", "setup", "up", "run", "down", "dry-run", "check", "provision")


def mapping(value: Any, allowed: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError(f"{label} must be a mapping with string keys")
    if unknown := value.keys() - allowed:
        raise ValueError(f"unknown {label} fields: {', '.join(sorted(unknown))}")
    return value


def text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def number(value: Any, label: str, *, integer: bool = False, minimum: float = 0) -> float:
    types = (int,) if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, types) or not math.isfinite(value) or value <= minimum:
        raise ValueError(f"{label} must be a finite {'integer' if integer else 'number'} > {minimum}")
    return value


@dataclass
class Example:
    path: Path
    data: dict[str, Any]

    @property
    def kind(self) -> str:
        return self.data["kind"]

    @property
    def parameters(self) -> dict[str, Any]:
        return self.data["parameters"]

    def resolve(self, value: str) -> Path:
        path = Path(text(value, "path")).expanduser()
        return (self.path.parent / path).resolve()

    @property
    def output(self) -> Path:
        return self.resolve(self.data["output_dir"])

    @property
    def python(self) -> str:
        if "python" in self.data:
            path = Path(self.data["python"]).expanduser()
            return str((self.path.parent / path).absolute())
        if override := os.environ.get("EMBODIRUN_SCENE_PYTHON"):
            return str(Path(shutil.which(override) or override).expanduser().absolute())
        if self.kind in {"snack", "microduck"}:
            candidate = environment_path(self.kind) / "bin/python"
            return str(candidate)
        return sys.executable

    @property
    def environment(self) -> Path:
        if "python" not in self.data and not os.environ.get("EMBODIRUN_SCENE_PYTHON"):
            return environment_path(self.kind)
        return environment_for_python(self.python)

    def host(self, *args: str) -> list[str]:
        return [
            *HOST,
            "--config",
            str(self.resolve(self.parameters["deployment"])),
            "--state-dir",
            str(self.output / "host-state"),
            *args,
        ]


def load_example(path: Path) -> Example:
    path = path.expanduser().resolve()
    data = mapping(_load_yaml(path), {"version", "name", "kind", "output_dir", "python", "parameters"}, "example")
    if type(data.get("version")) is not int or data["version"] != 1:
        raise ValueError("example.version must be 1")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", text(data.get("name"), "name")):
        raise ValueError("name must be a simple identifier")
    example = Example(path, data)
    text(data.get("output_dir"), "output_dir")
    if "python" in data:
        text(data["python"], "python")
    kind = text(data.get("kind"), "kind")
    fields = {
        "rollout": {"deployment", "runtimes", "prompt", "chunks", "chunk_steps", "control_hz", "request_timeout"},
        "microduck": {
            "project_root",
            "inference_root",
            "checkpoint",
            "episodes",
            "manifest",
            "num_episodes",
            "episode_offset",
            "max_steps",
            "seed",
            "fps",
            "slurm",
            "partition",
            "cpus",
            "time",
        },
        "snack": {"config", "deployment"},
    }
    if kind not in fields:
        raise ValueError("kind must be rollout, microduck, or snack")
    p = mapping(data.get("parameters"), fields[kind], "parameters")
    if kind == "rollout":
        deployment = load_config(example.resolve(p.get("deployment")))
        build_plan(deployment)
        runtimes = p.get("runtimes")
        if not isinstance(runtimes, list) or not runtimes or any(not isinstance(r, str) for r in runtimes):
            raise ValueError("runtimes must be a nonempty list of runtime IDs")
        if len(set(runtimes)) != len(runtimes):
            raise ValueError("runtimes must be unique")
        text(p.get("prompt"), "prompt")
        for key in ("chunks", "chunk_steps"):
            number(p.get(key), key, integer=True)
        for key in ("control_hz", "request_timeout"):
            number(p.get(key), key)
        robots = []
        for name in runtimes:
            if name not in deployment.runtimes:
                raise ValueError(f"unknown runtime: {name}")
            runtime = deployment.runtimes[name]
            if runtime.robot is None or runtime.binding is None:
                raise ValueError(f"{name} must be a robot runtime with a policy binding")
            if p["chunk_steps"] > binding_definition(runtime.binding).maximum_chunk_steps:
                raise ValueError(f"chunk_steps exceeds the binding limit for {name}")
            robot = deployment.robots[runtime.robot]
            robots.append((robot.node, robot.resource or runtime.robot))
        if len(set(robots)) != len(robots):
            raise ValueError("parallel runtimes must use separate robots")
    elif kind == "microduck":
        for key in ("project_root", "inference_root", "checkpoint", "episodes", "manifest"):
            text(p.get(key), key)
        for key in ("num_episodes", "max_steps", "cpus"):
            number(p.get(key), key, integer=True)
        for key in ("episode_offset", "seed"):
            number(p.get(key), key, integer=True, minimum=-1)
        if type(p.get("fps")) is not int or p["fps"] not in {5, 10, 20}:
            raise ValueError("fps must be 5, 10, or 20")
        if p.get("slurm") not in {"off", "auto"}:
            raise ValueError('slurm must be "off" or auto (quote "off" in YAML)')
        text(p.get("time"), "time")
        if "partition" in p:
            text(p["partition"], "partition")
    else:
        from examples.xlerobot_snack_delivery.run import load_recipe_config

        load_recipe_config(example.resolve(p.get("config")), example.resolve(p.get("deployment")))
        services = _load_yaml(example.resolve(p.get("deployment")))
        if not isinstance(services, dict) or not {"owner", "control", "model", "cameras"} <= services.keys():
            raise ValueError("snack deployment needs owner, control, model, and cameras")
        text(services["owner"].get("hardware_config"), "owner.hardware_config")
        if not isinstance(services["model"], dict) or not isinstance(services["cameras"], dict):
            raise ValueError("snack deployment model and cameras must be mappings")
        text(services["model"].get("endpoint"), "model.endpoint")
        if any(not isinstance(role, str) for role in services["cameras"].values()):
            raise ValueError("snack deployment camera roles must be strings")
    return example


def commands(
    example: Example, action: str, output: Path, *, mode: str = "hardware"
) -> tuple[list[list[str]], dict[str, str]]:
    """Build argv without invoking a shell or importing optional GPU packages."""
    p = example.parameters
    env: dict[str, str] = {}
    if action == "setup" and example.kind in {"snack", "microduck"}:
        command = [
            sys.executable,
            str(ROOT / "examples/setup_environment.py"),
            example.kind,
            "--environment",
            str(example.environment),
        ]
        if example.kind == "snack":
            command.extend(["--mode", mode])
        return [command], env
    if example.kind == "rollout":
        if action == "setup":
            return [example.host("init"), example.host("sync", "--source", str(ROOT))], env
        if action in {"up", "down"}:
            args = ("up", "--wait-timeout", "600") if action == "up" else ("down",)
            return [example.host(*args)], env
        if action == "run":
            return [
                example.host(
                    "run",
                    "--runtime",
                    runtime,
                    "--prompt",
                    p["prompt"],
                    "--max-steps",
                    str(p["chunks"]),
                    "--chunk-steps",
                    str(p["chunk_steps"]),
                    "--control-hz",
                    str(p["control_hz"]),
                    "--request-timeout",
                    str(p["request_timeout"]),
                )
                for runtime in p["runtimes"]
            ], env
    elif example.kind == "snack":
        directory = ROOT / "examples/xlerobot_snack_delivery"
        if action == "up":
            return [
                [
                    example.python,
                    str(directory / "services.py"),
                    "--deployment",
                    str(example.resolve(p["deployment"])),
                    "--state-dir",
                    str(example.output / "services"),
                ]
            ], env
        if action in {"run", "dry-run"}:
            args = [
                example.python,
                str(directory / "run.py"),
                "--config",
                str(example.resolve(p["config"])),
                "--deployment",
                str(example.resolve(p["deployment"])),
                "--output",
                str(output / "result"),
                "--mode",
                "hardware" if action == "run" else "dry-run",
            ]
            if action == "dry-run":
                args.append("--auto-confirm")
            return [args], env
    elif action in {"run", "check", "provision"}:
        env = {
            "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT / "integrations/microduck_vln/src")]),
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": str(p["cpus"]),
            "MKL_NUM_THREADS": str(p["cpus"]),
            "OPENBLAS_NUM_THREADS": "1",
        }
        args = [example.python, str(ROOT / "examples/microduck_vln/run_demo.py")]
        for key in ("project_root", "inference_root", "checkpoint", "episodes", "manifest"):
            args += ["--" + key.replace("_", "-"), str(example.resolve(p[key]))]
        for key in ("num_episodes", "episode_offset", "max_steps", "seed", "fps"):
            args += ["--" + key.replace("_", "-"), str(p[key])]
        args += ["--output", str(output / "result")]
        if action == "check":
            args.append("--check-only")
        if action == "provision":
            args.append("--write-manifest")
        elif p["slurm"] == "auto" and not os.environ.get("SLURM_JOB_ID"):
            allocation = [
                "srun",
                "--nodes=1",
                "--ntasks=1",
                f"--cpus-per-task={p['cpus']}",
                "--gres=gpu:1",
                f"--time={p['time']}",
                "--job-name=microduck-vln",
                "--export=ALL",
            ]
            if "partition" in p:
                allocation.append(f"--partition={p['partition']}")
            args = [*allocation, *args]
        return [args], env
    raise ValueError(f"{action} is not supported for {example.kind}; see examples/README.md")


def execute(
    argv: list[list[str]],
    env: dict[str, str],
    output: Path,
    *,
    parallel: bool,
    interactive: bool,
    supervisor: Supervisor | None = None,
) -> None:
    """Own and reap local children; preserve stdin for supervised handover."""
    children: list[subprocess.Popen] = []
    logs = []
    try:
        for index, command in enumerate(argv):
            print("+ " + shlex.join(command), flush=True)
            log = None if interactive else (output / f"command-{index}.log").open("w")
            if log is not None:
                logs.append(log)
            child = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            children.append(child)
            if not parallel:
                while child.poll() is None:
                    if supervisor is not None and supervisor.cancelled.wait(0.1):
                        raise KeyboardInterrupt
                    if supervisor is None:
                        time.sleep(0.1)
                if child.returncode != 0:
                    log_path = output / f"command-{index}.log"
                    raise RuntimeError(f"command {index} failed; inspect {output if interactive else log_path}")
        while True:
            if supervisor is not None and supervisor.cancelled.is_set():
                raise KeyboardInterrupt
            # Poll every client: short-circuiting on the first live child can
            # hide a later client's failure until the first rollout finishes.
            codes = [child.poll() for child in children]
            if any(code not in {None, 0} for code in codes):
                raise RuntimeError(f"one rollout failed; see {output}")
            if all(code is not None for code in codes):
                break
            time.sleep(0.1)
    finally:
        for child in children:
            if child.poll() is None:
                # Let the scene supervisor stop its children in the right order.
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        for log in logs:
            log.close()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "init":
        try:
            return init_main(argv[1:])
        except (OSError, ValueError) as error:
            print(f"example: {error}", file=sys.stderr)
            return 2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--allow-hardware", action="store_true", help="permit hardware service startup or motion")
    parser.add_argument("--mode", choices=("software", "hardware"), help="setup/check mode (default: hardware)")
    parser.add_argument("--json", action="store_true", help="emit one machine-readable local check report")
    args = parser.parse_args(argv)
    stop_error: RuntimeError | None = None
    output: Path | None = None
    report_path: Path | None = None
    try:
        if args.mode is not None and args.command not in {"setup", "check"}:
            raise ValueError("--mode is available for setup and check")
        if args.json and args.command != "check":
            raise ValueError("--json is available for check")
        if args.command == "down":
            path = args.config.expanduser().resolve()
            data = _load_yaml(path)
            example = Example(path, data)
            if example.kind not in {"rollout", "snack", "microduck"}:
                raise ValueError("unknown recipe kind")
        else:
            example = load_example(args.config)
        if args.mode is not None and example.kind != "snack":
            raise ValueError("--mode is available only for XLeRobot recipe setup/check")
        if args.command == "down":
            try:
                stop_owned(example.output)
            except RuntimeError as error:
                if example.kind != "rollout":
                    raise
                stop_error = error
                print(f"Local launcher stop unconfirmed: {error}", file=sys.stderr)
            if example.kind != "rollout":
                print("Owned launcher cleanup finished. Verify physical stop feedback before leaving hardware.")
                return 0
        if args.command == "check":
            report = check_report(example, mode=args.mode or "hardware")
            if args.json:
                print(json.dumps(report, indent=2))
                return 2 if report["issues"] else 0
            print(f"Stage: {report['mode']} local prerequisite check")
            for item in report["issues"]:
                print(f"Missing: {item['location']}: {item['message']}\nNext: {item['next_action']}", file=sys.stderr)
            print("Not verified: " + "; ".join(report["unverified"]))
            if report["issues"]:
                print("Result: needs attention. Correct the listed inputs and repeat this check.", file=sys.stderr)
                return 2
            if example.kind != "microduck":
                print("Result: local prerequisites passed; live robot/model readiness was not tested.")
                for action in report["next_actions"]:
                    print(f"Next: {action}")
                return 0
        if args.command == "validate":
            print(f"Valid example: {example.data['name']} (offline configuration check)")
            return 0
        action = "run" if args.command == "plan" else args.command
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        output = example.output / f"{stamp}-{action}"
        argv_list, overrides = commands(example, action, output, mode=args.mode or "hardware")
        if args.command == "plan":
            print(json.dumps({"commands": argv_list, "environment": overrides, "output": str(output)}, indent=2))
            return 0
        if example.kind in {"rollout", "snack"} and action in {"up", "run"} and not args.allow_hardware:
            raise ValueError(f"{action} needs --allow-hardware after calibration and emergency-stop checks")
        if example.kind == "rollout" and action != "down":
            deployment_values = _load_yaml(example.resolve(example.parameters["deployment"]))
            if "REPLACE_" in repr(deployment_values):
                raise ValueError("replace the REPLACE_* values in the deployment YAML before setup/up/run")
        if example.kind == "microduck" and action != "setup":
            for key in ("project_root", "inference_root", "checkpoint", "episodes", "manifest"):
                path = example.resolve(example.parameters[key])
                if action == "provision" and key == "manifest":
                    if not path.name.endswith(".local.json") or path.exists():
                        raise ValueError(
                            "provision requires a new *.local.json manifest; existing inventories are not overwritten"
                        )
                    continue
                if not path.exists():
                    raise ValueError(f"parameters.{key} does not exist: {path}")
        # Scene settings come only from the selected YAML manifest.
        env = {k: v for k, v in os.environ.items() if example.kind != "microduck" or not k.startswith("MICRODUCK_")}
        env.update(overrides)
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
        print(f"Stage: {action}" + (f" ({args.mode or 'hardware'})" if action == "setup" else ""), flush=True)
        if example.kind in {"snack", "microduck"}:
            print(f"Scene Python: {example.python}", flush=True)
        print(f"Outputs: {output}", flush=True)
        if example.kind == "snack" and action == "up":
            print(f"Service logs: {example.output / 'services'}", flush=True)
        elif not (example.kind == "snack" and action == "run"):
            print(f"Command logs: {output / 'command-0.log'}", flush=True)
        report = {"example": example.data, "command": action, "commands": argv_list, "status": "running"}
        try:
            report["revision"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            report["revision"] = os.environ.get("EMBODIRUN_REVISION")
        inputs = [example.path]
        for key in ("deployment", "config", "manifest", "episodes"):
            if key in example.parameters:
                path = example.resolve(example.parameters[key])
                if path.is_file():
                    inputs.append(path)
        snapshots = output / "inputs"
        snapshots.mkdir(mode=0o700)
        report["inputs_sha256"] = {}
        report["input_snapshots"] = {}
        for index, path in enumerate(inputs):
            content = path.read_bytes()
            snapshot = snapshots / f"{index}-{path.name}"
            snapshot.write_bytes(content)
            report["inputs_sha256"][str(path)] = hashlib.sha256(content).hexdigest()
            report["input_snapshots"][str(path)] = str(snapshot.relative_to(output))
        report_path = output / "run.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        with Supervisor(example.output, action) as supervisor:
            try:
                execute(
                    argv_list,
                    env,
                    output,
                    parallel=example.kind == "rollout" and action == "run",
                    interactive=example.kind == "snack" and action in {"up", "run"},
                    supervisor=supervisor,
                )
                report["status"] = "failed" if stop_error else "complete"
            except BaseException:
                report["status"] = "cancelled" if supervisor.cancelled.is_set() else "failed"
                if example.kind == "rollout" and action in {"up", "run"}:
                    # Killing a Host client alone would leave remote motion running.
                    print("Stopping this example's deployment after failure/interruption.", flush=True)
                    result = subprocess.run(example.host("down"), cwd=ROOT, env=env, check=False)
                    report["cleanup_returncode"] = result.returncode
                    if result.returncode:
                        print(
                            "Shutdown failed: use the physical emergency stop and inspect the robot services.",
                            file=sys.stderr,
                        )
                raise
            finally:
                report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Result: {report['status']}. Report: {report_path}", flush=True)
        if example.kind == "snack":
            if action == "setup":
                print(
                    "Next: "
                    + shlex.join(
                        [
                            "embodirun",
                            "example",
                            str(example.path),
                            "check",
                            "--mode",
                            args.mode or "hardware",
                        ]
                    ),
                    flush=True,
                )
            elif action == "dry-run":
                print("Software rehearsal finished; hardware execution and task success remain unverified.", flush=True)
                print(f"Task result: {output / 'result/status.json'}", flush=True)
                print(
                    "Next for robot preparation: "
                    + shlex.join(
                        [
                            "embodirun",
                            "example",
                            str(example.path),
                            "setup",
                            "--mode",
                            "hardware",
                        ]
                    ),
                    flush=True,
                )
            elif action == "run":
                print(f"Inspect task/physical evidence: {output / 'result/status.json'}", flush=True)
                print("Verify physical stop feedback before leaving the robot.", flush=True)
        return 2 if stop_error else 0
    except (OSError, ValueError, RuntimeError) as error:
        if report_path is not None and report["status"] == "running":
            report["status"] = "failed"
            report_path.write_text(json.dumps(report, indent=2) + "\n")
        if args.command == "check" and args.json:
            print(
                json.dumps(
                    {
                        "recipe": str(args.config.expanduser().absolute()),
                        "kind": None,
                        "mode": args.mode or "hardware",
                        "status": "needs_attention",
                        "issues": [
                            {
                                "code": "configuration_invalid",
                                "location": str(args.config),
                                "message": str(error),
                                "next_action": f"Correct the reported configuration/file error, then repeat check for {args.config}.",
                            }
                        ],
                        "verified": [],
                        "unverified": ["Configuration, dependencies, live services and physical readiness"],
                        "next_actions": [f"Correct the reported error, then repeat check for {args.config}."],
                    },
                    indent=2,
                )
            )
            return 2
        print(f"example: {error}", file=sys.stderr)
        if output is not None:
            print(f"Result: failed. Logs and report: {output}", file=sys.stderr)
            retry = ["embodirun", "example", str(args.config), args.command]
            if args.mode:
                retry.extend(["--mode", args.mode])
            if args.allow_hardware:
                retry.append("--allow-hardware")
            print(f"Next: correct the reported error, then retry {shlex.join(retry)}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(f"Result: interrupted. Inspect {output or args.config} before retrying.", file=sys.stderr)
        return 130


def interrupted(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
