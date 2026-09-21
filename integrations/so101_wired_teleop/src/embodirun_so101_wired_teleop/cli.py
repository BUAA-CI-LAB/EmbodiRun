"""Command line entry points for wired SO-101 teleoperation.

Three programs share one configuration file:

``embodirun-so101-leader``
    Read one leader arm and broadcast it. Run one of these per leader.
``embodirun-so101-follower``
    Receive targets and drive one follower arm. Run one of these per follower.
``embodirun-so101-collect``
    Operator console: pick a leader and a follower, then start and stop episodes.

The console talks to remote followers through the system ``ssh`` client rather
than an embedded SSH library, so a follower host only needs an entry in the
operator's SSH configuration. A follower without ``ssh_host`` runs locally.
"""

from __future__ import annotations

import argparse
import json
import select
import shlex
import signal
import subprocess
import sys
import termios
import time
import tty
from pathlib import Path

from .config import ConfigError, FollowerConfig, TeleopConfig, load_config
from .follower import FollowerNode
from .leader import LeaderBroadcaster

MIN_FREE_KB = 300 * 1024
FRAME_STALE_S = 3.0
EPISODE_TIMEOUT_S = 30.0
POLL_INTERVAL_S = 0.2


def _load(path: str, *, allow_placeholders: bool) -> TeleopConfig:
    try:
        config = load_config(path)
        if not allow_placeholders:
            config.require_resolved()
        return config
    except ConfigError as error:
        raise SystemExit(f"configuration error: {error}") from error


def _select(config: TeleopConfig, kind: str, identifier: str) -> object:
    items = config.leaders if kind == "leader" else config.followers
    for item in items:
        if item.id == identifier:
            return item
    known = ", ".join(item.id for item in items) or "(none)"
    raise SystemExit(f"unknown {kind} {identifier!r}; configured: {known}")


# --------------------------------------------------------------------------- #
# leader / follower
# --------------------------------------------------------------------------- #


def leader_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Broadcast one wired SO-101 leader arm.")
    parser.add_argument("--config", required=True, help="teleoperation YAML file")
    parser.add_argument("--leader", required=True, help="leader id from the configuration")
    parser.add_argument(
        "--network-test",
        action="store_true",
        help="send diagnostic packets that cannot command an arm",
    )
    parser.add_argument("--seconds", type=float, default=None, help="stop after N seconds")
    args = parser.parse_args(argv)

    config = _load(args.config, allow_placeholders=args.network_test)
    leader = _select(config, "leader", args.leader)
    broadcaster = LeaderBroadcaster(config, leader)
    if args.network_test:
        return broadcaster.network_test()
    return broadcaster.run(stop_after=args.seconds)


def follower_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Drive one wired SO-101 follower arm.")
    parser.add_argument("--config", required=True, help="teleoperation YAML file")
    parser.add_argument("--follower", required=True, help="follower id from the configuration")
    parser.add_argument(
        "--network-test",
        action="store_true",
        help="wait for the leader's packets without touching the arm",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="skip the step and lead limits (bench debugging only)",
    )
    args = parser.parse_args(argv)

    config = _load(args.config, allow_placeholders=args.network_test)
    follower = _select(config, "follower", args.follower)
    node = FollowerNode(config, follower)
    if args.network_test:
        return node.network_test()
    return node.run(direct=args.direct)


# --------------------------------------------------------------------------- #
# operator console
# --------------------------------------------------------------------------- #


class _Endpoint:
    """Run a follower node locally or on its host through the ssh client."""

    def __init__(self, follower: FollowerConfig) -> None:
        self.follower = follower
        self.remote = follower.ssh_host
        self.pid: int | None = None

    def _shell(self, command: str) -> str:
        if not self.remote:
            return subprocess.run(  # noqa: S603 - fixed argv, no shell
                ["/bin/sh", "-c", command],
                capture_output=True,
                text=True,
                check=False,
            ).stdout
        return subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["ssh", "-o", "BatchMode=yes", self.remote, command],
            capture_output=True,
            text=True,
            check=False,
        ).stdout

    def processes(self) -> list[int]:
        out = self._shell("ps -eo pid=,args=")
        matches = []
        for line in out.splitlines():
            try:
                pid, command = line.strip().split(maxsplit=1)
                args = shlex.split(command)
            except ValueError:
                continue
            if not any(Path(arg).name == "embodirun-so101-follower" for arg in args[:2]):
                continue
            selected = any(
                arg == f"--follower={self.follower.id}"
                or (arg == "--follower" and args[index + 1 : index + 2] == [self.follower.id])
                for index, arg in enumerate(args)
            )
            if selected and pid.isdigit():
                matches.append(int(pid))
        return matches

    def start(self, config_path: str) -> int:
        existing = self.processes()
        if len(existing) > 1:
            raise RuntimeError(f"follower {self.follower.id} has several nodes running")
        if existing:
            self.pid = existing[0]
            return self.pid
        log = f"/tmp/embodirun-so101-{self.follower.id}.log"
        command = (
            f"nohup embodirun-so101-follower --config {shlex.quote(config_path)} "
            f"--follower {shlex.quote(self.follower.id)} > {shlex.quote(log)} 2>&1 < /dev/null & echo $!"
        )
        out = self._shell(command).strip()
        self.pid = int(out) if out.isdigit() else None
        if self.pid is None:
            raise RuntimeError(f"could not start follower {self.follower.id}; see {log}")
        return self.pid

    def signal(self, number: int) -> None:
        if self.pid is None:
            raise RuntimeError(f"follower {self.follower.id} is not running")
        if self.pid not in self.processes():
            raise RuntimeError(f"follower {self.follower.id} exited; restart the console")
        self._shell(f"kill -USR{number} {self.pid}")

    def free_kb(self) -> int:
        out = self._shell("df -Pk . | tail -1").split()
        return int(out[3]) if len(out) >= 4 else 0

    def latest_episode(self) -> str | None:
        root = self.follower.record_dir
        if not root:
            return None
        out = self._shell(f"find {shlex.quote(root)} -maxdepth 1 -type d -name 'episode-*' | sort | tail -1").strip()
        return out or None

    def closed(self, episode: str | None) -> bool:
        if not episode:
            return True
        out = self._shell(f"tail -1 {shlex.quote(episode + '/frames.jsonl')}").strip()
        try:
            return json.loads(out).get("event") == "stop"
        except ValueError:
            return False


def _choose(prompt: str, options: dict[str, str]) -> str:
    for key, label in options.items():
        print(f"  {key}. {label}")
    while True:
        value = input(prompt).strip().lower()
        if value in options:
            return value
        print("请选择列出的选项。")


def collect_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record episodes from a wired follower.")
    parser.add_argument("--config", required=True, help="teleoperation YAML file")
    parser.add_argument("--leader", default=None, help="skip the leader prompt")
    parser.add_argument("--follower", default=None, help="skip the follower prompt")
    parser.add_argument("--task", default="", help="task text stored in decision.json")
    args = parser.parse_args(argv)

    config = _load(args.config, allow_placeholders=False)
    leader_id = args.leader or _choose("选择主臂：", {leader.id: leader.id for leader in config.leaders})
    leader = _select(config, "leader", leader_id)
    follower_id = args.follower or _choose("选择从臂：", {f.id: f"{f.id}  ->  {f.leader}" for f in config.followers})
    follower = _select(config, "follower", follower_id)
    if not follower.record_dir:
        raise SystemExit("the selected follower needs record_dir for episode collection")
    if follower.leader != leader.id:
        print(f"注意：从臂 {follower.id} 配置跟随的是 {follower.leader}，不是 {leader.id}。")

    endpoint = _Endpoint(follower)
    endpoint.start(args.config)
    print(f"从臂 {follower.id} 已就绪（leader={follower.leader}）")
    print("S 开始 | E 保存结束 | D 作废结束 | Q 保存并退出")

    active: str | None = None
    saved = 0
    started_at = 0.0
    original = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        while True:
            if not select.select([sys.stdin], [], [], POLL_INTERVAL_S)[0]:
                if active and time.monotonic() - started_at > FRAME_STALE_S + 1:
                    if endpoint.free_kb() < MIN_FREE_KB:
                        _finish(endpoint, active, discard=True)
                        active = None
                        print("\n磁盘不足，已结束录制。")
                        continue
                    started_at = time.monotonic()
                continue
            key = sys.stdin.read(1).lower()
            if key == "s":
                if active:
                    print("正在录制，请先 E 保存或 D 作废。")
                    continue
                previous = endpoint.latest_episode()
                endpoint.signal(1)
                deadline = time.monotonic() + EPISODE_TIMEOUT_S
                while time.monotonic() < deadline:
                    candidate = endpoint.latest_episode()
                    if candidate and candidate != previous:
                        active = candidate
                        started_at = time.monotonic()
                        break
                    time.sleep(POLL_INTERVAL_S)
                if not active:
                    print("开始录制未确认。")
                else:
                    print(f"\n● 正在录制：{active.rsplit('/', 1)[-1]}")
            elif key in ("e", "d") and active:
                path = _finish(
                    endpoint,
                    active,
                    discard=key == "d",
                    task=args.task,
                    leader=leader.id,
                    follower=follower.id,
                )
                active = None
                saved += int(key == "e")
                print(f"已{'保存' if key == 'e' else '作废'}；本次保存 {saved} 条。\n{path}")
            elif key in ("q", "\x03"):
                if active:
                    _finish(endpoint, active, task=args.task, leader=leader.id, follower=follower.id)
                print("采集已退出；从臂进程继续运行。")
                return 0
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original)
    return 0


def _finish(
    endpoint: _Endpoint,
    episode: str,
    *,
    discard: bool = False,
    task: str = "",
    leader: str = "",
    follower: str = "",
) -> str:
    endpoint.signal(2)
    deadline = time.monotonic() + EPISODE_TIMEOUT_S
    while not endpoint.closed(episode):
        if time.monotonic() > deadline:
            raise RuntimeError("结束未确认，请勿开始下一条")
        time.sleep(POLL_INTERVAL_S)
    endpoint._shell(  # noqa: SLF001 - the decision file is part of the same contract
        "cat > {}/decision.json <<'JSON'\n{}\nJSON".format(
            shlex.quote(episode),
            json.dumps(
                {
                    "decision": "discard" if discard else "keep",
                    "time": time.time(),
                    "task": task,
                    "leader": leader,
                    "follower": follower,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
    )
    return episode


__all__ = ["collect_main", "follower_main", "leader_main", "signal"]
