"""RPent entrypoint reusing Go2's tested camera, policy and motion-lease clients."""

from __future__ import annotations

import argparse
import io
import json
import os
import signal
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from embodirun.application.navigation.chunk_session import ChunkNavigationSession
from embodirun.bindings.unitree.go2.activevln.cli import NavigationConfig
from embodirun.bindings.unitree.go2.activevln.policy import ActiveVLNPolicy, check_capabilities
from embodirun.model_services.backends.embodiinfer.compat import connect_client
from embodirun.robots.unitree.go2.camera import Go2CameraClient
from embodirun.robots.unitree.go2.client import Go2ControlClient

from .skill import Feedback, NavigationSkill


@contextmanager
def cancellation_scope(toolkit):
    """Cancel at a complete-chunk boundary before releasing planner resources."""
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def interrupt(signum, _frame):
        # The RPent tool runs in another thread. Interrupting only the main
        # thread would otherwise leave that worker executing its remaining chunks.
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        toolkit.cancel_active_and_wait()
        raise SystemExit(128 + signum)

    try:
        for sig in previous:
            signal.signal(sig, interrupt)
        yield
    finally:
        try:
            toolkit.cancel_active_and_wait()
        finally:
            try:
                toolkit.close()
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)


class Go2Backend:
    """Bound each skill by existing complete-chunk leases; no new hardware owner."""

    def __init__(self, camera, base, config, *, execute=False, event_sink=None):
        self.camera, self.base, self.config = camera, base, config
        self.execute, self.event_sink = execute, event_sink
        self.reader = ChunkNavigationSession(None, camera, base, config=config)

    def observe(self):
        from PIL import Image

        observation = self.reader.capture_idle("rpent-inspect")
        buffer = io.BytesIO()
        Image.open(io.BytesIO(observation.latest_rgb.data)).convert("RGB").save(buffer, format="PNG")
        return Feedback(image=buffer.getvalue())

    def run(self, policy, instruction, episode_id, max_chunks, *, terminal_stop):
        if not self.execute:
            raise RuntimeError("motion disabled: restart with --execute after arming the existing owner")
        self.base.preflight(require_release=True)
        session = ChunkNavigationSession(
            policy,
            self.camera,
            self.base,
            config=replace(self.config, max_chunks=min(max_chunks, self.config.max_chunks)),
            event_sink=self.event_sink,
        )
        result = session.run(instruction, execute=True, episode_id=episode_id, keep_motion_ready=True)
        # All motion leases have finished/released before the slow planner sees
        # a new frame. Errors still use ChunkNavigationSession's disarming path.
        reason = "local_stop" if result["reason"] == "stop" else str(result["reason"])
        return replace(self.observe(), reason=reason)

    def finish(self, *, success):
        # There is no live owned lease between tools. An unscoped stop here
        # could interfere with a different controller and is unnecessary.
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--planner-model", required=True)
    parser.add_argument("--execute", action="store_true", help="enable motion on an already armed Go2 owner")
    parser.add_argument("--max-tool-calls", type=int, default=80)
    parser.add_argument("--max-runtime-s", type=int, default=1800)
    parser.add_argument("--profile", choices=("event", "legacy"), default="event")
    args = parser.parse_args()
    config = NavigationConfig.load(args.config)
    config.require_endpoints()
    args.output.mkdir(parents=True, exist_ok=False)
    client = connect_client(
        config.inference_url, os.environ.get(config.inference_token_env), timeout_s=config.inference_timeout_s
    )
    check_capabilities(client)
    camera = Go2CameraClient(
        config.camera_url,
        token=os.environ.get(config.camera_token_env),
        timeout_s=config.camera_timeout_s,
        require_depth=False,
    )
    base = Go2ControlClient(
        config.control_url, token=os.environ.get(config.control_token_env), timeout_s=config.control_timeout_s
    )

    def record(event):
        with (args.output / "execution.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")

    backend = Go2Backend(camera, base, config.session, execute=args.execute, event_sink=record)
    skill = NavigationSkill(
        backend,
        lambda: ActiveVLNPolicy(client, robot_id=config.robot_id, event_sink=record),
        max_chunks=config.session.max_chunks,
        max_runtime_s=args.max_runtime_s,
    )
    from .planner import solve
    from .toolkit import NavigationToolkit

    toolkit = NavigationToolkit(skill, args.output, max_tool_calls=args.max_tool_calls, profile=args.profile)
    instruction = args.instruction
    if not args.execute:
        instruction += (
            "\nMotion is disabled. Observe, then finish failure (unexecuted) with a proposed plan. Do not navigate."
        )
    with cancellation_scope(toolkit):
        result = solve(toolkit, instruction, args.output, model=args.planner_model, timeout_s=args.max_runtime_s)
        summary = {
            "finish": toolkit.finish_result,
            "error": result.error,
            "stats": result.stats,
            "model_calls": skill.model_calls,
            "subgoals": skill.subgoals,
            "motion_enabled": args.execute,
            "profile": args.profile,
        }
        (args.output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, ensure_ascii=False))
        return 1 if result.error else 0


if __name__ == "__main__":
    raise SystemExit(main())
