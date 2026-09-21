#!/usr/bin/env python3
"""中文人工关节活动范围复核向导（只读，不移动机器人）。"""

from __future__ import annotations

import argparse
import json
import select
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path
from threading import Event
from typing import Any

PRIORITY = ("left_arm_shoulder_pan", "right_arm_shoulder_lift", "right_arm_elbow_flex")
DESCRIPTIONS = {
    "shoulder_pan": "肩部水平旋转",
    "shoulder_lift": "肩部抬升",
    "elbow_flex": "肘部弯曲",
    "wrist_flex": "腕部俯仰",
    "wrist_roll": "腕部旋转",
    "gripper": "夹爪开合",
}


def joint_order(calibration: dict[str, Any]) -> list[str]:
    names = [name for name in calibration if name.startswith(("left_arm_", "right_arm_"))]
    ordered = [name for name in PRIORITY if name in names]
    ordered.extend(name for name in names if name not in ordered)
    return ordered


def _description(joint: str) -> str:
    return next((text for key, text in DESCRIPTIONS.items() if joint.endswith(key)), "关节")


def _save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def run_wizard(
    args: argparse.Namespace,
    inspector: Any,
    sampler: Any,
    *,
    input_stream=None,
    output_stream=None,
    poll_input: Callable[[float], str | None] | None = None,
) -> dict[str, Any]:
    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stdout
    calibration = json.loads(args.calibration.read_text())
    if not isinstance(calibration, dict):
        raise TypeError("校准文件顶层必须是对象")
    joints = joint_order(calibration)
    session = args.output_root / time.strftime("manual-calibration-review-%Y%m%d-%H%M%S")
    suffix = 0
    while session.exists():
        suffix += 1
        session = args.output_root / f"manual-calibration-review-{time.strftime('%Y%m%d-%H%M%S')}-{suffix}"
    session.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "source": "physical_manual_calibration_review",
        "success": None,
        "old_calibration_untouched": True,
        "joints": [],
        "session_dir": str(session),
    }
    quit_requested = False

    def say(text: str) -> None:
        print(text, file=output_stream, flush=True)

    def default_poll(timeout: float) -> str | None:
        ready, _, _ = select.select([input_stream], [], [], timeout)
        if not ready:
            return None
        line = input_stream.readline()
        if line == "":
            return "eof"
        return "q" if line.strip().lower() == "q" else "enter"

    poll = poll_input or default_poll
    try:
        say(
            "只读人工复核：不会自动移动、寻找硬限位或修改旧校准文件。先保持不动；机器人自身左右，请托住手臂，不强拧；READY 后才活动当前关节。"
        )
        for joint in joints:
            side = "左侧" if joint.startswith("left_") else "右侧"
            port_index = 0 if joint.startswith("left_") else 1
            port = args.port[port_index]
            cal = calibration[joint]
            say(
                f"\n准备 {side} {joint}（{_description(joint)}），总线尾部 {Path(port).name[-20:]}，ID {cal.get('id')}，保存范围 [{cal.get('range_min')}, {cal.get('range_max')}]。"
            )
            say("Enter 开始预检；s 跳过；q 保存并退出。")
            while True:
                command = input_stream.readline()
                if command == "":
                    quit_requested = True
                    break
                command = command.strip().lower()
                if command in {"", "s", "q"}:
                    break
                say("输入无效：只能按 Enter 开始、s 跳过或 q 退出。")
            if quit_requested:
                break
            if command == "q":
                quit_requested = True
                break
            if command == "s":
                summary["joints"].append({"joint": joint, "status": "skipped"})
                continue
            stop_event = Event()
            state = {"user_ended": False, "quit": False}

            def sleep_fn(interval: float, _state=state, _stop_event=stop_event) -> None:
                command = poll(interval)
                if command in {"enter", "q", "eof"}:
                    _state["user_ended"] = True
                    _state["quit"] = command in {"q", "eof"}
                    _stop_event.set()

            def ready_callback(report: dict[str, Any], _state=state) -> None:
                _state["ready_monotonic"] = time.monotonic()
                say("READY：预检通过，当前仍为只读。按 Enter 结束当前关节，按 q 退出并保存。")

            def sample_callback(sample: dict[str, Any], report: dict[str, Any], _state=state) -> None:
                pos = sample.get("fields", {}).get("Present_Position")
                elapsed = max(
                    0.0, sample.get("monotonic", time.monotonic()) - _state.get("ready_monotonic", time.monotonic())
                )
                say(
                    f"raw_position={pos} observed=[{report.get('observed_min')}, {report.get('observed_max')}] elapsed={elapsed:.1f}s"
                )

            joint_args = argparse.Namespace(
                output_dir=session / joint,
                calibration=args.calibration,
                port=port,
                joint=joint,
                duration=60.0,
                interval=0.2,
                sdk_src=args.sdk_src,
            )
            result, status = sampler.record_joint_range(
                joint_args,
                inspector,
                stop_event=stop_event,
                sleep_fn=sleep_fn,
                ready_callback=ready_callback,
                sample_callback=sample_callback,
            )
            artifact = str((session / joint / "joint_range.json").relative_to(session))
            summary["joints"].append(
                {"joint": joint, "artifact": artifact, "status": status, "user_ended": state["user_ended"]}
            )
            if result.get("interrupted"):
                state["quit"] = True
            if status != 0 or state["quit"]:
                quit_requested = True
                if status != 0:
                    say(f"预检或读数失败（{result.get('error')}），已保存现场，向导停止，不会自动进入下一个关节。")
                break
            observed_min = result.get("observed_min")
            observed_max = result.get("observed_max")
            if observed_min == observed_max:
                say("未观察到活动（观察跨度为 0）；这不是完整范围或安全限位结论。")
            else:
                say(f"本次观察跨度：{observed_min} 到 {observed_max}；仅记录观察结果。")
            say("本次为观察结果，不自动解释为安全限位；可选备注（Enter 留空）：")
            note = input_stream.readline()
            if note == "":
                quit_requested = True
                break
            if note.strip().lower() == "q":
                quit_requested = True
                break
            summary["joints"][-1]["human_note"] = note.strip() or None
    except KeyboardInterrupt:
        quit_requested = True
        say("收到中断，已停止并保存当前会话。")
    finally:
        summary["quit_requested"] = quit_requested
        summary["success"] = None
        intended = len(joints)
        completed = sum(item.get("status") == 0 for item in summary["joints"])
        summary["status"] = (
            "observations_complete"
            if intended and completed == intended and len(summary["joints"]) == intended
            else "partial"
            if summary["joints"]
            else "empty"
        )
        _save_json(session / "session_summary.json", summary)
        if summary["status"] == "observations_complete":
            say(f"{completed}/{intended}关节记录完成；尚未验证完整标定。")
        else:
            say(f"已记录 {completed}/{intended} 个完整观察；尚未验证完整标定。")
        say(f"已保存会话：{session}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-src")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--port", action="append")
    parser.add_argument("--output-root", type=Path, default=Path("evidence/manual-calibration-review"))
    args = parser.parse_args(argv)
    if not args.sdk_src or not args.calibration or not args.port:
        parser.error("普通运行需要 --sdk-src、--calibration 和两个 --port")
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        parser.error("需要在真实交互式终端运行")
    if len(args.port) != 2 or len(set(args.port)) != 2:
        parser.error("必须提供两个不同的 --port（左侧、右侧）")
    if not args.calibration.exists():
        parser.error(f"校准文件不存在: {args.calibration}")
    import inspect_xlerobot_motors as inspector
    import record_xlerobot_joint_range as sampler

    inspector.FeetechMotorsBus, inspector.Motor, inspector.MotorNormMode = inspector.import_sdk(args.sdk_src)

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, interrupt)
    try:
        summary = run_wizard(args, inspector, sampler)
    except Exception as exc:  # noqa: BLE001
        print(f"向导失败（可能已保存部分只读读数）: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 1 if any(item.get("status", 0) not in (0, "skipped") for item in summary["joints"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
