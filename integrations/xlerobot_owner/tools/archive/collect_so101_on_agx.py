#!/usr/bin/env python3
"""Interactive, fault-tolerant dual-SO101 collection console for the AGX."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from embodirun_xlerobot_owner.leader_collection import CollectionSettings, CollectionSupervisor


def _print_status(status: dict) -> None:
    print(
        "状态={state}  硬件={connected}  从臂={armed}  记录={recording}".format(
            state=status.get("state", "unknown"),
            connected="已连接" if status.get("connected") else "等待连接",
            armed="跟随中" if status.get("armed") else "未启用",
            recording="进行中" if status.get("recording") else "未开始",
        ),
        flush=True,
    )
    if status.get("task"):
        print(f"当前任务：{status['task']}", flush=True)
    if status.get("episode_path"):
        print(f"当前目录：{status['episode_path']}", flush=True)
    if status.get("last_episode"):
        episode = status["last_episode"]
        print(
            f"上一条：{episode.get('status')}，{episode.get('frame_count')} 帧，{episode.get('episode_path')}",
            flush=True,
        )
    if status.get("last_error"):
        print(f"最近错误：{status['last_error']}", flush=True)
    if status.get("clamped_joints"):
        print("注意：以下目标已限幅：" + ", ".join(status["clamped_joints"]), flush=True)
    if status.get("stop_confirmed") is False:
        print("警告：从臂 Stop 尚未得到确认，请切断动力并排查。", flush=True)


def _menu() -> None:
    print(
        "\n[n] 开始新一条  [y] 本条成功  [f] 本条失败  [a] 中断本条\n[s] 查看状态    [x] 停止跟随  [q] 安全退出",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Override the episode output directory from the JSON config.",
    )
    args = parser.parse_args()

    settings = CollectionSettings.from_file(args.config)
    if args.output is not None:
        settings = replace(settings, output=args.output)
    supervisor = CollectionSupervisor(settings)
    supervisor.start()
    print(
        "SO101 主从采集器已启动。SSH 断开不会结束 tmux 中的本程序。\n"
        "硬件故障会中断并保存当前 episode、请求从臂停止；重连后不会自动恢复运动。",
        flush=True,
    )
    _print_status(supervisor.status())
    last_task = ""
    try:
        while True:
            _menu()
            choice = input("选择：").strip().lower()
            try:
                if choice == "n":
                    status = supervisor.status()
                    if status["state"] not in {"ready", "following"}:
                        print("硬件尚未就绪。等待状态变为 ready 后，请重新按 n。", flush=True)
                        _print_status(status)
                        continue
                    if last_task:
                        entered_task = input(f"本条任务名（直接回车沿用“{last_task}”）：").strip()
                        task = entered_task or last_task
                    else:
                        task = input("本条任务名：").strip()
                    if not task:
                        print("任务名不能为空。", flush=True)
                        continue
                    last_task = task
                    confirm = (
                        input("确认两只从臂周围无人和障碍物，主臂保持静止；按回车开始，输入 c 取消：").strip().lower()
                    )
                    if confirm == "c":
                        print("已取消。", flush=True)
                        continue
                    result = supervisor.command("begin", task=task, timeout_s=60.0)
                    print("已开始记录；现在移动主臂，从臂会直接跟随。", flush=True)
                    _print_status(result)
                elif choice == "y":
                    result = supervisor.command("finish", success=True)
                    print("本条已标记成功并落盘；可按 n 开始下一条。", flush=True)
                    _print_status(supervisor.status())
                    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
                elif choice == "f":
                    result = supervisor.command("finish", success=False)
                    print("本条已标记失败并保留；可按 n 开始下一条。", flush=True)
                    _print_status(supervisor.status())
                    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
                elif choice == "a":
                    result = supervisor.command("abort")
                    print("本条已中止并保留。", flush=True)
                    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
                elif choice == "s":
                    _print_status(supervisor.status())
                elif choice == "x":
                    result = supervisor.command("stop")
                    print("已请求停止跟随。", flush=True)
                    _print_status(result)
                elif choice == "q":
                    if supervisor.status().get("recording"):
                        supervisor.command("abort")
                    break
                elif choice:
                    print("未知选项。", flush=True)
            except (RuntimeError, TimeoutError, ValueError) as exc:
                print(f"操作未完成：{type(exc).__name__}: {exc}", flush=True)
                _print_status(supervisor.status())
    except (EOFError, KeyboardInterrupt):
        print("\n终端输入结束，正在安全停止……", flush=True)
    finally:
        status = supervisor.shutdown()
        _print_status(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
