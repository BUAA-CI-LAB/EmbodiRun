"""Serial terminal tasks over an already running model and hardware owner."""

from __future__ import annotations

from collections.abc import Callable


def interact(run_task: Callable[[str], dict[str, object]]) -> int:
    """Read a fresh instruction after each stopped task; propagate operational errors."""
    print("Enter an instruction; Ctrl+C cancels the current task. /quit exits, /help shows help.", flush=True)
    while True:
        try:
            instruction = input("activevln> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nInteractive navigation closed. Model and owner services remain running.", flush=True)
            return 0
        if not instruction:
            continue
        if instruction in {"/quit", "/exit"}:
            print("Interactive navigation closed. Model and owner services remain running.", flush=True)
            return 0
        if instruction == "/help":
            print(
                "Each instruction starts a new navigation history. With --execute, tasks run until model STOP "
                "or the configured chunk/time limit; otherwise each task only infers once. "
                "Ctrl+C during a task stops it and returns here after inference cleanup; "
                "Ctrl+C at this prompt exits. /quit exits. Errors exit without automatic re-arming.",
                flush=True,
            )
            continue
        if instruction.startswith("/"):
            print("Unknown command. Use /help or enter a navigation instruction.", flush=True)
            continue
        try:
            result = run_task(instruction)
        except KeyboardInterrupt:
            print("Task cancelled; ready for another instruction.", flush=True)
            continue
        print(f"Task ended: {result['reason']} ({result['chunks']} chunks). Enter the next instruction.", flush=True)


__all__ = ["interact"]
