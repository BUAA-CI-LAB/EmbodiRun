"""Optional camera viewer isolated from simulator service threads."""

from __future__ import annotations

import argparse
import os
import select
import struct
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from queue import Empty, Queue
from typing import Any, BinaryIO

_MAX_FRAME_BYTES = 16 * 1024 * 1024
_START_TIMEOUT_S = 5.0
_STOP_TIMEOUT_S = 2.0


class CameraViewer:
    """Send rendered policy cameras to a dedicated local GUI process."""

    def __init__(self, window_name: str) -> None:
        self.window_name = window_name
        self._process = subprocess.Popen(
            (
                sys.executable,
                "-c",
                ("from embodirun.simulators.viewer import main; raise SystemExit(main())"),
                "--window",
                self.window_name,
            ),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        if self._process.stdout is None or self._process.stdin is None:
            self.close()
            raise RuntimeError("simulator viewer could not create its communication pipes")
        ready, _, _ = select.select((self._process.stdout,), (), (), _START_TIMEOUT_S)
        message = self._process.stdout.readline().decode("utf-8", errors="replace") if ready else ""
        if message.strip() != "READY":
            self.close()
            detail = message.strip() or "the GUI process did not respond"
            raise RuntimeError(f"simulator viewer could not start: {detail}")

    def show(self, frames: Mapping[str, Any]) -> None:
        if self._process.poll() is not None or self._process.stdin is None:
            return
        payload = _encode_frames(frames)
        try:
            self._process.stdin.write(struct.pack(">I", len(payload)))
            self._process.stdin.write(payload)
            self._process.stdin.flush()
        except (BrokenPipeError, OSError):
            self.close()

    def close(self) -> None:
        stream = getattr(self._process, "stdin", None)
        if stream is not None and not stream.closed:
            stream.close()
        try:
            self._process.wait(timeout=_STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            self._process.wait(timeout=_STOP_TIMEOUT_S)
        output = getattr(self._process, "stdout", None)
        if output is not None and not output.closed:
            output.close()


def _encode_frames(frames: Mapping[str, Any]) -> bytes:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError("simulator viewer requires NumPy and OpenCV") from error

    images = []
    for name, value in frames.items():
        image = np.asarray(value, dtype=np.uint8)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"simulator viewer frame {name!r} must be an HWC RGB image")
        images.append(image)
    if not images:
        raise ValueError("simulator viewer requires at least one camera frame")

    target_height = min(image.shape[0] for image in images)
    resized = [
        cv2.resize(
            image,
            (
                max(1, round(image.shape[1] * target_height / image.shape[0])),
                target_height,
            ),
            interpolation=cv2.INTER_AREA,
        )
        for image in images
    ]
    encoded, payload = cv2.imencode(
        ".jpg",
        cv2.cvtColor(np.hstack(resized), cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), 90],
    )
    if not encoded:
        raise RuntimeError("simulator viewer could not encode its camera frames")
    result = payload.tobytes()
    if len(result) > _MAX_FRAME_BYTES:
        raise RuntimeError("simulator viewer frame is too large")
    return result


def _read_frames(stream: BinaryIO, frames: Queue[bytes | None], stopped: threading.Event) -> None:
    try:
        while True:
            header = _read_exact(stream, 4, stopped)
            if header is None:
                break
            size = struct.unpack(">I", header)[0]
            if not 0 < size <= _MAX_FRAME_BYTES:
                break
            payload = _read_exact(stream, size, stopped)
            if payload is None:
                break
            while True:
                try:
                    frames.get_nowait()
                except Empty:
                    break
            frames.put(payload)
    finally:
        # A closed window no longer drains the queue; never block shutdown.
        while not frames.empty():
            try:
                frames.get_nowait()
            except Empty:
                break
        frames.put_nowait(None)


def _read_exact(stream: BinaryIO, size: int, stopped: threading.Event) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        if stopped.is_set():
            return None
        ready, _, _ = select.select((stream,), (), (), 0.1)
        if not ready:
            continue
        # Avoid a daemon thread holding stdin's buffered lock at interpreter exit.
        chunk = os.read(stream.fileno(), size - len(chunks))
        if not chunk:
            return None
        chunks.extend(chunk)
    return bytes(chunks)


def _run_window(window_name: str) -> int:
    try:
        import tkinter as tk
        from io import BytesIO

        from PIL import Image, ImageTk

        root = tk.Tk()
    except Exception as error:  # noqa: BLE001 - report GUI startup failures to parent
        print(f"{type(error).__name__}: {error}", flush=True)
        return 1

    root.title(window_name)
    label = tk.Label(root)
    label.pack()
    frames: Queue[bytes | None] = Queue(maxsize=1)
    stopped = threading.Event()
    reader = threading.Thread(
        target=_read_frames,
        args=(sys.stdin.buffer, frames, stopped),
    )
    reader.start()

    def update() -> None:
        try:
            payload = frames.get_nowait()
        except Empty:
            root.after(16, update)
            return
        if payload is None:
            root.destroy()
            return
        image = Image.open(BytesIO(payload))
        photo = ImageTk.PhotoImage(image)
        label.configure(image=photo)
        label.image = photo
        root.after(16, update)

    root.bind("q", lambda _event: root.destroy())
    root.bind("<Escape>", lambda _event: root.destroy())
    root.after(16, update)
    print("READY", flush=True)
    try:
        root.mainloop()
    finally:
        stopped.set()
        reader.join(timeout=_STOP_TIMEOUT_S)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", required=True)
    args = parser.parse_args(argv)
    return _run_window(args.window)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CameraViewer"]
