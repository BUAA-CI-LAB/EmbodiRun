import sys
import threading
import time
import types

from embodirun_xlerobot_owner import hardware


class _Encoded:
    def __init__(self, payload):
        self.payload = payload

    def tobytes(self):
        return self.payload

    def __bytes__(self):
        return self.payload


class _Capture:
    def __init__(self, reads=(), *, opened=True):
        self.reads = list(reads)
        self.opened = opened
        self.release_count = 0
        self.released = threading.Event()

    def isOpened(self):
        return self.opened

    def set(self, _property, _value):
        return True

    def read(self):
        if self.reads:
            result = self.reads.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        return True, "steady-frame"

    def release(self):
        self.release_count += 1
        self.released.set()


class _FakeCV2(types.ModuleType):
    CAP_V4L2 = 200
    CAP_PROP_FOURCC = 1
    CAP_PROP_FRAME_WIDTH = 2
    CAP_PROP_FRAME_HEIGHT = 3
    CAP_PROP_FPS = 4
    CAP_PROP_BUFFERSIZE = 5

    def __init__(self, captures):
        super().__init__("cv2")
        self.captures = list(captures)
        self.open_calls = []

    def VideoCapture(self, path, *args):
        self.open_calls.append((path, args))
        if not self.captures:
            raise AssertionError("test opened an unexpected additional camera")
        return self.captures.pop(0)

    @staticmethod
    def VideoWriter_fourcc(*_characters):
        return 1

    @staticmethod
    def imencode(_extension, _frame):
        return True, _Encoded(b"new-jpeg")


class _BackoffEvent(threading.Event):
    def __init__(self, *, gate_backoff=False):
        super().__init__()
        self.backoff_started = threading.Event()
        self.allow_backoff = threading.Event()
        self.gate_backoff = gate_backoff

    def wait(self, timeout=None):
        if timeout is not None and timeout >= 0.4:
            self.backoff_started.set()
            if self.gate_backoff:
                while not self.is_set() and not self.allow_backoff.is_set():
                    time.sleep(0.001)
                return self.is_set()
        return super().wait(timeout)


def _wait_until(predicate, timeout=1.5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate()


def _worker(path="/dev/video-test"):
    return hardware._CameraWorker("front", path, stale_after_s=60)


def test_read_failure_releases_and_reopens_same_path_with_fresh_payload(monkeypatch):
    first = _Capture(reads=[(False, None)])
    second = _Capture(reads=[(True, "new-frame")])
    cv2 = _FakeCV2([first, second])
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    worker = _worker()
    stop = _BackoffEvent(gate_backoff=True)
    worker._stop = stop
    old_timestamp = time.time_ns()
    with worker._lock:
        worker._frame = b"old-jpeg"
        worker._timestamp_ns = old_timestamp

    worker.start()
    try:
        _wait_until(first.released.is_set)
        _wait_until(stop.backoff_started.is_set)
        assert [call[0] for call in cv2.open_calls] == ["/dev/video-test"]
        old_snapshot = worker.snapshot()
        assert old_snapshot[0] == b"old-jpeg"
        assert old_snapshot[1] == old_timestamp
        assert old_snapshot[4] is True

        stop.allow_backoff.set()
        _wait_until(lambda: worker.snapshot()[4] and worker.snapshot()[0] == b"new-jpeg")
        payload, timestamp_ns, error, _error_timestamp_ns, fresh = worker.snapshot()
        assert payload == b"new-jpeg"
        assert timestamp_ns > old_timestamp
        assert error is None
        assert fresh is True
        assert [call[0] for call in cv2.open_calls[:2]] == ["/dev/video-test"] * 2
        assert first.release_count == 1
    finally:
        worker.close(timeout_s=1)


def test_open_failure_releases_capture_retries_and_then_succeeds(monkeypatch):
    first = _Capture(opened=False)
    second = _Capture(reads=[(True, "new-frame")])
    cv2 = _FakeCV2([first, second])
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    worker = _worker()
    worker.start()
    try:
        _wait_until(lambda: len(cv2.open_calls) == 2, timeout=2)
        _wait_until(lambda: worker.snapshot()[0] == b"new-jpeg")
        assert [call[0] for call in cv2.open_calls[:2]] == ["/dev/video-test"] * 2
        assert first.release_count == 1
        assert worker.snapshot()[4] is True
    finally:
        worker.close(timeout_s=1)


def test_close_interrupts_reconnect_backoff_without_opening_another_device(monkeypatch):
    first = _Capture(reads=[(False, None)])
    cv2 = _FakeCV2([first])
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    worker = _worker()
    stop = _BackoffEvent()
    worker._stop = stop
    worker.start()
    _wait_until(stop.backoff_started.is_set)
    try:
        started = time.monotonic()
        assert worker.close(timeout_s=0.4)
        assert time.monotonic() - started < 0.3
        assert [call[0] for call in cv2.open_calls] == ["/dev/video-test"]
        assert first.release_count == 1
    finally:
        worker.close(timeout_s=1)
