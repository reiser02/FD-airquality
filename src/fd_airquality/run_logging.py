"""Live, queued run logs and honest progress heartbeats using the standard library."""

from __future__ import annotations

import logging
import math
import os
import queue
import sys
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path

LOGGER = logging.getLogger("fd_airquality")
_CONTEXT = ContextVar("progress_context", default="")
_ACTIVE = []


class _RunLabel(logging.Filter):
    def __init__(self, label):
        super().__init__()
        self.label = label

    def filter(self, record):
        if not hasattr(record, "run_label"):
            record.run_label = self.label
        return True


@contextmanager
def progress_context(**fields):
    label = " ".join(f"{key}={value}" for key, value in fields.items())
    token = _CONTEXT.set(f"{_CONTEXT.get()} {label}".strip())
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def report_progress(stage: str, detail: str = "") -> None:
    message = f"stage={stage} {_CONTEXT.get()} {detail}".strip()
    if _ACTIVE:
        _ACTIVE[-1].update(stage=stage, detail=message)
        for parent in _ACTIVE[:-1]:
            parent.update(detail=f"[{_ACTIVE[-1].name}] {message}")
    LOGGER.info("%s", message)


def touch_progress(detail: str) -> None:
    if "[heartbeat]" not in detail:
        for active in _ACTIVE:
            active.update(detail=detail)


def flush_run_logs() -> None:
    for active in reversed(_ACTIVE):
        active.flush()


class RunLogging:
    def __init__(self, output: Path, name: str, *, filename: str = "run.log",
                 level: str = "INFO", heartbeat_seconds: float = 60.0, stream=None):
        if not math.isfinite(heartbeat_seconds) or heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds debe ser finito y positivo.")
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError("log_level debe ser DEBUG, INFO, WARNING o ERROR.")
        self.path = Path(output) / filename
        self.name, self.level = name, level
        self.interval = heartbeat_seconds
        self.stream = sys.stderr if stream is None else stream
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._state = {"stage": "initializing", "detail": ""}
        self._started = self._updated = time.monotonic()
        self._closed = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._previous = (list(LOGGER.handlers), LOGGER.level, LOGGER.propagate, LOGGER.disabled)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s pid=%(process)d [%(run_label)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._file = logging.FileHandler(self.path, encoding="utf-8")
        self._file.setFormatter(formatter)
        self._file.setLevel(self.level)
        self._handlers = [self._file]
        if self._previous[0]:
            self._handlers.extend(self._previous[0])
        else:
            console = logging.StreamHandler(self.stream)
            console.setFormatter(formatter)
            console.setLevel(self.level)
            self._handlers.append(console)
        self._queue = queue.Queue()
        self._listener = QueueListener(self._queue, *self._handlers, respect_handler_level=True)
        self._listener.start()
        handler = QueueHandler(self._queue)
        handler.addFilter(_RunLabel(self.name))
        LOGGER.handlers = [handler]
        LOGGER.setLevel(self.level)
        LOGGER.disabled = False
        self._thread = None
        if not _ACTIVE:
            self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True, name="run-heartbeat")
            self._thread.start()
        _ACTIVE.append(self)
        LOGGER.info("Run started; log=%s", self.path)
        return self

    def update(self, **state):
        with self._lock:
            self._state.update(state)
            self._updated = time.monotonic()

    def emit_heartbeat(self):
        with self._lock:
            state = dict(self._state)
            idle = time.monotonic() - self._updated
        LOGGER.info("[heartbeat] elapsed=%.1fs since_last_event=%.1fs %s; "
                    "heartbeat != avance del entrenamiento",
                    time.monotonic() - self._started, idle,
                    " ".join(f"{key}={value}" for key, value in state.items()),
                    extra={"run_label": self.name})

    def _heartbeat_loop(self):
        while not self._stop.wait(self.interval):
            self.emit_heartbeat()

    def flush(self):
        self._queue.join()
        for handler in self._handlers:
            handler.flush()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        _ACTIVE.remove(self)
        LOGGER.handlers, level, LOGGER.propagate, LOGGER.disabled = self._previous
        LOGGER.setLevel(level)
        self._listener.stop()
        self._file.close()

    def __exit__(self, exc_type, exc, traceback):
        try:
            elapsed = time.monotonic() - self._started
            if exc_type is None:
                LOGGER.info("Run completed; elapsed=%.1fs", elapsed)
            elif issubclass(exc_type, KeyboardInterrupt):
                LOGGER.error("Run interrupted; elapsed=%.1fs", elapsed)
            else:
                LOGGER.error("Run failed; elapsed=%.1fs: %s", elapsed, exc,
                             exc_info=(exc_type, exc, traceback))
        finally:
            self.close()


def run_label(settings, mode: str) -> str:
    return (f"mode={mode} model={settings.model} adaptation={settings.adaptation} "
            f"seed={settings.seed} rank={settings.lora_rank} requested_device={settings.device} "
            f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}")
