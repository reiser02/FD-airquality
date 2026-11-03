"""Stream subprocess output without buffering a complete training run in memory."""

from __future__ import annotations

import logging
import os
import re
import signal
import subprocess
import threading
from pathlib import Path

from fd_airquality.run_logging import touch_progress

LOGGER = logging.getLogger(__name__)
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class ProcessRunner:
    def __init__(self):
        self._lock = threading.Lock()
        self._processes = set()
        self._cancelled = False

    def stream(self, command: list[str], env: dict, path: Path, label: str,
               on_line=None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as output:
            with self._lock:
                if self._cancelled:
                    raise RuntimeError("Ejecución cancelada antes de iniciar el proceso.")
                process = subprocess.Popen(
                    command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", bufsize=1,
                    start_new_session=True,
                )
                self._processes.add(process)
            try:
                for line in process.stdout:
                    clean = _ANSI.sub("", line)
                    output.write(clean)
                    output.flush()
                    if on_line is not None:
                        on_line(clean)
                    touch_progress(f"{label}: {clean.rstrip()}")
                    LOGGER.info("[%s] %s", label, clean.rstrip())
                code = process.wait()
                if code:
                    raise subprocess.CalledProcessError(code, command)
            except BaseException:
                self._terminate(process)
                raise
            finally:
                process.stdout.close()
                with self._lock:
                    self._processes.discard(process)

    @staticmethod
    def _terminate(process):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        except ProcessLookupError:
            process.wait()

    def cancel(self):
        with self._lock:
            self._cancelled = True
            processes = list(self._processes)
        for process in processes:
            self._terminate(process)
