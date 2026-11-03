import logging
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from io import StringIO

import pytest

from fd_airquality.processes import ProcessRunner
from fd_airquality.run_logging import RunLogging, progress_context, report_progress
from fd_airquality.settings import RunSettings


def test_log_is_readable_before_completion_and_restores_logger(tmp_path):
    logger = logging.getLogger("fd_airquality")
    original = (list(logger.handlers), logger.level, logger.propagate, logger.disabled)
    stream = StringIO()
    with RunLogging(tmp_path, "test", stream=stream) as run:
        with progress_context(model="mlp", station="station-a", round=1):
            report_progress("train", "epoch=1/2 batches=3/4")
        run.emit_heartbeat()
        run.flush()
        live = run.path.read_text()
        assert "station=station-a" in live
        assert "epoch=1/2" in live
        assert "[heartbeat]" in live and "since_last_event=" in live
        assert "heartbeat != avance" in live
        assert "Run completed" not in live
    assert "Run completed" in run.path.read_text()
    assert "station=station-a" in stream.getvalue()
    assert (logger.handlers, logger.level, logger.propagate, logger.disabled) == original
    run.close()


@pytest.mark.parametrize("exception,marker", [(RuntimeError("broken"), "Run failed"),
                                               (KeyboardInterrupt(), "Run interrupted")])
def test_logging_preserves_errors_and_interruptions(tmp_path, exception, marker):
    with pytest.raises(type(exception)), RunLogging(tmp_path, "test", stream=StringIO()):
        raise exception
    text = (tmp_path / "run.log").read_text()
    assert marker in text
    if isinstance(exception, RuntimeError):
        assert "RuntimeError: broken" in text


def test_nested_runs_do_not_duplicate_records_or_close_parent_handlers(tmp_path):
    with RunLogging(tmp_path, "outer", filename="benchmark.log", stream=StringIO()) as parent:
        for index in range(2):
            with RunLogging(tmp_path / str(index), "inner"):
                report_progress("train", f"unique-event-{index}")
        report_progress("finalizing", "parent-still-open")
        parent.flush()
    text = (tmp_path / "benchmark.log").read_text()
    assert text.count("unique-event-0") == text.count("unique-event-1") == 1
    assert "[inner] stage=train  unique-event-0" in text
    assert text.count("parent-still-open") == 1
    assert (tmp_path / "0" / "run.log").read_text().count("unique-event-0") == 1


def test_nested_logs_share_one_heartbeat_monitor_and_keep_parent_state_current(tmp_path):
    with RunLogging(tmp_path, "outer", filename="benchmark.log", stream=StringIO()) as parent:
        parent.update(stage="local", completed=1, total=3)
        with RunLogging(tmp_path / "child", "candidate") as child:
            assert parent._thread is not None and child._thread is None
            report_progress("train", "epoch=2/4")
            parent.emit_heartbeat()
            child.flush()
            parent.flush()
    text = (tmp_path / "benchmark.log").read_text()
    heartbeat, = [line for line in text.splitlines() if "[heartbeat]" in line]
    assert "[outer]" in heartbeat and "stage=local" in heartbeat
    assert "completed=1" in heartbeat and "epoch=2/4" in heartbeat


def test_concurrent_records_have_a_single_file_writer(tmp_path):
    with RunLogging(tmp_path, "parallel", stream=StringIO()) as log:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda i: report_progress("task", f"unique={i:04d}"), range(100)))
        log.flush()
    lines = [line for line in log.path.read_text().splitlines() if "unique=" in line]
    assert len(lines) == 100
    assert len({line.split("unique=")[1] for line in lines}) == 100


def test_process_output_is_flushed_while_child_is_still_running(tmp_path):
    runner = ProcessRunner()
    path = tmp_path / "child.log"

    def received(line):
        assert "ready" in path.read_text()
        assert len(runner._processes) == 1
        raise RuntimeError("stop after observing live output")

    with pytest.raises(RuntimeError, match="live output"):
        runner.stream([sys.executable, "-u", "-c", "import time; print('ready'); time.sleep(30)"],
                      os.environ.copy(), path, "child", on_line=received)
    assert not runner._processes


def test_process_stream_preserves_stderr_and_strips_ansi(tmp_path):
    path = tmp_path / "child.log"
    ProcessRunner().stream(
        [sys.executable, "-u", "-c", "import sys; print('\\x1b[31mhola\\x1b[0m'); print('error', file=sys.stderr)"],
        os.environ.copy(), path, "child",
    )
    assert path.read_text() == "hola\nerror\n"


def test_cancelling_active_processes_unblocks_stream_readers(tmp_path):
    runner = ProcessRunner()
    ready = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            runner.stream,
            [sys.executable, "-u", "-c", "import time; print('ready'); time.sleep(30)"],
            os.environ.copy(), tmp_path / "child.log", "child", lambda _: ready.set(),
        )
        assert ready.wait(timeout=10)
        runner.cancel()
        with pytest.raises(subprocess.CalledProcessError):
            future.result(timeout=10)
    assert not runner._processes
    with pytest.raises(RuntimeError, match="cancelada"):
        runner.stream([sys.executable, "-c", "pass"], os.environ.copy(), tmp_path / "other.log", "child")


@pytest.mark.parametrize("kwargs", [{"heartbeat_seconds": 0}, {"heartbeat_seconds": float("inf")},
                                   {"log_level": "invalid"}])
def test_log_settings_are_validated(kwargs):
    with pytest.raises(ValueError):
        RunSettings(**kwargs)
