import json
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from io import StringIO
from pathlib import Path

import pytest

from fd_airquality.cli import build_benchmark_matrix, run_benchmark
from fd_airquality.execution import _candidate_process, execute_plan, usable_gpu_slots
from fd_airquality.processes import ProcessRunner
from fd_airquality.run_logging import RunLogging
from fd_airquality.settings import RunSettings


def candidate(model, mode):
    return {"model": model, "mode": mode, "adaptation": "none" if mode == "zero-shot" else "auto",
            "strategy": "fedavg", "seed": 42, "lora_rank": 4}


def result(root, index, device="cuda"):
    path = root / f"result-{index}"
    path.mkdir()
    (path / "run.json").write_text(json.dumps({"settings": {"device": device}}))
    return path


def test_matrix_groups_modes_in_requested_order():
    modes = ["local", "centralized", "federated", "zero-shot"]
    matrix = build_benchmark_matrix(["mlp", "ttm-r3"], modes, ["lora", "ffa-lora"])
    assert list(dict.fromkeys(case["mode"] for case in matrix)) == modes
    assert [modes.index(case["mode"]) for case in matrix] == sorted(modes.index(case["mode"]) for case in matrix)


def test_parallel_candidates_have_exclusive_gpu_slots_and_mode_barriers(tmp_path, monkeypatch):
    monkeypatch.setattr("fd_airquality.execution.usable_gpu_slots", lambda _: ["3", "7"])
    modes = ["local", "centralized", "federated", "zero-shot"]
    plan = [candidate(model, mode) for mode in modes for model in ("mlp", "autoencoder")]
    active, seen, barrier = {}, [], threading.Barrier(2)
    lock = threading.Lock()
    original_visible = os.environ.get("CUDA_VISIBLE_DEVICES")

    def parallel(settings, mode, gpu, log_path, runner):
        with lock:
            assert gpu not in active
            assert not active or set(active.values()) == {mode}
            active[gpu] = mode
            seen.append((mode, gpu))
            assert json.loads((tmp_path / "benchmark.json").read_text())
        barrier.wait(timeout=10)
        with lock:
            del active[gpu]
        return result(tmp_path, f"{mode}-{settings.model}")

    launches = []

    def launch(settings, dry_run):
        assert not active
        launches.append((settings.device, dry_run))
        seen.append(("federated", "Ray"))
        return result(tmp_path, f"federated-{settings.model}")

    monkeypatch.setattr("fd_airquality.execution._candidate_process", parallel)
    with RunLogging(tmp_path, "test", stream=StringIO()) as log:
        paths = execute_plan(RunSettings(device="cuda"), plan, tmp_path,
                             max_parallel_models=None, baseline=lambda *_: pytest.fail("GPU case ran inline"),
                             launch=launch, progress_log=log)
    assert len(paths) == 8
    assert [mode for mode, _ in seen] == [mode for mode in modes for _ in range(2)]
    assert launches == [("cuda", False)] * 2
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == original_visible
    rows = json.loads((tmp_path / "benchmark.json").read_text())
    assert all(row["status"] == "completed" for row in rows)
    assert {row["gpu"] for row in rows if row["mode"] != "federated"} == {"3", "7"}


def test_concurrency_cap_limits_non_federated_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr("fd_airquality.execution.usable_gpu_slots", lambda _: ["0", "1"])
    seen = []

    def parallel(settings, mode, gpu, *args):
        seen.append(gpu)
        return result(tmp_path, len(seen))

    monkeypatch.setattr("fd_airquality.execution._candidate_process", parallel)
    plan = [candidate(name, "local") for name in ("mlp", "autoencoder", "darts-nlinear")]
    with RunLogging(tmp_path, "test", stream=StringIO()) as log:
        execute_plan(RunSettings(device="cuda"), plan, tmp_path,
                     max_parallel_models=1, baseline=None, launch=None, progress_log=log)
    assert seen == ["0", "0", "0"]


def test_cuda_fallback_keeps_cpu_cases_sequential(tmp_path, monkeypatch):
    monkeypatch.setattr("fd_airquality.execution.usable_gpu_slots", lambda _: [])
    called = []

    def baseline(settings, mode):
        called.append((settings.device, mode))
        return result(tmp_path, len(called), "cpu")

    def launch(settings, dry_run):
        assert settings.device == "cpu"
        return baseline(settings, "federated")

    plan = [candidate("mlp", mode) for mode in ("local", "centralized", "federated")]
    with RunLogging(tmp_path, "test", stream=StringIO()) as log:
        execute_plan(RunSettings(device="cuda"), plan, tmp_path,
                     max_parallel_models=2, baseline=baseline, launch=launch, progress_log=log)
    assert called == [("cpu", mode) for mode in ("local", "centralized", "federated")]


def test_failure_cancels_running_and_future_candidates_and_stops_at_mode_barrier(tmp_path, monkeypatch):
    monkeypatch.setattr("fd_airquality.execution.usable_gpu_slots", lambda _: ["0", "1"])
    ready, stopped = threading.Barrier(2), threading.Event()
    original = ProcessRunner.cancel

    def cancel(self):
        original(self)
        stopped.set()

    def parallel(settings, mode, gpu, *args):
        assert mode == "local"
        ready.wait(timeout=10)
        if gpu == "0":
            raise RuntimeError("candidate failed")
        assert stopped.wait(timeout=10)
        return result(tmp_path, "cancelled")

    monkeypatch.setattr(ProcessRunner, "cancel", cancel)
    monkeypatch.setattr("fd_airquality.execution._candidate_process", parallel)
    plan = [candidate("mlp", "local"), candidate("autoencoder", "local"), candidate("mlp", "centralized")]
    with (pytest.raises(RuntimeError, match="candidate failed"),
          RunLogging(tmp_path, "test", stream=StringIO()) as log):
        execute_plan(RunSettings(device="cuda"), plan, tmp_path,
                     max_parallel_models=None, baseline=None, launch=None, progress_log=log)
    rows = json.loads((tmp_path / "benchmark.json").read_text())
    assert [row["status"] for row in rows] == ["failed", "cancelled", "cancelled"]
    assert "RuntimeError: candidate failed" in (tmp_path / "run.log").read_text()


def test_candidate_is_an_independent_process_with_explicit_configuration(prepared, tmp_path):
    path, _ = prepared
    log = tmp_path / "worker.log"
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "child-runs"), hidden_size=8,
                           num_server_rounds=1, mlflow_tracking_uri="disabled")
    output = _candidate_process(settings, "local", "", log, ProcessRunner())
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["execution"]["pid"] != os.getpid()
    assert metadata["execution"]["cuda_visible_devices"] == ""
    assert metadata["settings"]["hidden_size"] == 8
    assert "epoch=1/1" in log.read_text()
    assert "Run completed" in (output / "run.log").read_text()
    assert json.loads((output / "test_metrics.json").read_text())["test"]["num-clients"] == 2


def test_cancellation_allows_child_to_flush_logs_and_close_mlflow_as_failed(prepared, tmp_path, mlflow_uri):
    from mlflow import MlflowClient

    path, _ = prepared
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "cancelled-runs"),
                           num_server_rounds=1, mlflow_tracking_uri=mlflow_uri)
    script = (
        "import time\n"
        "import fd_airquality.baselines as baseline\n"
        "from fd_airquality.settings import RunSettings\n"
        "def train(*args, **kwargs):\n"
        "    print('TRAIN_READY', flush=True)\n"
        "    time.sleep(30)\n"
        "baseline.train_local = train\n"
        f"baseline.run_baseline(RunSettings(**{settings.__dict__!r}), 'centralized')\n"
    )
    runner = ProcessRunner()

    def on_line(line):
        if "TRAIN_READY" in line:
            runner.cancel()

    with pytest.raises(subprocess.CalledProcessError):
        runner.stream([sys.executable, "-u", "-c", script], os.environ.copy(),
                      tmp_path / "cancelled.log", "cancel-test", on_line=on_line)
    run, = (tmp_path / "cancelled-runs").glob("*/mlflow.json")
    metadata = json.loads(run.read_text())
    assert MlflowClient(tracking_uri=mlflow_uri).get_run(metadata["run_id"]).info.status == "FAILED"
    assert "Run interrupted" in run.with_name("run.log").read_text()


@pytest.mark.parametrize("limit", [0, -1, True])
def test_invalid_concurrency_is_rejected(tmp_path, limit):
    with pytest.raises(ValueError, match="max-parallel-models"):
        execute_plan(RunSettings(), [], tmp_path,
                     max_parallel_models=limit, baseline=None, launch=None, progress_log=None)


def test_cpu_does_not_probe_gpus(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CPU must not probe CUDA"))
    assert usable_gpu_slots("cpu") == []


def test_gpu_slots_probe_each_visible_device_and_preserve_physical_mapping(monkeypatch):
    import torch

    current = []

    @contextmanager
    def device(index):
        current.append(index)
        try:
            yield
        finally:
            current.pop()

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-A,GPU-B")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setattr("fd_airquality.execution.resolve_device",
                        lambda _: torch.device("cuda" if current[-1] == 1 else "cpu"))
    assert usable_gpu_slots("cuda") == ["GPU-B"]


@pytest.mark.parametrize("tracking", [False, True])
def test_real_process_benchmark_has_complete_comparable_panel_and_logs(prepared, tmp_path, monkeypatch, tracking, request):
    path, _ = prepared
    # Logical slots exercise concurrent processes on CPU, without claiming a two-GPU hardware test.
    monkeypatch.setattr("fd_airquality.execution.usable_gpu_slots", lambda _: ["", ""])
    uri = request.getfixturevalue("mlflow_uri") if tracking else "disabled"
    output = run_benchmark(RunSettings(manifest=str(path), output_dir=str(tmp_path / "matrix"),
                           num_server_rounds=1, hidden_size=8, mlflow_tracking_uri=uri),
                            ["mlp", "autoencoder"], ["local", "centralized"])
    assert (output / "comparison.csv").is_file()
    root = output.parent
    rows = json.loads((root / "benchmark.json").read_text())
    assert len(rows) == 4 and all(row["status"] == "completed" for row in rows)
    for row in rows:
        assert "epoch=1/1" in Path(row["log"]).read_text()
        assert (Path(row["output"]) / "run.log").is_file()
    assert "mode-completed" in (root / "benchmark.log").read_text()
    if tracking:
        from mlflow import MlflowClient

        client = MlflowClient(tracking_uri=uri)
        experiment = client.get_experiment_by_name("fd-airquality")
        runs = client.search_runs([experiment.experiment_id])
        assert len(runs) == 4
        assert all(run.info.status == "FINISHED" for run in runs)
        assert all("run.log" in {artifact.path for artifact in client.list_artifacts(run.info.run_id)} for run in runs)
