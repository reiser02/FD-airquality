"""Mode barriers and one isolated non-federated candidate per usable GPU."""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import replace
from pathlib import Path

from fd_airquality.processes import ProcessRunner
from fd_airquality.run_logging import RunLogging, report_progress, run_label
from fd_airquality.task import resolve_device

LOGGER = logging.getLogger(__name__)


def usable_gpu_slots(requested: str) -> list[str]:
    if requested == "cpu":
        return []
    import torch

    if not torch.cuda.is_available():
        resolve_device("cuda")
        return []
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    tokens = visible.split(",") if visible else [str(index) for index in range(torch.cuda.device_count())]
    slots = []
    for index in range(torch.cuda.device_count()):
        try:
            with torch.cuda.device(index):
                if resolve_device("cuda").type == "cuda":
                    slots.append(tokens[index].strip())
        except (RuntimeError, AssertionError) as exc:
            LOGGER.warning("GPU visible %d no utilizable: %s", index, exc)
    return slots


def write_progress(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _case_settings(settings, case, root, index):
    return replace(settings, output_dir=str(root / "runs" / f"case_{index + 1:04d}"),
                   **{key: value for key, value in case.items() if key != "mode"})


def _candidate_process(settings, mode: str, gpu: str, log_path: Path, runner: ProcessRunner) -> Path:
    config = log_path.with_suffix(".toml")
    config.write_text("[run]\n" + "\n".join(
        f"{key} = {json.dumps(value, ensure_ascii=False)}"
        for key, value in settings.flower_config().items()
    ) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS=str(settings.torch_threads), MKL_NUM_THREADS=str(settings.torch_threads))
    command = [sys.executable, "-m", "fd_airquality", "baseline", "--config", str(config), "--mode", mode]
    runner.stream(command, env, log_path, f"{run_label(settings, mode)} gpu={gpu}")
    runs = list(Path(settings.output_dir).glob("*/run.json"))
    if len(runs) != 1 or not (runs[0].parent / "test_metrics.json").is_file():
        raise RuntimeError(f"El candidato terminó sin un único resultado de test completo. Consulta {log_path}")
    return runs[0].parent


def execute_plan(settings, plan: list[dict], root: Path, *,
                 max_parallel_models: int | None, baseline, launch, progress_log) -> list[Path]:
    if max_parallel_models is not None and (type(max_parallel_models) is not int or max_parallel_models < 1):
        raise ValueError("max-parallel-models debe ser un entero positivo.")
    slots = usable_gpu_slots(settings.device)
    if settings.device == "cuda" and not slots:
        LOGGER.warning("Ninguna GPU utilizable; benchmark secuencial en CPU.")
        settings = replace(settings, device="cpu")
    slots = slots[:max_parallel_models] if max_parallel_models else slots
    (root / "logs").mkdir()
    rows = [case | {"status": "pending", "gpu": None, "log": str(root / "logs" / f"case_{i + 1:04d}.log")}
            for i, case in enumerate(plan)]
    paths = [None] * len(plan)
    progress_path = root / "benchmark.json"
    runner = ProcessRunner()

    def persist(mode):
        write_progress(progress_path, rows)
        progress_log.update(stage=mode, completed=sum(row["status"] == "completed" for row in rows),
                            total=len(rows), running=sum(row["status"] == "running" for row in rows),
                            pending=sum(row["status"] == "pending" for row in rows))

    def start(index, gpu):
        rows[index].update(status="running", gpu=gpu)
        persist(plan[index]["mode"])
        report_progress(plan[index]["mode"], f"case={index + 1}/{len(plan)} gpu={gpu or 'cpu/Ray'} "
                        f"{plan[index]} log={rows[index]['log']}")
        return _case_settings(settings, plan[index], root, index)

    def complete(index, path):
        paths[index] = path
        rows[index].update(status="completed", output=str(path))
        metadata = path / "run.json"
        if metadata.is_file():
            rows[index]["effective_device"] = json.loads(metadata.read_text())["settings"]["device"]
        persist(plan[index]["mode"])
        report_progress(plan[index]["mode"], f"case={index + 1}/{len(plan)} completed output={path}")

    def failed(index, exc):
        rows[index].update(status="failed", error=str(exc))
        persist(plan[index]["mode"])
        LOGGER.error("Caso %d falló; consulta %s", index + 1, rows[index]["log"],
                     exc_info=(type(exc), exc, exc.__traceback__))

    persist("initializing")
    try:
        for mode in dict.fromkeys(case["mode"] for case in plan):
            indices = [i for i, case in enumerate(plan) if case["mode"] == mode]
            report_progress(mode, f"mode-start candidates={len(indices)} "
                            f"parallel={min(len(slots), len(indices)) if mode != 'federated' and slots else 1}")
            if mode == "federated" or not slots:
                for index in indices:
                    current = start(index, None)
                    try:
                        with RunLogging(root / "logs", run_label(current, mode),
                                        filename=f"case_{index + 1:04d}.log", level=settings.log_level,
                                        heartbeat_seconds=settings.heartbeat_seconds):
                            path = (launch(current, dry_run=False)
                                    if mode == "federated" else baseline(current, mode))
                    except Exception as exc:
                        failed(index, exc)
                        raise
                    complete(index, path)
            else:
                pending = deque(indices)
                free = deque(slots)
                active = {}
                with ThreadPoolExecutor(max_workers=len(slots), thread_name_prefix="candidate-stream") as executor:
                    try:
                        while pending or active:
                            while pending and free:
                                index, gpu = pending.popleft(), free.popleft()
                                current = start(index, gpu)
                                future = executor.submit(_candidate_process, current, mode, gpu,
                                                         Path(rows[index]["log"]), runner)
                                active[future] = (index, gpu)
                            done, _ = wait(active, return_when=FIRST_COMPLETED)
                            for future in sorted(done, key=lambda item: active[item][0]):
                                index, gpu = active.pop(future)
                                try:
                                    complete(index, future.result())
                                except Exception as exc:
                                    failed(index, exc)
                                    raise
                                free.append(gpu)
                    except BaseException:
                        runner.cancel()
                        raise
            report_progress(mode, "mode-completed; barrera antes del siguiente modo")
    except BaseException:
        runner.cancel()
        for row in rows:
            if row["status"] in {"pending", "running"}:
                row["status"] = "cancelled"
        persist("interrupted-or-failed")
        raise
    return paths
