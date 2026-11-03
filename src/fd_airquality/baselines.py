"""Non-federated controls using exactly the prepared station windows and metrics."""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch
from flwr.app import MetricRecord, RecordDict
from torch.utils.data import ConcatDataset

from fd_airquality.artifacts import source_fingerprints
from fd_airquality.data import WindowDataset, load_client_data, read_manifest
from fd_airquality.models import (
    GRANITE,
    checkpoint_payload,
    effective_adaptation,
    load_shared_state,
    make_model,
    shared_state,
    validate_model_geometry,
    validate_run_mode,
)
from fd_airquality.panels import prepare_runtime_panel
from fd_airquality.run_logging import RunLogging, progress_context, report_progress, run_label
from fd_airquality.settings import RunSettings
from fd_airquality.strategy import aggregate_evaluation, aggregate_training, coverage_metrics
from fd_airquality.task import (
    configure_torch,
    evaluate_local,
    selection_value,
    train_local,
    validate_selection_coverage,
)
from fd_airquality.tracking import RunTracker, write_run


def _copy_state(model):
    return {key: value.detach().cpu().clone() for key, value in shared_state(model).items()}


@contextmanager
def _model_on_device(model, device, *, offload: bool):
    try:
        model.to(device)
        yield model
    finally:
        if offload:
            model.to("cpu")


class Recorder:
    def __init__(self, output: Path, manifest: dict, tracker: RunTracker):
        self.output, self.manifest, self.tracker = output, manifest, tracker
        self.rounds, self.stations = [], []

    def record(self, stage: int, split: str, records: list[dict], station_steps: list[int] | None = None) -> dict:
        contents = [RecordDict({"metrics": MetricRecord(record)}) for record in records]
        aggregate = aggregate_training if split == "train" else aggregate_evaluation
        metrics = dict(aggregate(contents, "num-examples"))
        if split != "train":
            metrics.update(coverage_metrics(contents, self.manifest))
        self.rounds.append({"round": stage, "split": split, **metrics})
        self.tracker.log_metrics(split, metrics, stage)
        for partition, record in enumerate(records):
            station_step = station_steps[partition] if station_steps is not None else stage
            row = {
                "round": station_step, "split": split, "partition-id": partition,
                "station": self.manifest["clients"][partition]["station"], **record,
            }
            if "mse" in row:
                row["rmse"] = math.sqrt(row["mse"])
                for name in ("mase", "rmsse"):
                    if row[f"{name}-count"]:
                        row[f"mean-{name}"] = row[f"{name}-sum"] / row[f"{name}-count"]
            self.stations.append(row)
            self.tracker.log_metrics(split, row, station_step, partition=partition)
        pd.DataFrame(self.rounds).to_csv(self.output / "round_metrics.csv", index=False)
        pd.DataFrame(self.stations).to_csv(self.output / "station_metrics.csv", index=False)
        return metrics


def run_baseline(settings: RunSettings, mode: str) -> Path:
    if mode not in {"local", "centralized", "zero-shot"}:
        raise ValueError("El baseline debe ser local, centralized o zero-shot.")
    if mode == "zero-shot" and settings.model not in GRANITE:
        raise ValueError("zero-shot está reservado a los checkpoints Granite preentrenados.")
    if mode == "zero-shot":
        settings = replace(settings, adaptation="none")
    validate_run_mode(settings, mode)
    manifest = read_manifest(settings.manifest)
    geometry = manifest["geometry"]
    validate_model_geometry(geometry, settings)
    output = Path(settings.output_dir).resolve() / (
        f"{manifest['pollutant']}_{settings.model}_{effective_adaptation(settings)}_{mode}_{datetime.now(UTC):%Y%m%d_%H%M%S_%f}"
    )
    output.mkdir(parents=True, exist_ok=False)
    with RunLogging(output, run_label(settings, mode), level=settings.log_level,
                    heartbeat_seconds=settings.heartbeat_seconds):
        return _run_baseline(settings, mode, manifest, output)


def _run_baseline(settings, mode, manifest, output):
    geometry = manifest["geometry"]
    if source_fingerprints() != manifest["preparation_sources"]:
        raise ValueError("Cambió el código local de preparación; vuelve a ejecutar prepare.")
    report_progress("panel", f"manifest={settings.manifest}")
    settings, manifest = prepare_runtime_panel(settings, output)
    validate_selection_coverage(manifest, settings, mode)
    settings = replace(settings, device=configure_torch(settings, settings.seed).type)
    report_progress("loading-data", f"stations={len(manifest['clients'])} device={settings.device}")
    artifacts, datasets = [], []
    for partition in range(len(manifest["clients"])):
        report_progress("loading-data", f"station={manifest['clients'][partition]['station']} "
                        f"partition={partition + 1}/{len(manifest['clients'])}")
        _, artifact = load_client_data(settings.manifest, partition)
        artifacts.append(artifact)
        datasets.append({split: WindowDataset(artifact, split, geometry["context"], geometry["horizon"])
                         for split in ("train", "validation", "test")})
    report_progress("loading-model", f"model={settings.model} device={settings.device}")
    write_run(output, settings, manifest, mode)
    with RunTracker(settings, output, mode, manifest) as tracker:
        _fit(settings, mode, manifest, artifacts, datasets, output, tracker)
    print(f"Resultados {mode}: {output}", flush=True)
    return output


def _fit(settings, mode, manifest, artifacts, datasets, output, tracker):
    geometry = manifest["geometry"]
    count = len(datasets)
    independent = mode == "local"
    offload = independent and settings.device == "cuda"
    models = []
    for index in range(count if independent else 1):
        report_progress("loading-model", f"instance={index + 1}/{count if independent else 1}")
        device = configure_torch(settings, settings.seed)
        models.append(make_model(geometry, settings).to("cpu" if offload else device))
    if offload:
        report_progress("model-residency", "local_models=cpu active_gpu_models=1")
    recorder = Recorder(output, manifest, tracker)
    best_scores = [float("inf")] * len(models)
    best_rounds = [-1] * len(models)
    selected_states = [None] * len(models)
    stages = 0 if mode == "zero-shot" else settings.num_server_rounds

    def evaluate(split):
        records = []
        for partition, data in enumerate(datasets):
            model = models[partition if independent else 0]
            with (
                progress_context(station=manifest["clients"][partition]["station"], partition=partition),
                _model_on_device(model, device, offload=offload),
            ):
                report_progress(split, f"station={partition + 1}/{count}")
                records.append(evaluate_local(
                    model, data[split], settings, device,
                    artifacts[partition], geometry["seasonality"],
                ))
        return records

    for stage in range(stages + 1):
        report_progress("round", f"round={stage}/{stages} mode={mode}")
        if stage:
            if independent:
                records = []
                for partition, (model, data) in enumerate(zip(models, datasets, strict=True)):
                    seed = settings.seed + 100_003 * stage + partition
                    configure_torch(settings, seed)
                    with (
                        progress_context(station=manifest["clients"][partition]["station"],
                                         partition=partition, round=stage),
                        _model_on_device(model, device, offload=offload),
                    ):
                        records.append(train_local(model, data["train"], settings, device, seed))
                recorder.record(stage, "train", records)
            else:
                # ConcatDataset concatenates window samples, NEVER raw series or timestamps.
                seed = settings.seed + 100_003 * stage
                configure_torch(settings, seed)
                metrics = train_local(models[0], ConcatDataset([data["train"] for data in datasets]),
                                      settings, device, seed)
                recorder.rounds.append({"round": stage, "split": "train", **metrics})
                tracker.log_metrics("train", metrics, stage)
        validation = evaluate("validation")
        pooled = recorder.record(stage, "validation", validation)
        report_progress("validation", f"round={stage} mae={pooled['mae']:.6g} "
                        f"{settings.selection_metric}={pooled.get(settings.selection_metric)}")
        candidates = [selection_value(record, settings, local=True) for record in validation] if independent else [selection_value(pooled, settings)]
        for index, candidate in enumerate(candidates):
            if candidate < best_scores[index]:
                best_scores[index], best_rounds[index] = candidate, stage
                selected_states[index] = _copy_state(models[index])
                report_progress("checkpoint", f"model_index={index} selected_round={stage} score={candidate:.6g}")
    for index, model in enumerate(models):
        load_shared_state(model, selected_states[index])
        filename = f"client_{index:03d}_best_model.pt" if independent else "best_model.pt"
        torch.save(checkpoint_payload(geometry, settings, selected_states[index], best_rounds[index]), output / filename)
    # Test is accessed once, only after validation-only checkpoint selection finishes.
    tests = evaluate("test")
    metrics = recorder.record(max(best_rounds), "test", tests, best_rounds if independent else None)
    report_progress("test", f"mae={metrics['mae']:.6g} rmse={metrics['rmse']:.6g}; test final único")
    for index, (stage, score) in enumerate(zip(best_rounds, best_scores, strict=True)):
        tracker.log_metrics("selection", {"round": stage, "validation_score": score}, stage,
                            partition=index if independent else None)
    (output / "test_metrics.json").write_text(json.dumps({
        "selected_round": None if independent else best_rounds[0],
        "selected_rounds": best_rounds,
        "selection_metric": f"{'station_' if independent else ''}validation_{settings.selection_metric}",
        "validation_score": best_scores if independent else best_scores[0], "test": metrics,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
