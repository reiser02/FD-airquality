"""Non-federated controls using exactly the prepared station windows and metrics."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch
from flwr.app import MetricRecord, RecordDict
from torch.utils.data import ConcatDataset

from fd_airquality.arrival import audit_normalization, normalization_artifact, training_view
from fd_airquality.artifacts import source_fingerprints
from fd_airquality.calibration import calibration_options, validation_view
from fd_airquality.data import WindowDataset, load_client_data, read_manifest
from fd_airquality.lightweight import HOLT_GRID, STATION_CONTROLS
from fd_airquality.models import (
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
from fd_airquality.settings import BASELINE_MODES, RunSettings
from fd_airquality.strategy import aggregate_evaluation, aggregate_training, coverage_metrics
from fd_airquality.task import (
    configure_torch,
    evaluate_local,
    selection_value,
    train_local,
    validate_selection_coverage,
)
from fd_airquality.tracking import RunTracker, write_run
from fd_airquality.training_control import controls_enabled, training_stage_count


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
            for name in ("mase", "rmsse"):
                if row.get(f"{name}-count"):
                    row[f"mean-{name}"] = row[f"{name}-sum"] / row[f"{name}-count"]
            self.stations.append(row)
        pd.DataFrame(self.rounds).to_csv(self.output / "round_metrics.csv", index=False)
        pd.DataFrame(self.stations).to_csv(self.output / "station_metrics.csv", index=False)
        return metrics


def run_baseline(settings: RunSettings, mode: str) -> Path:
    if mode not in BASELINE_MODES:
        raise ValueError("El baseline debe ser local o centralized.")
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
    train = bool(training_stage_count(settings, mode))
    report_progress("loading-data", f"stations={len(manifest['clients'])} device={settings.device}")
    artifacts, datasets = [], []
    for partition in range(len(manifest["clients"])):
        report_progress("loading-data", f"station={manifest['clients'][partition]['station']} "
                        f"partition={partition + 1}/{len(manifest['clients'])}")
        _, artifact = load_client_data(settings.manifest, partition)
        artifact = normalization_artifact(artifact, geometry, settings)
        audit_normalization(artifact, settings, partition, output / "clients")
        artifacts.append(artifact)
        data = {"validation": validation_view(artifact, geometry, settings),
                "test": WindowDataset(artifact, "test", geometry["context"], geometry["horizon"])}
        if train:
            data["train"] = training_view(artifact, geometry, settings)
            if not len(data["train"]):
                raise ValueError(f"Sin ventanas de entrenamiento para {manifest['clients'][partition]['station']} tras el umbral.")
        datasets.append(data)
    report_progress("loading-model", f"model={settings.model} device={settings.device}")
    write_run(output, settings, manifest, mode)
    with RunTracker(settings, output, mode, manifest) as tracker:
        _fit(settings, mode, manifest, artifacts, datasets, output, tracker, train=train)
    print(f"Resultados {mode}: {output}", flush=True)
    return output


def _fit(settings, mode, manifest, artifacts, datasets, output, tracker, *, train: bool):
    geometry = manifest["geometry"]
    count = len(datasets)
    independent = mode == "local" and (train or settings.model in STATION_CONTROLS)
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
    selected_steps = [-1] * len(models)
    selected_states = [None] * len(models)
    if settings.model == "holt":
        for index, model in enumerate(models):
            with _model_on_device(model, device, offload=offload):
                best = (float("inf"), HOLT_GRID[0])
                for alpha, beta in HOLT_GRID:
                    model.set_coefficients(alpha, beta)
                    record = evaluate_local(model, datasets[index]["validation"], settings, device,
                                            artifacts[index], geometry["seasonality"])
                    score = selection_value(record, settings, local=True)
                    if score < best[0]:
                        best = score, (alpha, beta)
                model.set_coefficients(*best[1])

    def evaluate(split):
        records = []
        for partition, data in enumerate(datasets):
            model = models[partition if independent else 0]
            with (
                progress_context(station=manifest["clients"][partition]["station"], partition=partition),
                _model_on_device(model, device, offload=offload),
            ):
                report_progress(split, f"station={partition + 1}/{count}")
                options = (calibration_options(
                    model, artifacts[partition], geometry, manifest["clients"][partition], settings,
                    device, output / "clients", partition,
                ) if split == "test" else {})
                records.append(evaluate_local(
                    model, data[split], settings, device,
                    artifacts[partition], geometry["seasonality"], **options,
                ))
        return records

    def validate_checkpoint(step, label):
        validation = evaluate("validation")
        pooled = recorder.record(step, "validation", validation)
        report_progress("validation", f"checkpoint={label} {settings.selection_metric}={pooled.get(settings.selection_metric)}")
        candidates = ([selection_value(record, settings, local=True) for record in validation]
                      if independent else [selection_value(pooled, settings)])
        for index, candidate in enumerate(candidates):
            if candidate < best_scores[index]:
                best_scores[index], selected_steps[index] = candidate, step
                selected_states[index] = _copy_state(models[index])
                report_progress("checkpoint", f"model_index={index} selected={label} score={candidate:.6g}")

    # Keep the historical "round" fields as checkpoint indices: 0 initial, 1 fitted.
    validate_checkpoint(0, "initial" if train else "reference")
    fit_step = int(train)
    if train:
        report_progress("train", f"single-fit mode={mode} max_epochs={settings.local_epochs}")
        fit_seed = settings.seed + 100_003
        monitor = controls_enabled(settings)
        if independent:
            records = []
            for partition, (model, data) in enumerate(zip(models, datasets, strict=True)):
                seed = fit_seed + partition
                configure_torch(settings, seed)
                with (
                    progress_context(station=manifest["clients"][partition]["station"], partition=partition),
                    _model_on_device(model, device, offload=offload),
                ):
                    records.append(train_local(
                        model, data["train"], settings, device, seed,
                        max_train_windows=geometry.get("max_train_windows"),
                        validation_data=data["validation"] if monitor else None,
                    ))
            recorder.record(fit_step, "train", records)
        else:
            # ConcatDataset concatenates window samples, NEVER raw series or timestamps.
            configure_torch(settings, fit_seed)
            metrics = train_local(
                models[0], ConcatDataset([data["train"] for data in datasets]), settings, device, fit_seed,
                max_train_windows=geometry.get("max_train_windows"),
                validation_data=ConcatDataset([data["validation"] for data in datasets]) if monitor else None,
            )
            recorder.rounds.append({"round": fit_step, "split": "train", **metrics})
            tracker.log_metrics("train", metrics, fit_step)
        validate_checkpoint(fit_step, "fitted")

    for index, model in enumerate(models):
        last_filename = f"client_{index:03d}_last_model.pt" if independent else "last_model.pt"
        torch.save(checkpoint_payload(geometry, settings, _copy_state(model), fit_step, state_role="training"), output / last_filename)
        load_shared_state(model, selected_states[index])
        filename = f"client_{index:03d}_best_model.pt" if independent else "best_model.pt"
        torch.save(checkpoint_payload(geometry, settings, selected_states[index], selected_steps[index]), output / filename)
    # Test is accessed once, only after validation-only checkpoint selection finishes.
    tests = evaluate("test")
    metrics = recorder.record(max(selected_steps), "test", tests, selected_steps if independent else None)
    report_progress("test", f"macro-mase={metrics.get('macro-mase', '-')} "
                    f"macro-rmsse={metrics.get('macro-rmsse', '-')}; test final único")
    if not independent:
        tracker.log_metrics("selection", {"round": selected_steps[0], "validation_score": best_scores[0]}, selected_steps[0])
    (output / "test_metrics.json").write_text(json.dumps({
        "selected_round": None if independent else selected_steps[0],
        "selection_metric": f"{'station_' if independent else ''}validation_{settings.selection_metric}",
        "validation_score": None if independent else best_scores[0], "test": metrics,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
