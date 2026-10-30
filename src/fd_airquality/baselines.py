"""Non-federated controls using exactly the prepared station windows and metrics."""

from __future__ import annotations

import json
import math
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
    if source_fingerprints() != manifest["preparation_sources"]:
        raise ValueError("Cambió el código local de preparación; vuelve a ejecutar prepare.")
    output = Path(settings.output_dir).resolve() / (
        f"{manifest['pollutant']}_{settings.model}_{effective_adaptation(settings)}_{mode}_{datetime.now(UTC):%Y%m%d_%H%M%S_%f}"
    )
    output.mkdir(parents=True, exist_ok=False)
    settings, manifest = prepare_runtime_panel(settings, output)
    validate_selection_coverage(manifest, settings, mode)
    settings = replace(settings, device=configure_torch(settings, settings.seed).type)
    artifacts, datasets = [], []
    for partition in range(len(manifest["clients"])):
        _, artifact = load_client_data(settings.manifest, partition)
        artifacts.append(artifact)
        datasets.append({split: WindowDataset(artifact, split, geometry["context"], geometry["horizon"])
                         for split in ("train", "validation", "test")})
    write_run(output, settings, manifest, mode)
    with RunTracker(settings, output, mode, manifest) as tracker:
        _fit(settings, mode, manifest, artifacts, datasets, output, tracker)
    print(f"Resultados {mode}: {output}", flush=True)
    return output


def _fit(settings, mode, manifest, artifacts, datasets, output, tracker):
    geometry = manifest["geometry"]
    count = len(datasets)
    independent = mode == "local"
    models = []
    for _ in range(count if independent else 1):
        device = configure_torch(settings, settings.seed)
        models.append(make_model(geometry, settings).to(device))
    recorder = Recorder(output, manifest, tracker)
    best_scores = [float("inf")] * len(models)
    best_rounds = [-1] * len(models)
    selected_states = [None] * len(models)
    stages = 0 if mode == "zero-shot" else settings.num_server_rounds

    def evaluate(split):
        return [evaluate_local(
            models[partition if independent else 0], data[split], settings, device,
            artifacts[partition], geometry["seasonality"],
        ) for partition, data in enumerate(datasets)]

    for stage in range(stages + 1):
        if stage:
            if independent:
                records = []
                for partition, (model, data) in enumerate(zip(models, datasets, strict=True)):
                    seed = settings.seed + 100_003 * stage + partition
                    configure_torch(settings, seed)
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
        candidates = [selection_value(record, settings, local=True) for record in validation] if independent else [selection_value(pooled, settings)]
        for index, candidate in enumerate(candidates):
            if candidate < best_scores[index]:
                best_scores[index], best_rounds[index] = candidate, stage
                selected_states[index] = _copy_state(models[index])
    for index, model in enumerate(models):
        load_shared_state(model, selected_states[index])
        filename = f"client_{index:03d}_best_model.pt" if independent else "best_model.pt"
        torch.save(checkpoint_payload(geometry, settings, selected_states[index], best_rounds[index]), output / filename)
    # Test is accessed once, only after validation-only checkpoint selection finishes.
    tests = evaluate("test")
    metrics = recorder.record(max(best_rounds), "test", tests, best_rounds if independent else None)
    for index, (stage, score) in enumerate(zip(best_rounds, best_scores, strict=True)):
        tracker.log_metrics("selection", {"round": stage, "validation_score": score}, stage,
                            partition=index if independent else None)
    (output / "test_metrics.json").write_text(json.dumps({
        "selected_round": None if independent else best_rounds[0],
        "selected_rounds": best_rounds,
        "selection_metric": f"{'station_' if independent else ''}validation_{settings.selection_metric}",
        "validation_score": best_scores if independent else best_scores[0], "test": metrics,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
