"""Flower Message API client; only this layer opens station-local artifacts."""

import gc
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from torch.utils.data import ConcatDataset

from fd_airquality.arrival import (
    arrival_metrics,
    audit_normalization,
    normalization_artifact,
    round_epoch_limit,
    round_validation_view,
    training_view,
)
from fd_airquality.artifacts import source_fingerprints
from fd_airquality.calibration import calibration_options, validation_view
from fd_airquality.data import WindowDataset, load_client_data
from fd_airquality.federated_adapters import tensor_bytes
from fd_airquality.models import load_shared_state, shared_state
from fd_airquality.run_logging import RunLogging, progress_context, report_progress, run_label
from fd_airquality.settings import GATEWAY_MODE, RunSettings
from fd_airquality.task import configure_torch, evaluate_local, make_model, train_local
from fd_airquality.training_control import controls_enabled


def _local_task(message: Message, context: Context, training: bool) -> Message:
    settings = RunSettings.from_mapping(context.run_config)
    config = message.content["config"]
    if "manifest" in config:
        settings = replace(settings, manifest=str(config["manifest"]))
    partition = int(config["partition-id"] if "partition-id" in config else context.node_config["partition-id"])
    logs = (RunLogging(Path(config["log-dir"]), run_label(settings, settings.federation_mode) + f" partition={partition}",
                       filename=f"client_{partition:03d}.log", level=settings.log_level,
                       heartbeat_seconds=settings.heartbeat_seconds) if "log-dir" in config else nullcontext())
    with logs, progress_context(partition=partition, round=config.get("server-round", 0)):
        report_progress("client", f"training={training}")
        return _execute_task(message, settings, partition, training)


def _execute_task(message, settings, partition, training):
    config = message.content["config"]
    gateway = settings.federation_mode == GATEWAY_MODE
    members = config["station-partitions"] if gateway else [partition]
    split = "train" if training else str(config.get("split", "validation"))
    if not training and split not in {"validation", "test"}:
        raise ValueError(f"Split no permitido: {split}")
    if not members:
        metrics = _idle_training() if training else {}
        metrics.update({"partition-id": partition, "num-stations": 0, "idle-client": 1})
        if training:
            metrics.update({"uplink-tensor-bytes": 0, "downlink-tensor-bytes": 0,
                            "candidate-windows": 0, "availability-excluded-windows": 0,
                             "round-validation-excluded-windows": 0, "round-validation-windows": 0,
                             "round-validation-fallback": 0, "arrival-unobserved-windows": 0})
        return Message(content=RecordDict({"metrics": MetricRecord(metrics)}), reply_to=message)
    log_dir = Path(config["log-dir"]) if "log-dir" in config else None
    artifacts = []
    for station in members:
        manifest, artifact = load_client_data(settings.manifest, station)
        if not artifacts and source_fingerprints() != manifest["preparation_sources"]:
            raise ValueError("Cambió el código local de preparación; vuelve a ejecutar prepare.")
        artifact = normalization_artifact(artifact, manifest["geometry"], settings)
        audit_normalization(artifact, settings, station, log_dir)
        artifacts.append(artifact)
    geometry = manifest["geometry"]
    round_number = int(config.get("server-round", 0))
    seed = settings.seed + 100_003 * round_number + partition
    device = configure_torch(settings, seed)
    model = None
    try:
        model = make_model(geometry, settings)
        initial = message.content["arrays"].to_torch_state_dict()
        load_shared_state(model, initial)
        downlink_bytes = tensor_bytes(initial) if training else 0
        del initial
        # Upstream constructors seed their initialization. Restore the round/client
        # RNG for any stochastic training layers, independently of model loading.
        configure_torch(settings, seed)
        if training:
            metrics, rows = _fit_client(model, geometry, settings, device, seed, config, artifacts)
            content = RecordDict()
            if metrics["num-examples"] or not gateway:
                state = {key: value.detach().cpu() for key, value in shared_state(model).items()
                         if not key.startswith("history.")}
                content["arrays"] = ArrayRecord(state)
                metrics["uplink-tensor-bytes"] = tensor_bytes(state)
            else:
                metrics["uplink-tensor-bytes"] = 0
            metrics["downlink-tensor-bytes"] = downlink_bytes
        else:
            rows = []
            for station, artifact in zip(members, artifacts, strict=True):
                dataset = (validation_view(artifact, geometry, settings) if split == "validation" else
                           WindowDataset(artifact, "test", geometry["context"], geometry["horizon"]))
                options = (calibration_options(
                    model, artifact, geometry, manifest["clients"][station], settings, device,
                    log_dir, station,
                ) if split == "test" else {})
                rows.append(evaluate_local(model, dataset, settings, device, artifact,
                                           geometry["seasonality"], **options))
            metrics = {} if gateway else rows[0]
            content = RecordDict()
        if gateway:
            metrics["num-stations"] = len(members)
            for station, row in zip(members, rows, strict=True):
                content[f"station-{station}"] = MetricRecord(row | {"partition-id": station})
        metrics["partition-id"] = partition
        content["metrics"] = MetricRecord(metrics)
        return Message(content=content, reply_to=message)
    finally:
        # Ray reuses actors. Offload even if an error's traceback still owns the
        # model; otherwise collect PatchTST's bound-method cycle before empty_cache.
        if device.type == "cuda":
            try:
                if model is not None:
                    model.to("cpu")
            finally:
                del model
                gc.collect()
                torch.cuda.empty_cache()
                report_progress("cuda-cleanup", f"allocated_mib={torch.cuda.memory_allocated(device) / 1024**2:.1f} "
                                f"reserved_mib={torch.cuda.memory_reserved(device) / 1024**2:.1f}")


def _idle_training() -> dict:
    return {
        "num-examples": 0, "num-targets": 0, "train-mse-normalized": 0.,
        "train-seconds": 0., "optimizer-steps": 0,
        "admissible-windows": 0, "unique-training-windows": 0,
        "selected-target-occurrences": 0, "padded-context-slots": 0, "padded-target-slots": 0,
        "epochs-completed": 0, "early-stopped": 0, "learning-rate-reductions": 0,
        "sampled-window-occurrences": 0, "sampled-target-occurrences": 0,
        "replacement-window-occurrences": 0,
    }


def _fit_client(model, geometry, settings, device, seed, config, artifacts):
    gateway = settings.federation_mode == GATEWAY_MODE
    round_number = int(config.get("server-round", 0))
    views = [training_view(artifact, geometry, settings, round_number) for artifact in artifacts]
    dataset = ConcatDataset(views) if gateway else views[0]
    update = replace(settings, local_epochs=round_epoch_limit(geometry, settings, round_number))
    rows, options = [], {}
    fallback = False
    if len(dataset) and controls_enabled(update):
        validation = [
            held_out for artifact, view in zip(artifacts, views, strict=True)
            if len(view) and (held_out := round_validation_view(artifact, geometry, settings, view)) is not None
        ]
        if validation:
            options["validation_data"] = ConcatDataset(validation) if gateway else validation[0]
        else:
            fallback = True
            update = replace(update, local_epochs=min(update.local_epochs, update.round_validation_fallback_epochs),
                             early_stopping=0, lr_scheduler="none")
            report_progress("training-validation", f"Sin conjunto de validación separado; épocas={update.local_epochs}.")
    if gateway:
        options["group_metrics"] = rows
    metrics = train_local(
        model, dataset, update, device, seed, proximal_mu=float(config.get("proximal-mu", 0.0)),
        max_train_windows=geometry.get("max_train_windows"), **options,
    ) if len(dataset) else _idle_training()
    diagnostics = [arrival_metrics(view) for view in views]
    for diagnostic in diagnostics:
        if config.get("available-through-ns") != diagnostic.get("available-through-ns"):
            raise ValueError("El reloj de llegada del cliente no coincide con el del servidor.")
        if config.get("available-from-ns") != diagnostic.get("available-from-ns"):
            raise ValueError("El bloque de llegada del cliente no coincide con el del servidor.")
    metrics["idle-client"] = int(not len(dataset))
    metrics["round-validation-fallback"] = int(fallback)
    if gateway:
        if not rows:
            rows = [_idle_training() for _ in views]
        for row, diagnostic in zip(rows, diagnostics, strict=True):
            row.update(diagnostic)
        for key in ("candidate-windows", "availability-excluded-windows", "round-validation-windows",
                    "round-validation-excluded-windows",
                    "arrival-unobserved-windows"):
            metrics[key] = sum(row[key] for row in diagnostics)
    else:
        metrics.update(diagnostics[0])
    return metrics, rows


def build_client_app(run_config: dict | None = None) -> ClientApp:
    app = ClientApp()

    @app.train()
    def train(message: Message, context: Context) -> Message:
        if run_config is not None:
            context.run_config.update(run_config)
        return _local_task(message, context, training=True)

    @app.evaluate()
    def evaluate(message: Message, context: Context) -> Message:
        if run_config is not None:
            context.run_config.update(run_config)
        return _local_task(message, context, training=False)

    return app


app = build_client_app()
