"""Flower Message API client; only this layer opens station-local artifacts."""

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fd_airquality.arrival import (
    arrival_metrics,
    audit_normalization,
    normalization_artifact,
    round_validation_view,
    training_view,
)
from fd_airquality.artifacts import source_fingerprints
from fd_airquality.data import WindowDataset, load_client_data
from fd_airquality.federated_adapters import tensor_bytes
from fd_airquality.models import load_shared_state, shared_state
from fd_airquality.run_logging import RunLogging, progress_context, report_progress, run_label
from fd_airquality.settings import RunSettings
from fd_airquality.task import configure_torch, evaluate_local, make_model, train_local
from fd_airquality.training_control import controls_enabled


def _local_task(message: Message, context: Context, training: bool) -> Message:
    settings = RunSettings.from_mapping(context.run_config)
    config = message.content["config"]
    if "manifest" in config:
        settings = replace(settings, manifest=str(config["manifest"]))
    partition = int(config.get("partition-id", context.node_config["partition-id"]))
    logs = (RunLogging(Path(config["log-dir"]), run_label(settings, "federated") + f" partition={partition}",
                       filename=f"client_{partition:03d}.log", level=settings.log_level,
                       heartbeat_seconds=settings.heartbeat_seconds) if "log-dir" in config else nullcontext())
    with logs, progress_context(partition=partition, round=config.get("server-round", 0)):
        report_progress("client", f"training={training}")
        return _execute_task(message, settings, partition, training)


def _execute_task(message, settings, partition, training):
    manifest, artifact = load_client_data(settings.manifest, partition)
    if source_fingerprints() != manifest["preparation_sources"]:
        raise ValueError("Cambió el código local de preparación; vuelve a ejecutar prepare.")
    geometry = manifest["geometry"]
    round_number = int(message.content["config"].get("server-round", 0))
    seed = settings.seed + 100_003 * round_number + partition
    artifact = normalization_artifact(artifact, geometry, settings)
    log_dir = message.content["config"].get("log-dir")
    audit_normalization(artifact, settings, partition, Path(log_dir) if log_dir else None)
    device = configure_torch(settings, seed)
    model = make_model(geometry, settings)
    load_shared_state(model, message.content["arrays"].to_torch_state_dict())
    # Upstream constructors seed their initialization. Restore the round/client
    # RNG for any stochastic training layers, independently of model loading.
    configure_torch(settings, seed)
    split = "train" if training else str(message.content["config"].get("split", "validation"))
    if split not in {"train", "validation", "test"} or (not training and split == "train"):
        raise ValueError(f"Split no permitido: {split}")
    dataset = (training_view(artifact, geometry, settings, round_number) if training else
               WindowDataset(artifact, split, geometry["context"], geometry["horizon"]))
    if training:
        metrics = train_local(
            model, dataset, settings, device, seed,
            proximal_mu=float(message.content["config"].get("proximal-mu", 0.0)),
            max_train_windows=geometry.get("max_train_windows"),
            **({"validation_data": round_validation_view(artifact, geometry, settings)}
               if controls_enabled(settings) else {}),
        ) if len(dataset) else {
            "num-examples": 0, "num-targets": 0, "train-mse-normalized": 0.,
            "train-seconds": 0., "optimizer-steps": 0,
            "admissible-windows": 0, "unique-training-windows": 0,
            "selected-target-occurrences": 0, "padded-context-slots": 0, "padded-target-slots": 0,
            "epochs-completed": 0, "early-stopped": 0, "learning-rate-reductions": 0,
            "sampled-window-occurrences": 0, "sampled-target-occurrences": 0,
            "replacement-window-occurrences": 0,
        }
        metrics.update(arrival_metrics(dataset), **{"idle-client": int(not len(dataset))})
        if message.content["config"].get("available-through-ns") != metrics.get("available-through-ns"):
            raise ValueError("El reloj de llegada del cliente no coincide con el del servidor.")
        if message.content["config"].get("available-from-ns") != metrics.get("available-from-ns"):
            raise ValueError("El bloque de llegada del cliente no coincide con el del servidor.")
        state = {key: value.detach().cpu() for key, value in shared_state(model).items() if not key.startswith("history.")}
        metrics["uplink-tensor-bytes"] = tensor_bytes(state)
        metrics["downlink-tensor-bytes"] = tensor_bytes(message.content["arrays"].to_torch_state_dict())
    else:
        metrics = evaluate_local(model, dataset, settings, device, artifact, geometry["seasonality"])
    # Integer identity is metadata, not an aggregation weight.
    metrics["partition-id"] = partition
    content = RecordDict({"metrics": MetricRecord(metrics)})
    if training:
        content["arrays"] = ArrayRecord(state)
    return Message(content=content, reply_to=message)


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
