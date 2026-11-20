"""Audit and checkpoint an official Flower strategy without reimplementing FedAvg."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg, FedProx, Strategy

from fd_airquality.arrival import round_bounds
from fd_airquality.calibration import calibration_enabled
from fd_airquality.federated_adapters import aggregate_adapter_updates
from fd_airquality.models import GRANITE, checkpoint_payload
from fd_airquality.run_logging import report_progress
from fd_airquality.settings import RunSettings
from fd_airquality.task import selection_value
from fd_airquality.tracking import RunTracker


def aggregate_training(contents: list[RecordDict], weighted_by_key: str) -> MetricRecord:
    records = [content["metrics"] for content in contents]
    count = sum(record[weighted_by_key] for record in records)
    metrics = {
        "num-examples": sum(record["num-examples"] for record in records),
        "train-mse-normalized": sum(
            record["train-mse-normalized"] * record[weighted_by_key] for record in records
        ) / count if count else 0.,
    }
    if all("num-targets" in record for record in records):
        total_targets = sum(record["num-targets"] for record in records)
        metrics["num-targets"] = total_targets
        metrics["train-mse-normalized"] = (sum(record["train-mse-normalized"] * record["num-targets"]
                                                for record in records) / total_targets if total_targets else 0.)
        if total_targets and all("train-objective-normalized" in record for record in records):
            metrics["train-objective-normalized"] = sum(
                record["train-objective-normalized"] * record["num-targets"] for record in records
            ) / total_targets
    for key in ("candidate-windows", "availability-excluded-windows", "idle-client", "monitoring-excluded-windows",
                "arrival-unobserved-windows", "admissible-windows", "unique-training-windows",
                "padded-context-slots", "padded-target-slots"):
        if all(key in record for record in records):
            metrics[key] = sum(record[key] for record in records)
    for key in ("optimizer-steps", "epochs-completed", "early-stopped", "learning-rate-reductions",
                "selected-target-occurrences", "sampled-window-occurrences", "sampled-target-occurrences",
                "replacement-window-occurrences"):
        if all(key in record for record in records):
            metrics[key] = sum(record[key] for record in records)
    for key in ("train-seconds", "uplink-tensor-bytes", "downlink-tensor-bytes"):
        if all(key in record for record in records):
            metrics[key] = sum(record[key] for record in records)
    for key in ("trainable-parameters", "peak-process-rss-mib", "peak-cuda-allocated-mib"):
        if all(key in record for record in records):
            metrics[key] = max(record[key] for record in records)
    return MetricRecord(metrics)


def _check_training_sampling(metrics, available: int, cap: int | None, dynamic: bool, settings, horizon: int) -> None:
    admissible = metrics["admissible-windows"]
    selected = metrics["num-examples"]
    unique = metrics["unique-training-windows"]
    epochs = metrics["epochs-completed"]
    if not (0 <= admissible <= available and (dynamic or admissible == available)
            and selected == (min(admissible, cap) if cap is not None else admissible)
            and selected <= unique <= min(admissible, selected * epochs)
            and int(selected > 0) <= epochs <= (settings.local_epochs if selected else 0)):
        raise ValueError("Número de ventanas o épocas incompatible con el presupuesto de muestreo.")
    sampled = metrics["sampled-window-occurrences"]
    replacement = metrics["replacement-window-occurrences"]
    originals = selected * epochs
    targets = metrics["selected-target-occurrences"]
    repeated_targets = metrics["sampled-target-occurrences"] - targets
    if (replacement != (-selected) % settings.batch_size * epochs
            or sampled != originals + replacement
            or not originals <= targets <= originals * horizon
            or not replacement <= repeated_targets <= replacement * horizon
            or metrics["num-targets"] != (targets / epochs if epochs else 0)):
        raise ValueError("Soporte de entrenamiento incompatible con el muestreo y el reemplazo.")


def aggregate_evaluation(contents: list[RecordDict], weighted_by_key: str) -> MetricRecord:
    """Aggregate scaled errors by target, origin, and station without mixing supports."""
    records = [content["metrics"] for content in contents]
    total = sum(record["num-targets"] for record in records)
    sums = {key: sum(record[key] for record in records) for key in (
        "mase-sum", "mase-count", "rmsse-sum", "rmsse-count",
    )}
    metrics = {
        "num-clients": len(records),
        "num-examples": sum(record["num-examples"] for record in records),
        "num-targets": total,
        "mase-count": sums["mase-count"],
        "rmsse-count": sums["rmsse-count"],
    }
    for key in ("num-target-slots", "missing-targets", "missing-context-values", "signed-error-sum", "negative-predictions"):
        if all(key in record for record in records):
            metrics[key] = sum(record[key] for record in records)
    if "num-target-slots" in metrics:
        metrics["target-coverage"] = total / metrics["num-target-slots"]
        if "negative-predictions" in metrics:
            metrics["negative-prediction-fraction"] = metrics["negative-predictions"] / metrics["num-target-slots"]
    if "signed-error-sum" in metrics:
        metrics["bias"] = metrics["signed-error-sum"] / total
    probability_keys = {key for record in records for key in record
                        if key.startswith(("quantile-", "interval-", "wis-", "probabilistic-", "crps-"))}
    for key in sorted(probability_keys):
        if all(key in record for record in records):
            metrics[key] = sum(record[key] for record in records)
    if metrics.get("quantile-count", 0):
        metrics["mean-quantile-loss"] = metrics["quantile-loss-sum"] / metrics["quantile-count"]
    if metrics.get("wis-count", 0):
        metrics["wis"] = metrics["wis-sum"] / metrics["wis-count"]
    if metrics.get("crps-count", 0):
        metrics["crps"] = metrics["crps-sum"] / metrics["crps-count"]
    if all("scaled-crps-eligible" in record for record in records):
        eligible = [record for record in records if record["scaled-crps-eligible"]]
        complete = [record for record in eligible if record.get("scaled-crps-count") == record["num-targets"]]
        metrics.update({
            "scaled-crps-eligible-stations": len(eligible),
            "macro-scaled-crps-stations": len(complete),
            "scaled-crps-eligible-targets": sum(record["num-targets"] for record in eligible),
            "scaled-crps-count": sum(record["scaled-crps-count"] for record in complete),
        })
        if complete and len(complete) == len(eligible):
            metrics["macro-scaled-crps"] = float(np.mean([record["scaled-crps"] for record in complete]))
    for key in sorted(probability_keys):
        if key.startswith("interval-") and key.endswith("-count") and metrics.get(key, 0):
            prefix = key.removesuffix("-count")
            metrics[f"{prefix}-coverage"] = metrics[f"{prefix}-covered"] / metrics[key]
            metrics[f"{prefix}-mean-width"] = metrics[f"{prefix}-width-sum"] / metrics[key]
    granular = {key for record in records for key in record if key.startswith(("lead-", "block-"))}
    for key in sorted(granular):
        metrics[key] = sum(record.get(key, 0) for record in records)
        if key.startswith("lead-") and key.endswith("-scaled-error-sum"):
            prefix = key.removesuffix("-scaled-error-sum")
            support = sum(record.get(f"{prefix}-target-count", 0) for record in records)
            if support:
                pooled = metrics[key] / support
                metrics[prefix] = math.sqrt(pooled) if prefix.endswith("-rmsse") else pooled
    for metric in ("mase", "rmsse"):
        for suffix in ("scaled-error-sum", "target-count"):
            key = f"{metric}-{suffix}"
            metrics[key] = sum(record.get(key, 0) for record in records)
        if metrics[f"{metric}-target-count"]:
            pooled = metrics[f"{metric}-scaled-error-sum"] / metrics[f"{metric}-target-count"]
            metrics[f"pooled-{metric}"] = pooled if metric == "mase" else math.sqrt(pooled)
        if sums[f"{metric}-count"]:
            metrics[f"mean-{metric}"] = sums[f"{metric}-sum"] / sums[f"{metric}-count"]
        valid = [record.get(metric, record[f"{metric}-sum"] / record[f"{metric}-count"])
                 for record in records if record[f"{metric}-count"]]
        metrics[f"macro-{metric}-stations"] = len(valid)
        if valid:
            metrics[f"macro-{metric}"] = float(np.mean(valid))
    return MetricRecord(metrics)


def coverage_metrics(contents: list[RecordDict], manifest: dict) -> dict:
    robust = [content for content, client in zip(contents, manifest["clients"], strict=True) if client.get("robust_evaluation", False)]
    if not robust or len(robust) == len(contents):
        return {"robust-num-clients": len(robust)}
    return {f"robust-{key}": value for key, value in aggregate_evaluation(robust, "num-examples").items()}


class AuditedStrategy(Strategy):
    """Require the full declared station panel and select weights on validation only."""

    def __init__(self, settings: RunSettings, manifest: dict, output: Path, tracker: RunTracker | None = None):
        self.settings, self.manifest, self.output = settings, manifest, output
        self.tracker = tracker
        count = len(manifest["clients"])
        kwargs = {
            "fraction_train": 1.0, "fraction_evaluate": 1.0,
            "min_train_nodes": count, "min_evaluate_nodes": count, "min_available_nodes": count,
            "train_metrics_aggr_fn": aggregate_training,
            "evaluate_metrics_aggr_fn": aggregate_evaluation,
        }
        if manifest.get("schema_version", 4) >= 5:
            kwargs["weighted_by_key"] = "num-targets"
        self.delegate = (
            FedProx(**kwargs, proximal_mu=settings.proximal_mu)
            if settings.strategy == "fedprox" else FedAvg(**kwargs)
        )
        self.current_arrays: ArrayRecord | None = None
        self.best_arrays: ArrayRecord | None = None
        self.best_score, self.best_round = float("inf"), -1
        self.history: list[dict] = []
        self.station_history: list[dict] = []
        self.runtime_nodes: dict[int, int] = {}
        self.validation_support: dict[int, tuple[int, int]] = {}
        self.trained_targets = 0

    def summary(self) -> None:
        self.delegate.summary()

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        report_progress("train", f"round={server_round}/{self.settings.num_server_rounds} "
                        f"waiting_clients={len(self.manifest['clients'])}; clientes gestionados por Ray")
        self.current_arrays = arrays
        lower, cutoff = round_bounds(self.manifest["geometry"], self.settings, server_round)
        if cutoff is not None:
            config = ConfigRecord(dict(config) | {"available-through-ns": cutoff.value})
            if lower is not None:
                config["available-from-ns"] = lower.value
        return self._runtime_messages(self.delegate.configure_train(server_round, arrays, config, grid))

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        self.current_arrays = arrays
        # No validation information is revealed until the fold's training
        # stream is complete. In particular it cannot select an earlier-round
        # state which would then be backdated into a subsequent arrival stream.
        if (config.get("split", "validation") == "validation"
                and self.progressive and server_round < self.settings.num_server_rounds):
            return []
        report_progress(str(config.get("split", "validation")), f"round={server_round} "
                        f"waiting_clients={len(self.manifest['clients'])}")
        return self._runtime_messages(self.delegate.configure_evaluate(server_round, arrays, config, grid))

    @property
    def progressive(self) -> bool:
        return bool(self.settings.arrival_period_hours
                    and self.manifest["geometry"].get("split_mode") == "global")

    def _runtime_messages(self, messages: Iterable[Message]) -> list[Message]:
        messages = list(messages)
        filtered = "runtime_selection" in self.manifest
        if filtered:
            available = {message.metadata.dst_node_id for message in messages}
            if not self.runtime_nodes:
                count = len(self.manifest["clients"])
                if len(available) < count:
                    raise RuntimeError("No hay suficientes nodos para el panel efectivo.")
                self.runtime_nodes = {node: partition for partition, node in enumerate(sorted(available)[:count])}
            if not self.runtime_nodes.keys() <= available:
                raise RuntimeError("Panel incompleto: falta un nodo del panel efectivo.")
        result = []
        for message in messages:
            node = message.metadata.dst_node_id
            if filtered and node not in self.runtime_nodes:
                continue
            extra = {"log-dir": str(self.output / "clients")}
            if filtered:
                extra.update({
                "manifest": self.settings.manifest, "partition-id": self.runtime_nodes[node],
                })
            config = ConfigRecord(dict(message.content["config"]) | extra)
            result.append(Message(content=RecordDict(dict(message.content) | {"config": config}),
                                  dst_node_id=node, message_type=message.metadata.message_type))
        return result

    def _check_panel(self, replies: Iterable[Message], split: str, server_round: int = 0) -> list[Message]:
        replies = list(replies)
        expected = set(range(len(self.manifest["clients"])))
        seen = set()
        for reply in replies:
            if reply.has_error():
                raise RuntimeError(f"Falló una estación: {reply.error.reason}")
            metrics = reply.content["metrics"]
            partition = metrics["partition-id"]
            if partition not in expected or partition in seen:
                raise ValueError(f"Identidad de cliente incorrecta o repetida: {partition}")
            seen.add(partition)
            client = self.manifest["clients"][partition]
            geometry = self.manifest["geometry"]
            horizon = geometry["horizon"]
            selection_half = split == "validation" and calibration_enabled(self.settings)
            dynamic = selection_half or split == "train" and (
                self.progressive
                or self.settings.training_data_availability_threshold != "0.0"
                or self.settings.training_availability != "{}")
            available = client[f"num_{split}_windows"]
            cap = geometry.get("max_train_windows") if split == "train" else None
            limit = min(available, cap) if cap is not None else available
            if (not 0 <= metrics["num-examples"] <= limit or (not dynamic and metrics["num-examples"] != limit)):
                raise ValueError(f"Número de ventanas incoherente para {client['station']}.")
            if split == "train" and self.progressive:
                lower, cutoff = round_bounds(self.manifest["geometry"], self.settings, server_round)
                if metrics.get("available-through-ns") != cutoff.value:
                    raise ValueError("El cliente no respeta el reloj de llegada.")
                if metrics.get("available-from-ns") != (lower.value if lower is not None else None):
                    raise ValueError("El cliente no respeta el bloque de llegada.")
                if metrics["num-examples"]:
                    first = metrics.get("train-first-target-ns")
                    last = metrics.get("train-last-target-ns")
                    if first is None or last is None or first > last:
                        raise ValueError("El cliente no acredita el intervalo de targets de entrenamiento.")
                    if last >= cutoff.value:
                        raise ValueError("Un target de entrenamiento aún no ha llegado.")
                    if lower is not None and first < lower.value:
                        raise ValueError("Un target de entrenamiento pertenece a un bloque anterior.")
            if any(not np.isfinite(value) for value in metrics.values()):
                raise ValueError("Una estación devolvió métricas no finitas.")
            if split == "train":
                if "admissible-windows" in metrics:
                    _check_training_sampling(metrics, available, cap, dynamic, self.settings, horizon)
                arrays = reply.content["arrays"].to_numpy_ndarrays()
                if any(not np.isfinite(array).all() for array in arrays):
                    raise ValueError("Una estación devolvió pesos no finitos.")
            elif selection_half:
                examples, targets = metrics["num-examples"], metrics["num-targets"]
                maximum = client.get("num_validation_targets", available * horizon)
                support = (examples, targets)
                if (not 0 < examples <= targets <= min(examples * horizon, maximum)
                        or self.validation_support.setdefault(partition, support) != support):
                    raise ValueError("Soporte de selección incoherente en la primera mitad de validación.")
            elif metrics["num-targets"] != client.get(f"num_{split}_targets", metrics["num-examples"] * horizon):
                raise ValueError("Número de targets incoherente.")
            if split == "train" and self.manifest.get("schema_version", 4) >= 5:
                support = metrics["num-targets"]
                if (not metrics["num-examples"] <= support <= min(limit * horizon, client["num_train_targets"])
                        or (not dynamic and limit == available and support != client["num_train_targets"])):
                    raise ValueError("Número de targets de entrenamiento incoherente.")
        if seen != expected:
            raise RuntimeError(f"Panel incompleto; faltan estaciones {sorted(expected - seen)}")
        # Ray completion order must not change floating-point aggregation order.
        return sorted(replies, key=lambda reply: reply.content["metrics"]["partition-id"])

    def _record(self, server_round: int, split: str, replies: list, metrics: MetricRecord) -> None:
        self.history.append({"round": server_round, "split": split, **dict(metrics)})
        if self.tracker is not None:
            self.tracker.log_metrics(split, dict(metrics), server_round)
        for reply in replies:
            local_metrics = dict(reply.content["metrics"])
            partition = local_metrics["partition-id"]
            row = {
                "round": server_round, "split": split,
                "station": self.manifest["clients"][partition]["station"], **local_metrics,
            }
            for name in ("mase", "rmsse"):
                if row.get(f"{name}-count"):
                    row[f"mean-{name}"] = row[f"{name}-sum"] / row[f"{name}-count"]
            self.station_history.append(row)
        pd.DataFrame(self.history).to_csv(self.output / "round_metrics.csv", index=False)
        pd.DataFrame(self.station_history).to_csv(self.output / "station_metrics.csv", index=False)

    def _save(self, path: Path, arrays: ArrayRecord, server_round: int) -> None:
        torch.save(checkpoint_payload(
            self.manifest["geometry"], self.settings, arrays.to_torch_state_dict(), server_round,
            state_role="training" if path.name == "last_model.pt" else "selected",
        ), path)

    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        replies = self._check_panel(replies, "train", server_round)
        active = [reply for reply in replies if reply.content["metrics"]["num-examples"] > 0]
        self.trained_targets += sum(reply.content["metrics"].get("num-targets", reply.content["metrics"]["num-examples"]) for reply in active)
        if server_round == self.settings.num_server_rounds and not self.trained_targets:
            raise ValueError("No hay ventanas admisibles de entrenamiento en ninguna ronda del fold.")
        if not active:
            if self.current_arrays is None:
                raise RuntimeError("No hay estado global para la ronda sin datos.")
            metrics = aggregate_training([reply.content for reply in replies], "num-examples")
            self._save(self.output / "last_model.pt", self.current_arrays, server_round)
            self._record(server_round, "train", replies, metrics)
            return self.current_arrays, metrics
        if self.settings.model in GRANITE:
            if self.current_arrays is None:
                raise RuntimeError("Falta el estado global anterior para agregar adaptadores.")
            states = [reply.content["arrays"].to_torch_state_dict() for reply in active]
            weight_key = "num-targets" if self.manifest.get("schema_version", 4) >= 5 else "num-examples"
            counts = [reply.content["metrics"][weight_key] for reply in active]
            previous = self.current_arrays.to_torch_state_dict()
            previous_rounds = {int(key.split(".")[1]) for key in previous if key.startswith("history.")}
            cumulative_round = max(previous_rounds, default=0) + 1 if self.settings.adaptation == "flora" else server_round
            state, diagnostics = aggregate_adapter_updates(states, counts, self.settings,
                previous=previous, server_round=cumulative_round)
            arrays = ArrayRecord(state)
            metrics = aggregate_training([reply.content for reply in replies], "num-examples")
            metrics.update(diagnostics)
        else:
            arrays, _ = self.delegate.aggregate_train(server_round, active)
            metrics = aggregate_training([reply.content for reply in replies], "num-examples")
        if arrays is None or metrics is None:
            raise RuntimeError("Flower no produjo pesos agregados.")
        self.current_arrays = arrays
        self._save(self.output / "last_model.pt", arrays, server_round)
        self._record(server_round, "train", replies, metrics)
        return arrays, metrics

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]):
        replies = list(replies)
        if not replies and self.progressive and server_round < self.settings.num_server_rounds:
            return None
        replies = self._check_panel(replies, "validation")
        # Calendar support and undefined scaled metrics legitimately differ by
        # station. Our audited reducer pools them without Flower's equal-key
        # restriction; parameter aggregation still uses official FedAvg/FedProx.
        metrics = aggregate_evaluation([reply.content for reply in replies], "num-examples")
        if metrics is None or self.current_arrays is None:
            raise RuntimeError("No hay evaluación o pesos globales.")
        metrics.update(coverage_metrics([reply.content for reply in replies], self.manifest))
        self._record(server_round, "validation", replies, metrics)
        score = selection_value(dict(metrics), self.settings)
        if score < self.best_score:
            self.best_score, self.best_round = score, server_round
            self.best_arrays = ArrayRecord(self.current_arrays.to_torch_state_dict())
            self._save(self.output / "best_model.pt", self.best_arrays, server_round)
            report_progress("checkpoint", f"selected_round={server_round} "
                            f"{self.settings.selection_metric}={score:.6g}")
        return metrics

    def evaluate_test(self, grid: Grid) -> MetricRecord:
        if self.best_arrays is None:
            raise RuntimeError("No se ha seleccionado ningún checkpoint por validación.")
        messages = self.configure_evaluate(
            self.best_round, self.best_arrays, ConfigRecord({"split": "test"}), grid
        )
        replies = self._check_panel(grid.send_and_receive(messages, timeout=3600), "test")
        metrics = aggregate_evaluation([reply.content for reply in replies], "num-examples")
        if metrics is None:
            raise RuntimeError("No se recibió el test federado.")
        metrics.update(coverage_metrics([reply.content for reply in replies], self.manifest))
        self._record(self.best_round, "test", replies, metrics)
        if self.tracker is not None:
            self.tracker.log_metrics("selection", {"round": self.best_round, "validation_score": self.best_score}, self.best_round)
        (self.output / "test_metrics.json").write_text(json.dumps({
            "selected_round": self.best_round,
            "selection_metric": f"validation_{self.settings.selection_metric}",
            "validation_score": self.best_score,
            "test": dict(metrics),
        }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return metrics
