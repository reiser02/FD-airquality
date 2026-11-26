"""Audit and checkpoint an official Flower strategy without reimplementing FedAvg."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg, FedProx, Strategy

from fd_airquality.arrival import round_bounds, round_epoch_limit, round_validation_bounds
from fd_airquality.calibration import calibration_enabled
from fd_airquality.federated_adapters import aggregate_adapter_updates
from fd_airquality.models import GRANITE, checkpoint_payload
from fd_airquality.run_logging import report_progress
from fd_airquality.settings import RunSettings
from fd_airquality.task import selection_value
from fd_airquality.tracking import RunTracker
from fd_airquality.training_control import controls_enabled


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
    for key in ("candidate-windows", "availability-excluded-windows", "idle-client", "round-validation-excluded-windows",
                "round-validation-windows", "round-validation-fallback",
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


def _check_training_windows(metrics, available: int, cap: int | None, dynamic: bool,
                            epoch_limit: int) -> None:
    admissible = metrics["admissible-windows"]
    selected = metrics["num-examples"]
    unique = metrics["unique-training-windows"]
    epochs = metrics["epochs-completed"]
    if not (0 <= admissible <= available and (dynamic or admissible == available)
            and selected == (min(admissible, cap) if cap is not None else admissible)
            and selected <= unique <= min(admissible, selected * epochs)
            and int(selected > 0) <= epochs <= (epoch_limit if selected else 0)):
        raise ValueError("Número de ventanas o épocas incompatible con el presupuesto de muestreo.")


def _check_training_sampling(metrics, settings, horizon: int) -> None:
    selected, epochs = metrics["num-examples"], metrics["epochs-completed"]
    sampled = metrics["sampled-window-occurrences"]
    replacement = metrics["replacement-window-occurrences"]
    originals = selected * epochs
    targets = metrics["selected-target-occurrences"]
    repeated_targets = metrics["sampled-target-occurrences"] - targets
    expected_replacements = (-selected) % settings.batch_size * epochs if settings.pad_last_batch else 0
    if (replacement != expected_replacements
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

    def __init__(self, settings: RunSettings, manifest: dict, output: Path, tracker: RunTracker | None = None,
                 *, gateway_members: list[list[int]] | None = None):
        self.settings, self.manifest, self.output = settings, manifest, output
        self.tracker = tracker
        self.gateway_members = gateway_members
        count = len(manifest["clients"]) if gateway_members is None else len(gateway_members)
        self.client_count = count
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
        self.gateway_history: list[dict] = []
        self.runtime_nodes: dict[int, int] = {}
        self.validation_support: dict[int, tuple[int, int]] = {}
        self.trained_targets = 0

    def summary(self) -> None:
        self.delegate.summary()

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        geometry = self.manifest["geometry"]
        report_progress("train", f"round={server_round}/{self.settings.num_server_rounds} "
                        f"max_epochs={round_epoch_limit(geometry, self.settings, server_round)} "
                        f"waiting_clients={self.client_count}; clientes gestionados por Ray")
        self.current_arrays = arrays
        lower, cutoff = round_bounds(geometry, self.settings, server_round)
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
                        f"waiting_clients={self.client_count}")
        return self._runtime_messages(self.delegate.configure_evaluate(server_round, arrays, config, grid))

    @property
    def progressive(self) -> bool:
        return bool(self.settings.arrival_period_hours
                    and self.manifest["geometry"].get("split_mode") == "global")

    def _runtime_messages(self, messages: Iterable[Message]) -> list[Message]:
        messages = list(messages)
        filtered = self.gateway_members is not None or "runtime_selection" in self.manifest
        if filtered:
            available = {message.metadata.dst_node_id for message in messages}
            count = self.client_count
            if self.gateway_members is not None and len(available) != count:
                raise RuntimeError(f"Flower debe proporcionar {count} nodos para la topología de gateways.")
            if not self.runtime_nodes:
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
            contents = dict(message.content)
            if filtered:
                partition = self.runtime_nodes[node]
                extra.update({"manifest": self.settings.manifest, "partition-id": partition})
                if self.gateway_members is not None:
                    members = self.gateway_members[partition]
                    extra["station-partitions"] = members
                    if not members:
                        del contents["arrays"]
            config = ConfigRecord(dict(message.content["config"]) | extra)
            result.append(Message(content=RecordDict(contents | {"config": config}),
                                  dst_node_id=node, message_type=message.metadata.message_type))
        return result

    def _check_round_validation(self, metrics, geometry, server_round) -> None:
        windows = metrics.get("round-validation-windows", 0)
        lower, upper = metrics.get("round-validation-from-ns"), metrics.get("round-validation-through-ns")
        if windows == 0 and lower is None and upper is None:
            return
        nominal = round_validation_bounds(geometry, self.settings, server_round)
        extended = round_validation_bounds(geometry, self.settings, server_round, allow_extension=True)
        if (nominal is None or type(windows) is not int or windows < 1
                or type(lower) is not int or type(upper) is not int or upper != nominal[1].value
                or not extended[0].value <= lower <= nominal[0].value or not metrics["num-examples"]):
            raise ValueError("El conjunto de validación no respeta el bloque de la ronda.")
        if metrics["train-last-target-ns"] >= lower:
            raise ValueError("El entrenamiento utiliza targets del conjunto de validación de la ronda.")

    def _check_round_validation_budget(self, metrics, geometry, server_round) -> None:
        if not metrics["num-examples"] or not controls_enabled(self.settings):
            return
        epochs = round_epoch_limit(geometry, self.settings, server_round)
        if epochs == 1:
            return
        windows = metrics["round-validation-windows"]
        if metrics["round-validation-fallback"] != int(not windows):
            raise ValueError("El estado de validación no coincide con el soporte declarado.")
        if windows:
            if metrics.get("training-validation-windows") != windows:
                raise ValueError("La evaluación de validación no usa el conjunto declarado.")
        elif (metrics["epochs-completed"] != min(epochs, self.settings.round_validation_fallback_epochs)
              or "training-validation-windows" in metrics):
            raise ValueError("El cliente excede o no respeta el presupuesto sin validación.")

    def _check_panel(self, replies: Iterable[Message], split: str, server_round: int = 0) -> list[Message]:
        replies = list(replies)
        expected = set(range(self.client_count))
        seen = set()
        for reply in replies:
            if reply.has_error():
                raise RuntimeError(f"Falló un cliente: {reply.error.reason}")
            metrics = reply.content["metrics"]
            partition = metrics["partition-id"]
            if partition not in expected or partition in seen:
                raise ValueError(f"Identidad de cliente incorrecta o repetida: {partition}")
            seen.add(partition)
            if self.gateway_members is None:
                self._check_station(metrics, partition, split, server_round)
                if split == "train" and "admissible-windows" in metrics:
                    _check_training_sampling(metrics, self.settings, self.manifest["geometry"]["horizon"])
                    if self.progressive:
                        self._check_round_validation_budget(metrics, self.manifest["geometry"], server_round)
            else:
                if self.runtime_nodes.get(reply.metadata.src_node_id) != partition:
                    raise ValueError(f"La identidad del nodo no coincide con el gateway {partition}.")
                self._check_gateway(reply.content, partition, split, server_round)
            if split == "train" and (self.gateway_members is None or metrics["num-examples"]):
                arrays = reply.content["arrays"].to_numpy_ndarrays()
                if any(not np.isfinite(array).all() for array in arrays):
                    raise ValueError("Un cliente devolvió pesos no finitos.")
        if seen != expected:
            raise RuntimeError(f"Panel incompleto; faltan clientes {sorted(expected - seen)}")
        # Ray completion order must not change floating-point aggregation order.
        return sorted(replies, key=lambda reply: reply.content["metrics"]["partition-id"])

    def _check_station(self, metrics, partition: int, split: str, server_round: int) -> None:
        client = self.manifest["clients"][partition]
        geometry = self.manifest["geometry"]
        horizon = geometry["horizon"]
        selection_half = split == "validation" and calibration_enabled(self.settings)
        dynamic = selection_half or split == "train" and (
            self.progressive or self.settings.training_data_availability_threshold != "0.0"
            or self.settings.training_availability != "{}")
        available = client[f"num_{split}_windows"]
        cap = geometry.get("max_train_windows") if split == "train" else None
        limit = min(available, cap) if cap is not None else available
        examples = metrics["num-examples"]
        if not 0 <= examples <= limit or (not dynamic and examples != limit):
            raise ValueError(f"Número de ventanas incoherente para {client['station']}.")
        if split == "train" and self.progressive:
            lower, cutoff = round_bounds(geometry, self.settings, server_round)
            if metrics.get("available-through-ns") != cutoff.value:
                raise ValueError("El cliente no respeta el reloj de llegada.")
            if metrics.get("available-from-ns") != (lower.value if lower is not None else None):
                raise ValueError("El cliente no respeta el bloque de llegada.")
            if examples:
                first, last = metrics.get("train-first-target-ns"), metrics.get("train-last-target-ns")
                if first is None or last is None or first > last:
                    raise ValueError("El cliente no acredita el intervalo de targets de entrenamiento.")
                if last >= cutoff.value:
                    raise ValueError("Un target de entrenamiento aún no ha llegado.")
                if lower is not None and first < lower.value:
                    raise ValueError("Un target de entrenamiento pertenece a un bloque anterior.")
            self._check_round_validation(metrics, geometry, server_round)
        if any(not np.isfinite(value) for value in metrics.values()):
            raise ValueError("Una estación devolvió métricas no finitas.")
        if split == "train":
            if "admissible-windows" in metrics:
                epochs = round_epoch_limit(geometry, self.settings, server_round)
                _check_training_windows(metrics, available, cap, dynamic, epochs)
            if self.manifest.get("schema_version", 4) >= 5:
                support = metrics["num-targets"]
                if (not examples <= support <= min(limit * horizon, client["num_train_targets"])
                        or (not dynamic and limit == available and support != client["num_train_targets"])):
                    raise ValueError("Número de targets de entrenamiento incoherente.")
        elif selection_half:
            maximum = client.get("num_validation_targets", available * horizon)
            support = (examples, metrics["num-targets"])
            if (not 0 < examples <= support[1] <= min(examples * horizon, maximum)
                    or self.validation_support.setdefault(partition, support) != support):
                raise ValueError("Soporte de selección incoherente en la primera mitad de validación.")
        elif metrics["num-targets"] != client.get(f"num_{split}_targets", examples * horizon):
            raise ValueError("Número de targets incoherente.")

    def _check_gateway(self, content: RecordDict, partition: int, split: str, server_round: int) -> None:
        members = self.gateway_members[partition]
        metrics = content["metrics"]
        if (set(content.metric_records) != {"metrics", *(f"station-{station}" for station in members)}
                or metrics.get("num-stations") != len(members)
                or any(not np.isfinite(value) for value in metrics.values())):
            raise ValueError(f"Panel o métricas incorrectos para el gateway {partition}.")
        rows = []
        for station in members:
            row = content[f"station-{station}"]
            if row["partition-id"] != station:
                raise ValueError(f"Estación incorrecta para el gateway {partition}.")
            self._check_station(row, station, split, server_round)
            rows.append(row)
        if split == "train":
            epochs = metrics["epochs-completed"]
            for row in rows:
                if (row["epochs-completed"] != (epochs if row["num-examples"] else 0)
                        or row["num-targets"] != row["selected-target-occurrences"] / (epochs or 1)):
                    raise ValueError("Soporte local incompatible con las épocas del gateway.")
            for key in ("num-examples", "admissible-windows", "unique-training-windows",
                        "selected-target-occurrences", "round-validation-windows"):
                if metrics[key] != sum(row[key] for row in rows):
                    raise ValueError(f"La suma de las estaciones no coincide con el gateway: {key}.")
            if ("arrays" in content) != bool(metrics["num-examples"]):
                raise ValueError("El gateway debe actualizar pesos solo cuando entrenó con datos.")
            _check_training_sampling(metrics, self.settings, self.manifest["geometry"]["horizon"])
            if self.progressive:
                self._check_round_validation_budget(metrics, self.manifest["geometry"], server_round)

    def _station_contents(self, replies: list[Message]) -> list[RecordDict]:
        if self.gateway_members is None:
            return [reply.content for reply in replies]
        stations = {}
        for reply in replies:
            gateway = reply.content["metrics"]["partition-id"]
            for station in self.gateway_members[gateway]:
                metrics = MetricRecord(dict(reply.content[f"station-{station}"]) | {"gateway-id": gateway})
                stations[station] = RecordDict({"metrics": metrics})
        return [stations[station] for station in sorted(stations)]

    def _record(self, server_round: int, split: str, replies: list, metrics: MetricRecord) -> None:
        if self.gateway_members is not None:
            activity_key = "num-examples" if split == "train" else "num-stations"
            metrics.update({"num-clients": self.client_count, "num-stations": len(self.manifest["clients"]),
                            "active-gateways": sum(reply.content["metrics"][activity_key] > 0
                                                   for reply in replies)})
            for reply in replies:
                row = dict(reply.content["metrics"])
                self.gateway_history.append({"round": server_round, "split": split,
                                             "gateway-id": row.pop("partition-id"), **row})
            pd.DataFrame(self.gateway_history).to_csv(self.output / "gateway_metrics.csv", index=False)
        self.history.append({"round": server_round, "split": split, **dict(metrics)})
        if self.tracker is not None:
            self.tracker.log_metrics(split, dict(metrics), server_round)
        for content in self._station_contents(replies):
            local_metrics = dict(content["metrics"])
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
            # Flower requires identical metric keys across replies. Validation
            # diagnostics are optional; retain the full records for our audit,
            # and send only the common metrics and model update to the delegate.
            keys = set.intersection(*(set(reply.content["metrics"]) for reply in active))
            projected = []
            for reply in active:
                message = copy(reply)
                metrics = reply.content["metrics"]
                message.content = RecordDict({
                    "arrays": reply.content["arrays"],
                    "metrics": MetricRecord({key: value for key, value in metrics.items() if key in keys}),
                })
                projected.append(message)
            arrays, _ = self.delegate.aggregate_train(server_round, projected)
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
        contents = self._station_contents(replies)
        metrics = aggregate_evaluation(contents, "num-examples")
        if metrics is None or self.current_arrays is None:
            raise RuntimeError("No hay evaluación o pesos globales.")
        metrics.update(coverage_metrics(contents, self.manifest))
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
        contents = self._station_contents(replies)
        metrics = aggregate_evaluation(contents, "num-examples")
        if metrics is None:
            raise RuntimeError("No se recibió el test federado.")
        metrics.update(coverage_metrics(contents, self.manifest))
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
