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
        "num-examples": count,
        "train-mse-normalized": sum(
            record["train-mse-normalized"] * record[weighted_by_key] for record in records
        ) / count,
    }
    for key in ("train-seconds", "uplink-tensor-bytes", "downlink-tensor-bytes"):
        if all(key in record for record in records):
            metrics[key] = sum(record[key] for record in records)
    for key in ("trainable-parameters", "peak-process-rss-mib", "peak-cuda-allocated-mib"):
        if all(key in record for record in records):
            metrics[key] = max(record[key] for record in records)
    return MetricRecord(metrics)


def aggregate_evaluation(contents: list[RecordDict], weighted_by_key: str) -> MetricRecord:
    """Pool error sums; never average client RMSEs to obtain global RMSE."""
    records = [content["metrics"] for content in contents]
    total = sum(record["num-targets"] for record in records)
    sums = {key: sum(record[key] for record in records) for key in (
        "absolute-error-sum", "squared-error-sum",
        "baseline-absolute-error-sum", "baseline-squared-error-sum",
        "mase-sum", "mase-count", "rmsse-sum", "rmsse-count",
    )}
    metrics = {
        "num-clients": len(records),
        "num-examples": sum(record[weighted_by_key] for record in records),
        "num-targets": total,
        "mae": sums["absolute-error-sum"] / total,
        "rmse": math.sqrt(sums["squared-error-sum"] / total),
        "macro-mae": float(np.mean([record["mae"] for record in records])),
        "macro-rmse": float(np.mean([math.sqrt(record["mse"]) for record in records])),
        "baseline-mae": sums["baseline-absolute-error-sum"] / total,
        "baseline-rmse": math.sqrt(sums["baseline-squared-error-sum"] / total),
        "mase-count": sums["mase-count"],
        "rmsse-count": sums["rmsse-count"],
    }
    for metric in ("mase", "rmsse"):
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
    if not robust:
        return {"robust-num-clients": 0}
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

    def summary(self) -> None:
        self.delegate.summary()

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        report_progress("train", f"round={server_round}/{self.settings.num_server_rounds} "
                        f"waiting_clients={len(self.manifest['clients'])}; clientes gestionados por Ray")
        return self._runtime_messages(self.delegate.configure_train(server_round, arrays, config, grid))

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        self.current_arrays = arrays
        report_progress(str(config.get("split", "validation")), f"round={server_round} "
                        f"waiting_clients={len(self.manifest['clients'])}")
        return self._runtime_messages(self.delegate.configure_evaluate(server_round, arrays, config, grid))

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

    def _check_panel(self, replies: Iterable[Message], split: str) -> list[Message]:
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
            if metrics["num-examples"] != client[f"num_{split}_windows"]:
                raise ValueError(f"Número de ventanas incoherente para {client['station']}.")
            if any(not np.isfinite(value) for value in metrics.values()):
                raise ValueError("Una estación devolvió métricas no finitas.")
            if split == "train":
                arrays = reply.content["arrays"].to_numpy_ndarrays()
                if any(not np.isfinite(array).all() for array in arrays):
                    raise ValueError("Una estación devolvió pesos no finitos.")
            elif metrics["num-targets"] != metrics["num-examples"] * self.manifest["geometry"]["horizon"]:
                raise ValueError("Número de targets incoherente.")
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
            report_progress(split, f"round={server_round} station={row['station']} "
                            f"windows={row['num-examples']} mae={row.get('mae', '-')} "
                            f"train_seconds={row.get('train-seconds', '-')}")
            if "mse" in row:
                row["rmse"] = math.sqrt(row["mse"])
                for name in ("mase", "rmsse"):
                    if row[f"{name}-count"]:
                        row[f"mean-{name}"] = row[f"{name}-sum"] / row[f"{name}-count"]
            self.station_history.append(row)
            if self.tracker is not None:
                self.tracker.log_metrics(split, row, server_round, partition=partition)
        pd.DataFrame(self.history).to_csv(self.output / "round_metrics.csv", index=False)
        pd.DataFrame(self.station_history).to_csv(self.output / "station_metrics.csv", index=False)

    def _save(self, path: Path, arrays: ArrayRecord, server_round: int) -> None:
        torch.save(checkpoint_payload(self.manifest["geometry"], self.settings, arrays.to_torch_state_dict(), server_round), path)

    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        replies = self._check_panel(replies, "train")
        if self.settings.model in GRANITE:
            if self.current_arrays is None:
                raise RuntimeError("Falta el estado global anterior para agregar adaptadores.")
            states = [reply.content["arrays"].to_torch_state_dict() for reply in replies]
            counts = [int(reply.content["metrics"]["num-examples"]) for reply in replies]
            state, diagnostics = aggregate_adapter_updates(states, counts, self.settings,
                previous=self.current_arrays.to_torch_state_dict(), server_round=server_round)
            arrays = ArrayRecord(state)
            metrics = aggregate_training([reply.content for reply in replies], "num-examples")
            metrics.update(diagnostics)
        else:
            arrays, metrics = self.delegate.aggregate_train(server_round, replies)
        if arrays is None or metrics is None:
            raise RuntimeError("Flower no produjo pesos agregados.")
        self.current_arrays = arrays
        self._save(self.output / "last_model.pt", arrays, server_round)
        self._record(server_round, "train", replies, metrics)
        return arrays, metrics

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]):
        replies = self._check_panel(replies, "validation")
        metrics = self.delegate.aggregate_evaluate(server_round, replies)
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
        metrics = self.delegate.aggregate_evaluate(self.best_round, replies)
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
