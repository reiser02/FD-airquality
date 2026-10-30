import json

import numpy as np
import pandas as pd
import pytest
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.supercore.task_identity import TaskIdentity

from fd_airquality.client_app import build_client_app
from fd_airquality.data import WindowDataset, load_client_data
from fd_airquality.server_app import run_server
from fd_airquality.settings import RunSettings
from fd_airquality.strategy import AuditedStrategy, aggregate_evaluation
from fd_airquality.task import configure_torch, make_model, train_local


@pytest.fixture(autouse=True)
def flower_runtime_identity(monkeypatch):
    # The real runtime supplies this process identity before dispatching messages.
    monkeypatch.setattr(TaskIdentity, "_run_id", 1)
    monkeypatch.setattr(TaskIdentity, "_task_id", 1)
    monkeypatch.setattr(TaskIdentity, "_node_id", 0)


class LocalTestGrid:
    """Transport-only double; Flower still performs scheduling and aggregation."""

    def __init__(self, settings, count):
        self.app = build_client_app()
        self.contexts = {
            node + 10: Context(
                run_id=1, node_id=node + 10,
                node_config={"partition-id": node, "num-partitions": count},
                state=RecordDict(), run_config=settings.flower_config(),
            ) for node in range(count)
        }
        self.requests = []

    def get_node_ids(self):
        return list(self.contexts)

    def send_and_receive(self, messages, timeout=None):
        replies = []
        for message in messages:
            self.requests.append((message.metadata.message_type, dict(message.content["config"])))
            replies.append(self.app(message, self.contexts[message.metadata.dst_node_id]))
        return replies


@pytest.mark.parametrize("strategy", ["fedavg", "fedprox"])
@pytest.mark.parametrize("model", ["mlp", "autoencoder", "darts-nlinear", "flowstate-r1"])
def test_full_flower_strategy_selects_validation_then_tests_once(prepared, tmp_path, strategy, model, monkeypatch):
    if model == "flowstate-r1":
        from test_models import DummyBackbone

        monkeypatch.setattr("fd_airquality.models._load_granite", lambda *_: DummyBackbone())
    path, manifest = prepared
    settings = RunSettings(
        manifest=str(path), output_dir=str(tmp_path / "reports"),
        num_server_rounds=2, hidden_size=8, strategy=strategy, batch_size=32, model=model,
        lora_targets="proj" if model == "flowstate-r1" else "auto",
    )
    grid = LocalTestGrid(settings, len(manifest["clients"]))
    output = run_server(grid, settings)
    results = json.loads((output / "test_metrics.json").read_text())
    assert results["selection_metric"] == "validation_macro-mase"
    assert 0 <= results["selected_round"] <= 2
    assert results["test"]["num-clients"] == 2
    assert results["test"]["num-targets"] == 2 * 5 * 2
    checkpoint = torch.load(output / "best_model.pt", weights_only=True)
    assert checkpoint["round"] == results["selected_round"]
    assert checkpoint["hidden_size"] == 8
    assert checkpoint["model"]["name"] == model
    if model == "flowstate-r1":
        assert len(checkpoint["model"]["revision"]) == 40
        assert not any("backbone" in key for key in checkpoint["model_state_dict"])
    test_requests = [config for kind, config in grid.requests if config.get("split") == "test"]
    assert len(test_requests) == 2  # one test request per station, not one per round
    assert all(config["server-round"] == results["selected_round"] for config in test_requests)
    metrics = pd.read_csv(output / "round_metrics.csv")
    assert len(metrics[metrics.split == "train"]) == 2
    assert len(metrics[metrics.split == "validation"]) == 3
    assert len(metrics[metrics.split == "test"]) == 1
    per_station = pd.read_csv(output / "station_metrics.csv")
    assert len(per_station[per_station.split == "test"]) == 2
    assert (output / "last_model.pt").is_file()


def test_actual_flower_fedavg_weights_by_valid_window_count(tmp_path):
    manifest = {
        "geometry": {"context": 6, "horizon": 2},
        "clients": [
            {"station": "a", "num_train_windows": 1},
            {"station": "b", "num_train_windows": 3},
        ]
    }
    strategy = AuditedStrategy(RunSettings(), manifest, tmp_path)
    request = Message(content=RecordDict(), dst_node_id=10, message_type="train")
    replies = []
    for partition, weight, value in ((0, 1, 2.0), (1, 3, 10.0)):
        replies.append(Message(content=RecordDict({
            "arrays": ArrayRecord([np.array([value], dtype=np.float32)]),
            "metrics": MetricRecord({
                "partition-id": partition, "num-examples": weight, "train-mse-normalized": 1.0,
            }),
        }), reply_to=request))
    arrays, _ = strategy.aggregate_train(1, replies)
    np.testing.assert_allclose(arrays.to_numpy_ndarrays()[0], [8.0])
    with pytest.raises(RuntimeError, match="Panel incompleto"):
        strategy.aggregate_train(2, replies[:1])
    with pytest.raises(ValueError, match="repetida"):
        strategy.aggregate_train(2, [replies[0], replies[0]])


def test_global_rmse_is_pooled_not_mean_client_rmse():
    contents = []
    for count, mae, mse in ((1, 1.0, 1.0), (3, 3.0, 9.0)):
        contents.append(RecordDict({"metrics": MetricRecord({
            "num-examples": count, "num-targets": count, "mae": mae, "mse": mse,
            "absolute-error-sum": mae * count, "squared-error-sum": mse * count,
            "baseline-absolute-error-sum": 2.0 * count,
            "baseline-squared-error-sum": 4.0 * count,
            "mase-sum": 0.0, "mase-count": 0, "rmsse-sum": 0.0, "rmsse-count": 0,
        })}))
    metrics = aggregate_evaluation(contents, "num-examples")
    assert metrics["mae"] == 2.5
    assert metrics["rmse"] == pytest.approx(np.sqrt(7))
    assert metrics["macro-rmse"] == 2.0
    assert "mean-mase" not in metrics
    assert metrics["mase-count"] == 0


def test_macro_scaled_errors_weight_stations_not_origins():
    contents = []
    for origins, scaled in ((1, 1.), (9, 3.)):
        contents.append(RecordDict({"metrics": MetricRecord({
            "num-examples": origins, "num-targets": origins, "mae": scaled, "mse": scaled**2,
            "absolute-error-sum": origins * scaled, "squared-error-sum": origins * scaled**2,
            "baseline-absolute-error-sum": origins * scaled, "baseline-squared-error-sum": origins * scaled**2,
            "mase-sum": origins * scaled, "mase-count": origins, "mase": scaled,
            "rmsse-sum": origins * scaled, "rmsse-count": origins, "rmsse": scaled,
        })}))
    metrics = aggregate_evaluation(contents, "num-examples")
    assert metrics["mean-mase"] == 2.8
    assert metrics["macro-mase"] == metrics["macro-rmsse"] == 2.


def test_local_training_is_reproducible_and_proximal_penalty_changes_updates(prepared):
    path, manifest = prepared
    _, artifact = load_client_data(str(path), 0)
    settings = RunSettings(hidden_size=8, local_epochs=2, batch_size=16)
    dataset = WindowDataset(artifact, "train", 6, 2)

    def trained(mu):
        device = configure_torch(settings, seed=17)
        model = make_model(manifest["geometry"], settings)
        train_local(model, dataset, settings, device, 17, proximal_mu=mu)
        return model.state_dict()

    one, two, proximal = trained(0), trained(0), trained(1000)
    for name in one:
        torch.testing.assert_close(one[name], two[name], rtol=0, atol=0)
    assert any(not torch.equal(one[name], proximal[name]) for name in one)


def test_client_sends_no_series_or_normalization_statistics(prepared):
    path, manifest = prepared
    settings = RunSettings(manifest=str(path), hidden_size=8)
    grid = LocalTestGrid(settings, 2)
    model = make_model(manifest["geometry"], settings)
    reply, = grid.send_and_receive([Message(content=RecordDict({
        "arrays": ArrayRecord(model.state_dict()), "config": ConfigRecord({"server-round": 1}),
    }), dst_node_id=10, message_type="train")])
    assert set(reply.content) == {"arrays", "metrics"}
    assert {"partition-id", "num-examples", "train-mse-normalized", "train-seconds", "trainable-parameters",
            "uplink-tensor-bytes", "downlink-tensor-bytes"} <= set(reply.content["metrics"])
    assert not {"mean", "scale", "values", "timestamps"} & set(reply.content["metrics"])


def test_changed_preparation_code_requires_preparation(prepared, monkeypatch):
    path, manifest = prepared
    settings = RunSettings(manifest=str(path), hidden_size=8)
    grid = LocalTestGrid(settings, 2)
    monkeypatch.setattr("fd_airquality.client_app.source_fingerprints", lambda: {"changed": "hash"})
    with pytest.raises(ValueError, match="Cambió el código local"):
        grid.send_and_receive([Message(content=RecordDict({
            "arrays": ArrayRecord(make_model(manifest["geometry"], settings).state_dict()),
            "config": ConfigRecord(),
        }), dst_node_id=10, message_type="evaluate")])


@pytest.mark.parametrize("model", ["ttm-r3", "patchtst-fm-r2", "flowstate-r1"])
@pytest.mark.parametrize("algorithm", ["lora", "ffa-lora", "flora", "lora-fair"])
def test_all_foundation_algorithms_complete_flower_rounds_and_checkpoint_history(global_prepared, tmp_path, monkeypatch, model, algorithm):
    from test_models import DummyBackbone

    from fd_airquality.models import restore_checkpoint

    monkeypatch.setattr("fd_airquality.models._load_granite", lambda *_: DummyBackbone())
    path, manifest = global_prepared
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"), model=model,
                           adaptation=algorithm, lora_targets="proj", batch_size=32,
                           num_server_rounds=2, fair_iterations=5, mlflow_tracking_uri="disabled")
    output = run_server(LocalTestGrid(settings, len(manifest["clients"])), settings)
    result = json.loads((output / "test_metrics.json").read_text())
    assert result["selection_metric"] == "validation_macro-mase"
    assert result["test"]["macro-mase-stations"] == 2
    checkpoint = torch.load(output / "last_model.pt", weights_only=True)
    state = checkpoint["model_state_dict"]
    if algorithm == "flora":
        assert any(key.startswith("history.000001.") for key in state)
        assert any(key.startswith("history.000002.") for key in state)
        assert state["history.000001.proj.lora_A"].shape[0] == 2 * settings.lora_rank
    elif algorithm == "ffa-lora":
        assert not any(key.endswith("lora_A") for key in state)
        assert checkpoint["fixed_adapter_state"]
    restore_checkpoint(output / "best_model.pt")
    history = pd.read_csv(output / "round_metrics.csv")
    assert (history.query("split == 'train'")["uplink-tensor-bytes"] > 0).all()
    assert (history.query("split == 'train'")["server-aggregation-seconds"] > 0).all()


def test_server_discards_unusable_stations_before_any_training(selection_prepared, tmp_path, caplog):
    path, source = selection_prepared
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"), num_server_rounds=1,
                           mlflow_tracking_uri="disabled")
    grid = LocalTestGrid(settings, len(source["clients"]))
    output = run_server(grid, settings)
    result = json.loads((output / "test_metrics.json").read_text())
    assert result["test"]["num-clients"] == 1
    assert result["test"]["macro-mase-stations"] == 1
    assert set(pd.read_csv(output / "station_metrics.csv").station) == {"station-b"}
    assert pd.read_csv(output / "excluded_stations.csv").station.to_list() == ["station-a"]
    assert "station-a" in caplog.text
    assert len([config for kind, config in grid.requests if kind == "train"]) == 1
    assert len([config for kind, config in grid.requests if config.get("split") == "test"]) == 1
    assert all(config["partition-id"] == 0 for _, config in grid.requests)
    assert all(config["manifest"] != str(path) for _, config in grid.requests)
    assert len(json.loads(path.read_text())["clients"]) == 2


@pytest.mark.parametrize("selection_prepared", [("station-a", "station-b", "station-c")], indirect=True)
def test_effective_flower_partitions_are_stable_and_do_not_share_message_configuration(selection_prepared, tmp_path):
    path, source = selection_prepared
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"), num_server_rounds=2,
                           mlflow_tracking_uri="disabled")
    grid = LocalTestGrid(settings, len(source["clients"]))
    grid.contexts = dict(reversed(list(grid.contexts.items())))
    output = run_server(grid, settings)
    stations = pd.read_csv(output / "station_metrics.csv")
    assert set(stations.station) == {"station-b", "station-c"}
    assert set(stations["partition-id"]) == {0, 1}
    assert set(stations.loc[stations["partition-id"] == 0, "station"]) == {"station-b"}
    for stage in (1, 2):
        requests = [config for kind, config in grid.requests if kind == "train" and config["server-round"] == stage]
        assert sorted(config["partition-id"] for config in requests) == [0, 1]
    assert all(context.run_config["manifest"] == str(path) for context in grid.contexts.values())
