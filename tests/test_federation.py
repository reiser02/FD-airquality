import gc
import json
import weakref
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict

from fd_airquality.data import WindowDataset, load_client_data
from fd_airquality.models import restore_checkpoint
from fd_airquality.server_app import run_server
from fd_airquality.settings import RunSettings
from fd_airquality.strategy import AuditedStrategy, aggregate_evaluation, coverage_metrics
from fd_airquality.task import configure_torch, make_model, train_local


@pytest.mark.parametrize(("model", "strategy"), [
    ("mlp", "fedprox"), ("autoencoder", "fedavg"), ("nlinear", "fedavg"), ("flowstate-r1", "fedavg"),
])
def test_flower_rounds_write_metrics_and_checkpoints(prepared, tmp_path, flower_grid, mock_granite, model, strategy):
    path, manifest = prepared
    settings = RunSettings(
        manifest=str(path), output_dir=str(tmp_path / "reports"),
        num_server_rounds=2, hidden_size=8, strategy=strategy, batch_size=32, model=model,
        lora_targets="proj" if model == "flowstate-r1" else "auto",
        mlflow_tracking_uri="disabled", probabilistic=int(model == "nlinear"),
    )
    clients = manifest["clients"]
    output = run_server(flower_grid(settings, len(clients)), settings)
    results = json.loads((output / "test_metrics.json").read_text())
    assert results["test"]["num-clients"] == len(clients)
    assert results["test"]["num-targets"] == sum(client["num_test_targets"] for client in clients)
    checkpoint = torch.load(output / "best_model.pt", weights_only=True)
    assert 0 <= checkpoint["round"] == results["selected_round"] <= settings.num_server_rounds
    assert checkpoint["hidden_size"] == 8
    assert checkpoint["model"]["name"] == model
    if model == "flowstate-r1":
        assert len(checkpoint["model"]["revision"]) == 40
        assert set(checkpoint["model_state_dict"]) == {"adapter.proj.lora_A", "adapter.proj.lora_B"}
    metrics = pd.read_csv(output / "round_metrics.csv")
    assert metrics["split"].tolist() == ["validation", "train", "validation", "train", "validation", "test"]
    test = pd.read_csv(output / "station_metrics.csv").query("split == 'test'")
    assert test.station.tolist() == [client["station"] for client in clients]
    if model == "nlinear":
        assert results["test"]["quantile-crossings"] == 0
        assert results["test"]["quantile-raw-crossings"] == test["quantile-raw-crossings"].sum() > 0
    assert (output / "last_model.pt").is_file()


def test_flower_tests_once_with_weights_from_best_validation_round(prepared, tmp_path, flower_grid):
    path, manifest = prepared
    count = len(manifest["clients"])
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"), hidden_size=8,
                           num_server_rounds=2, mlflow_tracking_uri="disabled")
    grid = flower_grid(settings, count)
    app = grid.app
    evaluations = []

    def respond(message, context):
        reply = app(message, context)
        config = message.content["config"]
        split, stage = config.get("split"), config["server-round"]
        if split in {"validation", "test"}:
            evaluations.append((split, stage, message.content["arrays"].to_torch_state_dict()))
        if split == "validation":
            # The selection criterion prefers round 1 while RMSSE favors round 2.
            record = reply.content["metrics"]
            mase, rmsse = ((2., 3.), (1., 2.), (3., 1.))[stage]
            record.update({"mase": mase, "mase-sum": mase * record["mase-count"],
                           "rmsse": rmsse, "rmsse-sum": rmsse * record["rmsse-count"]})
        return reply

    grid.app = respond
    output = run_server(grid, settings)
    assert [(split, stage) for split, stage, _ in evaluations] == [
        (split, stage)
        for split, stage in (("validation", 0), ("validation", 1), ("validation", 2), ("test", 1))
        for _ in range(count)
    ]
    expected = evaluations[count][2]
    last = evaluations[2 * count][2]
    assert any(not torch.equal(expected[key], last[key]) for key in expected)
    checkpoint = torch.load(output / "best_model.pt", weights_only=True)
    result = json.loads((output / "test_metrics.json").read_text())
    assert result["selection_metric"] == "validation_macro-mase"
    assert result["selected_round"] == checkpoint["round"] == 1
    torch.testing.assert_close(checkpoint["model_state_dict"], expected, rtol=0, atol=0)
    for _, _, state in evaluations[-count:]:
        torch.testing.assert_close(state, expected, rtol=0, atol=0)


def test_server_tracks_one_run_with_aggregate_fair_metrics(prepared, tmp_path, mlflow_uri, flower_grid, mock_granite):
    from mlflow import MlflowClient

    path, manifest = prepared
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"), model="flowstate-r1",
                           adaptation="lora-fair", lora_targets="proj", fair_iterations=3,
                           num_server_rounds=1, mlflow_tracking_uri=mlflow_uri)
    output = run_server(flower_grid(settings, len(manifest["clients"])), settings)
    metadata = json.loads((output / "mlflow.json").read_text())
    client = MlflowClient(tracking_uri=mlflow_uri)
    run, = client.search_runs([metadata["experiment_id"]])
    assert run.info.run_id == metadata["run_id"]
    assert run.info.status == "FINISHED"
    assert run.data.params["num_clients"] == "2"
    candidate = "FlowState-R1 | federado | LoRA-FAIR | FedAvg"
    assert run.data.tags["candidate"] == candidate
    assert run.info.run_name.startswith(f"{candidate} | PM25 | ")
    assert not {"backtest", "fold"} & run.data.tags.keys()
    assert run.data.params["fair_iterations"] == "3"
    assert run.data.params["strategy"] == "fedavg"
    assert "proximal_mu" not in run.data.params
    assert {"test/macro-mase", "test/macro-rmsse", "train/server-aggregation-seconds"} <= run.data.metrics.keys()
    history = client.get_metric_history(run.info.run_id, "train/server-aggregation-seconds")
    assert [(point.step, point.value > 0) for point in history] == [(1, True)]
    assert run.data.metrics["train/server-aggregation-cuda"] == 0
    assert not any(key.startswith("station_") or "refinement-objective" in key for key in run.data.metrics)
    assert "refinement-objective-final" in pd.read_csv(output / "round_metrics.csv")
    assert "run.log" in {artifact.path for artifact in client.list_artifacts(run.info.run_id)}


def test_actual_flower_fedavg_weights_by_valid_window_count(tmp_path, flower_identity):
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


@pytest.mark.parametrize("strategy_name", ["fedavg", "fedprox", "lora"])
@pytest.mark.parametrize("pad_last_batch", [0, 1])
def test_federation_uses_fractional_sampled_target_support_before_replacement(
    tmp_path, monkeypatch, strategy_name, pad_last_batch, flower_identity,
):
    settings = (RunSettings(model="flowstate-r1", adaptation="lora") if strategy_name == "lora"
                else RunSettings(strategy=strategy_name))
    settings = replace(settings, local_epochs=2, batch_size=3, pad_last_batch=pad_last_batch)
    manifest = {
        "schema_version": 6,
        "geometry": {"context": 6, "horizon": 2, "max_train_windows": 2},
        "clients": [{"station": station, "num_train_windows": 12, "num_train_targets": 20}
                    for station in ("a", "b")],
    }
    strategy = AuditedStrategy(settings, manifest, tmp_path)
    width = (settings.lora_rank, settings.lora_rank)
    keys = (["adapter.proj.lora_A", "adapter.proj.lora_B"] if strategy_name == "lora" else ["weight"])

    def arrays(value):
        return ArrayRecord({key: torch.full(width, value) for key in keys})

    strategy.current_arrays = arrays(0.)
    # Aggregation only; no model-specific checkpoint metadata required here.
    monkeypatch.setattr(strategy, "_save", lambda *_: None)
    request = Message(content=RecordDict(), dst_node_id=10, message_type="train")
    replies = []
    for partition, selected, repeated, value in ((0, 5, 3, 2.), (1, 7, 4, 10.)):
        replies.append(Message(content=RecordDict({
            "arrays": arrays(value),
            "metrics": MetricRecord({
                "partition-id": partition, "num-examples": 2, "num-targets": selected / 2,
                "admissible-windows": 12, "unique-training-windows": 3, "epochs-completed": 2,
                "selected-target-occurrences": selected, "sampled-window-occurrences": 4 + 2 * pad_last_batch,
                "replacement-window-occurrences": 2 * pad_last_batch,
                "sampled-target-occurrences": selected + repeated * pad_last_batch,
                "train-mse-normalized": 1.,
            }),
        }), reply_to=request))
    result, metrics = strategy.aggregate_train(1, replies)
    assert metrics["num-targets"] == 6.
    for tensor in result.to_numpy_ndarrays():
        np.testing.assert_allclose(tensor, (2.5 * 2 + 3.5 * 10) / 6)
    station = replies[0].content["metrics"]
    station["replacement-window-occurrences"] += 1
    with pytest.raises(ValueError, match="Soporte de entrenamiento"):
        strategy.aggregate_train(2, replies)
    station["replacement-window-occurrences"] -= 1
    station["num-targets"] = 3.0
    with pytest.raises(ValueError, match="Soporte de entrenamiento"):
        strategy.aggregate_train(2, replies)


def test_pooled_rmsse_weights_observed_targets_and_takes_root_after_pooling():
    contents = []
    for count, score in ((1, 1.0), (3, 3.0)):
        contents.append(RecordDict({"metrics": MetricRecord({
            "num-examples": count, "num-targets": count,
            "mase-sum": 0.0, "mase-count": 0, "rmsse-sum": score * count, "rmsse-count": count,
            "rmsse-scaled-error-sum": count * score**2, "rmsse-target-count": count,
        })}))
    metrics = aggregate_evaluation(contents, "num-examples")
    assert metrics["mean-rmsse"] == 2.5
    assert metrics["pooled-rmsse"] == pytest.approx(np.sqrt(7))
    assert metrics["macro-rmsse"] == 2.0
    assert "mean-mase" not in metrics
    assert metrics["mase-count"] == 0


def test_macro_scaled_errors_weight_stations_not_origins():
    contents = []
    for origins, scaled in ((1, 1.), (9, 3.)):
        contents.append(RecordDict({"metrics": MetricRecord({
            "num-examples": origins, "num-targets": origins,
            "mase-sum": origins * scaled, "mase-count": origins, "mase": scaled,
            "rmsse-sum": origins * scaled, "rmsse-count": origins, "rmsse": scaled,
        })}))
    metrics = aggregate_evaluation(contents, "num-examples")
    assert metrics["mean-mase"] == 2.8
    assert metrics["macro-mase"] == metrics["macro-rmsse"] == 2.
    for flags, expected in (((False, False), {"robust-num-clients": 0}),
                            ((True, True), {"robust-num-clients": 2})):
        assert coverage_metrics(contents, {"clients": [{"robust_evaluation": flag} for flag in flags]}) == expected
    subset = coverage_metrics(contents, {"clients": [{"robust_evaluation": True}, {}]})
    assert subset["robust-num-clients"] == subset["robust-macro-mase"] == 1.


def test_scaled_lead_scores_do_not_include_stations_without_a_defined_scale():
    shared = {"mase-sum": 0., "mase-count": 0, "rmsse-sum": 0., "rmsse-count": 0}
    contents = [RecordDict({"metrics": MetricRecord(shared | record)}) for record in (
        {"num-examples": 1, "num-targets": 1, "lead-01-count": 1,
         "lead-01-mase-scaled-error-sum": 2., "lead-01-mase-target-count": 1,
         "lead-01-rmsse-scaled-error-sum": 9., "lead-01-rmsse-target-count": 1},
        {"num-examples": 9, "num-targets": 9, "lead-01-count": 9},
    )]
    metrics = aggregate_evaluation(contents, "num-examples")
    assert metrics["lead-01-count"] == 10
    assert metrics["lead-01-mase"] == 2.
    assert metrics["lead-01-rmsse"] == 3.


def test_local_training_is_reproducible_and_proximal_penalty_changes_updates(prepared):
    path, manifest = prepared
    _, artifact = load_client_data(str(path), 0)
    settings = RunSettings(hidden_size=8, local_epochs=2, batch_size=16,
                           early_stopping=0, lr_scheduler="none")
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


def test_client_sends_no_series_or_normalization_statistics(prepared, flower_grid):
    path, manifest = prepared
    settings = RunSettings(manifest=str(path), hidden_size=8)
    grid = flower_grid(settings, len(manifest["clients"]))
    model = make_model(manifest["geometry"], settings)
    reply, = grid.send_and_receive([Message(content=RecordDict({
        "arrays": ArrayRecord(model.state_dict()), "config": ConfigRecord({"server-round": 1}),
    }), dst_node_id=10, message_type="train")])
    assert set(reply.content) == {"arrays", "metrics"}
    assert {"partition-id", "num-examples", "train-mse-normalized", "train-seconds", "trainable-parameters",
            "uplink-tensor-bytes", "downlink-tensor-bytes"} <= set(reply.content["metrics"])
    assert not {"mean", "scale", "values", "timestamps"} & set(reply.content["metrics"])


def test_changed_preparation_code_requires_preparation(prepared, monkeypatch, flower_grid):
    path, manifest = prepared
    settings = RunSettings(manifest=str(path), hidden_size=8)
    grid = flower_grid(settings, len(manifest["clients"]))
    monkeypatch.setattr("fd_airquality.client_app.source_fingerprints", lambda: {"changed": "hash"})
    with pytest.raises(ValueError, match="Cambió el código local"):
        grid.send_and_receive([Message(content=RecordDict({
            "arrays": ArrayRecord(make_model(manifest["geometry"], settings).state_dict()),
            "config": ConfigRecord(),
        }), dst_node_id=10, message_type="evaluate")])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(("training", "scenario"), [
    (True, "active"), (False, "evaluation"), (True, "idle"), (True, "clock"),
    (True, "construction-error"), (False, "state-error"), (True, "task-error"),
])
def test_cuda_client_cleanup_preserves_arrival_and_validation_paths(
    monkeypatch, caplog, device, training, scenario, flower_identity,
):
    from fd_airquality import client_app

    manifest = {"preparation_sources": {}, "geometry": {"context": 2, "horizon": 1, "seasonality": 1,
                                                         "max_train_windows": 3}}
    artifact = {"mean": np.array(0.)}
    settings = RunSettings(device=device, local_epochs=2)
    trace = []
    validation = object()
    training_data = [] if scenario == "idle" else [0, 1]

    class Model:
        def to(self, destination):
            trace.append(f"model:{destination}")

    def make(*_):
        if scenario == "construction-error":
            raise RuntimeError("client failure")
        return Model()

    def load(*_):
        if scenario == "state-error":
            raise RuntimeError("client failure")

    def run(model, dataset, _, resolved, *args, **kwargs):
        model.to(resolved)
        if training:
            assert kwargs == {"proximal_mu": .25, "max_train_windows": 3, "validation_data": validation}
            assert args == (settings.seed + 100_003,)
            assert dataset is training_data and dataset
        if scenario == "task-error":
            raise RuntimeError("client failure")
        return {"num-examples": len(dataset)}

    monkeypatch.setattr(client_app, "load_client_data", lambda *_: (manifest, artifact))
    monkeypatch.setattr(client_app, "source_fingerprints", dict)
    monkeypatch.setattr(client_app, "configure_torch", lambda *_: torch.device(device))
    monkeypatch.setattr(client_app, "make_model", make)
    monkeypatch.setattr(client_app, "load_shared_state", load)
    monkeypatch.setattr(client_app, "training_view", lambda *_: training_data)
    monkeypatch.setattr(client_app, "arrival_metrics", lambda _: {"available-through-ns": 20, "available-from-ns": 10})
    monkeypatch.setattr(client_app, "round_validation_view", lambda *_: validation)
    monkeypatch.setattr(client_app, "WindowDataset", lambda *_: [0])
    monkeypatch.setattr(client_app, "validation_view", lambda *_: [0])
    monkeypatch.setattr(client_app, "train_local", run)
    monkeypatch.setattr(client_app, "evaluate_local", run)
    monkeypatch.setattr(client_app, "shared_state", lambda _: {"weight": torch.tensor(1.)})
    monkeypatch.setattr(gc, "collect", lambda: trace.append("collect"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: trace.append("empty_cache"))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _: 0)

    message = Message(content=RecordDict({
        "arrays": ArrayRecord({"weight": torch.tensor(1.)}),
        "config": ConfigRecord({"server-round": 1, "proximal-mu": .25, "available-through-ns": 20,
                                "available-from-ns": 0 if scenario == "clock" else 10}),
    }), dst_node_id=10, message_type="train" if training else "evaluate")
    with caplog.at_level("INFO", logger="fd_airquality"):
        if scenario.endswith("-error"):
            with pytest.raises(RuntimeError, match="client failure"):
                client_app._execute_task(message, settings, 0, training)
        elif scenario == "clock":
            with pytest.raises(ValueError, match="bloque de llegada"):
                client_app._execute_task(message, settings, 0, training)
        else:
            reply = client_app._execute_task(message, settings, 0, training)
            assert reply.content["metrics"]["partition-id"] == 0
            assert ("arrays" in reply.content) == training
            if training:
                assert reply.content["metrics"]["idle-client"] == int(scenario == "idle")
                assert reply.content["metrics"]["available-from-ns"] == 10
                assert reply.content["arrays"].to_torch_state_dict()["weight"].device.type == "cpu"

    expected = [f"model:{device}"] if scenario in {"active", "evaluation", "clock", "task-error"} else []
    if device == "cuda":
        if scenario != "construction-error":
            expected.append("model:cpu")
        expected += ["collect", "empty_cache"]
        assert "stage=cuda-cleanup" in caplog.text
        assert "allocated_mib=0.0 reserved_mib=0.0" in caplog.text
    else:
        assert "stage=cuda-cleanup" not in caplog.text
    assert trace == expected


def test_cuda_client_collects_cyclic_models_before_clearing_cache(monkeypatch, flower_identity):
    from fd_airquality import client_app

    references, transfers = [], []

    class CyclicModel(torch.nn.Linear):
        def __init__(self):
            super().__init__(2, 1)
            self.decode_func = self.forward  # PatchTST's bound-method cycle.

        def to(self, device):
            transfers.append(str(device))
            return super().to("cpu")

    def make(*_):
        model = CyclicModel()
        references.append(weakref.ref(model))
        return model

    def train(model, dataset, _, device, *args, **kwargs):
        model.to(device)(torch.ones(1, 2)).sum().backward()
        return {"num-examples": len(dataset)}

    def empty_cache():
        assert all(reference() is None for reference in references)

    manifest = {"preparation_sources": {}, "geometry": {"context": 2, "horizon": 1}}
    monkeypatch.setattr(client_app, "load_client_data", lambda *_: (manifest, {"mean": np.array(0.)}))
    monkeypatch.setattr(client_app, "source_fingerprints", dict)
    monkeypatch.setattr(client_app, "configure_torch", lambda *_: torch.device("cuda"))
    monkeypatch.setattr(client_app, "make_model", make)
    monkeypatch.setattr(client_app, "training_view", lambda *_: [0, 1])
    monkeypatch.setattr(client_app, "arrival_metrics", lambda _: {})
    monkeypatch.setattr(client_app, "train_local", train)
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _: 0)
    state = {"weight": torch.ones(1, 2), "bias": torch.zeros(1)}
    settings = RunSettings(device="cuda", local_epochs=1)
    automatic_collection = gc.isenabled()
    gc.disable()
    try:
        for stage in range(1, 4):
            message = Message(content=RecordDict({"arrays": ArrayRecord(state),
                                                  "config": ConfigRecord({"server-round": stage})}),
                              dst_node_id=10, message_type="train")
            reply = client_app._execute_task(message, settings, 0, True)
            assert references[-1]() is None
            state = reply.content["arrays"].to_torch_state_dict()
            assert all(value.device.type == "cpu" for value in state.values())
    finally:
        if automatic_collection:
            gc.enable()
        gc.collect()
    assert transfers == ["cuda", "cpu"] * 3


@pytest.mark.parametrize(("model", "algorithm"), [
    ("ttm-r3", "lora"), ("patchtst-fm-r2", "ffa-lora"), ("flowstate-r1", "flora"), ("ttm-r3", "lora-fair"),
])
def test_foundation_algorithms_complete_flower_rounds_and_checkpoint_history(
    global_prepared, tmp_path, flower_grid, mock_granite, model, algorithm,
):
    path, manifest = global_prepared
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"), model=model,
                           adaptation=algorithm, lora_targets="proj", batch_size=32,
                           num_server_rounds=2, fair_iterations=5, mlflow_tracking_uri="disabled")
    output = run_server(flower_grid(settings, len(manifest["clients"])), settings)
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
    if algorithm == "flora":
        restore_checkpoint(output / "last_model.pt")
    train = pd.read_csv(output / "round_metrics.csv").query("split == 'train'")
    assert train["round"].tolist() == [1, 2]
    assert (train[["uplink-tensor-bytes", "server-aggregation-seconds"]] > 0).all().all()
    if algorithm == "lora-fair":
        assert train["server-aggregation-cuda"].tolist() == [0, 0]


def test_server_discards_unusable_stations_before_any_training(selection_prepared, tmp_path, caplog, flower_grid):
    path, source = selection_prepared
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"), num_server_rounds=1,
                           mlflow_tracking_uri="disabled")
    grid = flower_grid(settings, len(source["clients"]))
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
def test_effective_flower_partitions_are_stable_and_do_not_share_message_configuration(
    selection_prepared, tmp_path, flower_grid,
):
    path, source = selection_prepared
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"), num_server_rounds=2,
                           mlflow_tracking_uri="disabled")
    grid = flower_grid(settings, len(source["clients"]))
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
