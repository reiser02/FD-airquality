import json

import pandas as pd
import pytest
import torch

from fd_airquality.baselines import run_baseline
from fd_airquality.models import AUTOENCODER_MODELS
from fd_airquality.probabilistic import MEDIAN_ANCHORED_REARRANGEMENT
from fd_airquality.settings import RunSettings


@pytest.mark.parametrize(("mode", "model"), [
    ("centralized", "mlp"), ("local", "autoencoder"), ("centralized", "nlinear"), ("local", "nhits"),
    ("local", "autoencoder-tiny"), ("centralized", "autoencoder-tiny"),
])
def test_classical_baselines_write_test_metrics_and_checkpoints(prepared, tmp_path, mode, model):
    path, manifest = prepared
    settings = RunSettings(
        manifest=str(path), output_dir=str(tmp_path / "runs"), model=model,
        num_server_rounds=1, hidden_size=8, batch_size=32, mlflow_tracking_uri="disabled",
        probabilistic=1,
    )
    output = run_baseline(settings, mode)
    results = json.loads((output / "test_metrics.json").read_text())
    clients = manifest["clients"]
    assert results["test"]["num-clients"] == len(clients)
    test = pd.read_csv(output / "station_metrics.csv").query("split == 'test'")
    assert test["num-examples"].tolist() == [client["num_test_windows"] for client in clients]
    assert not (output / "mlflow.json").exists()
    protocol = json.loads((output / "run.json").read_text())["probabilistic_protocol"]
    if model in AUTOENCODER_MODELS:
        assert protocol["quantile_postprocessing"] == "none"
        assert not {"quantile-raw-crossings", "crps"} & results["test"].keys()
    else:
        assert protocol["quantile_postprocessing"] == MEDIAN_ANCHORED_REARRANGEMENT
        assert results["test"]["quantile-crossings"] == 0
        assert results["test"]["quantile-raw-crossings"] == test["quantile-raw-crossings"].sum() > 0
        assert results["test"]["interval-90-mean-width"] >= 0
    if mode == "local":
        assert results["selected_round"] is None
        stages = test["round"].tolist()
        names = [f"client_{client['partition_id']:03d}_best_model.pt" for client in clients]
    else:
        stages = [results["selected_round"]]
        names = ["best_model.pt"]
    for name, stage in zip(names, stages, strict=True):
        checkpoint = torch.load(output / name, weights_only=True)
        assert checkpoint["round"] == stage and stage in {0, 1}
        assert checkpoint["model"]["name"] == model


@pytest.mark.parametrize("selected_step", [0, 1])
def test_centralized_fits_full_fold_once_and_restores_validation_winner(
    arrival_prepared, tmp_path, monkeypatch, selected_step,
):
    from fd_airquality.baselines import evaluate_local

    path, manifest = arrival_prepared
    geometry, clients = manifest["geometry"], manifest["clients"]
    model = torch.nn.Linear(geometry["context"], geometry["horizon"], bias=False)
    torch.nn.init.zeros_(model.weight)
    evaluations = []

    def train(model, dataset, settings, *args, **kwargs):
        assert [(station.split, len(station)) for station in dataset.datasets] == [
            ("train", client["num_train_windows"]) for client in clients
        ]
        assert all(station.available_through is None for station in dataset.datasets)
        assert settings.local_epochs == 4
        assert all(station.split == "validation" for station in kwargs["validation_data"].datasets)
        with torch.no_grad():
            model.weight.add_(1)
        evaluations.append(("train", model.weight[0, 0].item()))
        return {"num-examples": len(dataset), "train-mse-normalized": 1.}

    def evaluate(model, dataset, *args):
        stage = model.weight[0, 0].item()
        evaluations.append((dataset.split, stage))
        metrics = evaluate_local(model, dataset, *args)
        if dataset.split == "validation":
            mase, rmsse = (1., 2.) if stage == selected_step else (2., 1.)
            metrics.update({"mase": mase, "mase-sum": mase * metrics["mase-count"],
                            "rmsse": rmsse, "rmsse-sum": rmsse * metrics["rmsse-count"]})
        return metrics

    monkeypatch.setattr("fd_airquality.baselines.make_model", lambda *_: model)
    monkeypatch.setattr("fd_airquality.baselines.train_local", train)
    monkeypatch.setattr("fd_airquality.baselines.evaluate_local", evaluate)
    output = run_baseline(RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"),
                                      num_server_rounds=5, arrival_period_hours=24, local_epochs=4,
                                      missing_policy="ffill", mlflow_tracking_uri="disabled"),
                          "centralized")
    assert evaluations == ([("validation", 0)] * len(clients) + [("train", 1)]
                           + [("validation", 1)] * len(clients) + [("test", selected_step)] * len(clients))
    result = json.loads((output / "test_metrics.json").read_text())
    checkpoint = torch.load(output / "best_model.pt", weights_only=True)
    assert result["selection_metric"] == "validation_macro-mase"
    assert result["selected_round"] == checkpoint["round"] == selected_step
    torch.testing.assert_close(checkpoint["model_state_dict"]["weight"], torch.full_like(model.weight, selected_step))
    last = torch.load(output / "last_model.pt", weights_only=True)
    assert last["round"] == 1
    torch.testing.assert_close(last["model_state_dict"]["weight"], torch.ones_like(model.weight))


@pytest.mark.parametrize("mode", ["local", "centralized"])
def test_zero_shot_never_trains(prepared, tmp_path, monkeypatch, mock_granite, mode):
    path, _ = prepared
    monkeypatch.setattr("fd_airquality.baselines.train_local",
                        lambda *args, **kwargs: pytest.fail("zero-shot entrenó"))
    output = run_baseline(RunSettings(
        manifest=str(path), output_dir=str(tmp_path / "runs"), model="flowstate-r1",
        adaptation="zero-shot", mlflow_tracking_uri="disabled",
    ), mode)
    history = pd.read_csv(output / "round_metrics.csv")
    assert list(history["split"]) == ["validation", "test"]
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["training_budget"]["stages"] == 0
    assert metadata["mode"] == mode
    assert metadata["model"]["adaptation"] == "zero-shot"


def test_baseline_is_reproducible(prepared, tmp_path):
    path, _ = prepared
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"),
                           num_server_rounds=1, mlflow_tracking_uri="disabled")
    a = run_baseline(settings, "centralized")
    b = run_baseline(settings, "centralized")
    assert json.loads((a / "test_metrics.json").read_text()) == json.loads((b / "test_metrics.json").read_text())


@pytest.mark.parametrize(("model", "mode", "algorithm"), [
    ("ttm-r3", "local", "lora"), ("flowstate-r1", "centralized", "lora"),
    ("patchtst-fm-r2", "local", "ffa-lora"), ("ttm-r3", "centralized", "ffa-lora"),
])
def test_foundation_controls_use_internal_adapters_on_identical_global_test(
    global_prepared, tmp_path, mock_granite, model, mode, algorithm,
):
    path, manifest = global_prepared
    output = run_baseline(RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"), model=model,
        adaptation=algorithm, lora_targets="proj", num_server_rounds=1, batch_size=32,
        mlflow_tracking_uri="disabled"), mode)
    result = json.loads((output / "test_metrics.json").read_text())
    prefix = "station_" if mode == "local" else ""
    assert result["selection_metric"] == f"{prefix}validation_macro-mase"
    assert result["test"]["num-examples"] == sum(c["num_test_windows"] for c in manifest["clients"])
    names = ([f"client_{client['partition_id']:03d}_best_model.pt" for client in manifest["clients"]]
             if mode == "local" else ["best_model.pt"])
    expected = {"adapter.proj.lora_A", "adapter.proj.lora_B"} if algorithm == "lora" else {"adapter.proj.lora_B"}
    for name in names:
        payload = torch.load(output / name, weights_only=True)
        assert payload["model"]["adaptation"] == algorithm
        assert set(payload["model_state_dict"]) == expected


def test_local_cuda_keeps_only_active_station_in_gpu_and_independent_checkpoints(
    prepared, tmp_path, monkeypatch,
):
    path, manifest = prepared
    models, residents, calls = [], set(), []

    class ResidentModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.index = len(models)
            self.weight = torch.nn.Parameter(torch.tensor(float(self.index + 1)))
            models.append(self)

        def to(self, device):
            if torch.device(device).type == "cuda":
                residents.add(self.index)
                assert len(residents) == 1, "Only the active station may reside in GPU"
            else:
                residents.discard(self.index)
            return self

    def train(model, dataset, *args, **kwargs):
        assert residents == {model.index}
        assert dataset.split == "train"
        assert len(dataset) == manifest["clients"][model.index]["num_train_windows"]
        with torch.no_grad():
            model.weight.add_(1)
        calls.append((model.index, "train", model.weight.item()))
        return {"num-examples": len(dataset), "train-mse-normalized": 1.0}

    def evaluate(model, dataset, *args):
        assert residents == {model.index}
        weight = model.weight.item()
        calls.append((model.index, dataset.split, weight))
        count = len(dataset)
        targets = count * dataset.horizon
        # Each station independently chooses initial or fitted weights.
        scaled_error = float(1 + abs(weight - (1, 3)[model.index]))
        return {
            "num-examples": count, "num-targets": targets,
            "mase-sum": count * scaled_error, "mase-count": count,
            "rmsse-sum": count * scaled_error, "rmsse-count": count,
            "mase": scaled_error, "rmsse": scaled_error,
        }

    monkeypatch.setattr("fd_airquality.baselines.make_model", lambda *_: ResidentModel())
    monkeypatch.setattr("fd_airquality.baselines.configure_torch", lambda *_: torch.device("cuda"))
    monkeypatch.setattr("fd_airquality.baselines.train_local", train)
    monkeypatch.setattr("fd_airquality.baselines.evaluate_local", evaluate)
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"),
                           device="cuda", num_server_rounds=2, mlflow_tracking_uri="disabled")
    output = run_baseline(settings, "local")
    assert not residents
    assert len(models) == 2
    result = json.loads((output / "test_metrics.json").read_text())
    assert result["selection_metric"] == "station_validation_macro-mase"
    assert result["selected_round"] is None
    assert pd.read_csv(output / "station_metrics.csv").query("split == 'test'")["round"].tolist() == [0, 1]
    for index, (selected_step, best_weight) in enumerate(((0, 1.0), (1, 3.0))):
        assert [(split, weight) for i, split, weight in calls if i == index] == [
            ("validation", index + 1), ("train", index + 2), ("validation", index + 2),
            ("test", best_weight),
        ]
        saved = torch.load(output / f"client_{index:03d}_best_model.pt", weights_only=True)
        assert saved["model_state_dict"]["weight"].item() == best_weight
        assert saved["round"] == selected_step


@pytest.mark.parametrize("fails", [False, True])
def test_local_model_is_offloaded_even_after_failure(fails):
    from fd_airquality.baselines import _model_on_device

    class Model:
        def __init__(self):
            self.moves = []

        def to(self, device):
            self.moves.append(str(device))
            return self

    model = Model()

    def execute():
        with _model_on_device(model, torch.device("cuda:1"), offload=True) as active:
            assert active is model
            assert model.moves == ["cuda:1"]
            if fails:
                raise RuntimeError("failed station")

    if fails:
        with pytest.raises(RuntimeError, match="failed station"):
            execute()
    else:
        execute()
    assert model.moves == ["cuda:1", "cpu"]
