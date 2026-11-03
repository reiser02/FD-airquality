import json
from dataclasses import replace

import pandas as pd
import pytest
import torch

from fd_airquality.baselines import run_baseline
from fd_airquality.comparison import compare_runs
from fd_airquality.settings import RunSettings


@pytest.mark.parametrize("mode", ["local", "centralized"])
@pytest.mark.parametrize("model", ["mlp", "autoencoder", "darts-nlinear"])
def test_non_federated_controls_select_validation_and_test_once(prepared, tmp_path, monkeypatch, mode, model):
    from fd_airquality.baselines import evaluate_local as original

    path, manifest = prepared
    calls = []

    def evaluate(network, dataset, *args):
        calls.append(dataset)
        return original(network, dataset, *args)

    monkeypatch.setattr("fd_airquality.baselines.evaluate_local", evaluate)
    settings = RunSettings(
        manifest=str(path), output_dir=str(tmp_path / "runs"), model=model,
        num_server_rounds=2, hidden_size=8, batch_size=32, mlflow_tracking_uri="disabled",
    )
    output = run_baseline(settings, mode)
    results = json.loads((output / "test_metrics.json").read_text())
    assert results["test"]["num-clients"] == 2
    assert len(calls) == 2 * (3 + 1)
    assert all(len(dataset) == manifest["clients"][i]["num_test_windows"] for i, dataset in enumerate(calls[-2:]))
    assert len(pd.read_csv(output / "station_metrics.csv").query("split == 'test'")) == 2
    if mode == "local":
        assert results["selected_round"] is None
        assert len(results["selected_rounds"]) == 2
        assert len(list(output.glob("client_*_best_model.pt"))) == 2
    else:
        assert 0 <= results["selected_round"] <= 2
        checkpoint = torch.load(output / "best_model.pt", weights_only=True)
        assert checkpoint["round"] == results["selected_round"]


def test_centralized_epoch_contains_only_original_training_windows(prepared, tmp_path, monkeypatch):
    from fd_airquality.baselines import train_local as original

    path, manifest = prepared
    counts = []

    def train(model, dataset, *args, **kwargs):
        counts.append([len(station) for station in dataset.datasets])
        return original(model, dataset, *args, **kwargs)

    monkeypatch.setattr("fd_airquality.baselines.train_local", train)
    run_baseline(RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"), num_server_rounds=2,
                             mlflow_tracking_uri="disabled"), "centralized")
    assert counts == [[client["num_train_windows"] for client in manifest["clients"]]] * 2


def test_zero_shot_never_trains(prepared, tmp_path, monkeypatch):
    from test_models import DummyBackbone

    path, _ = prepared
    monkeypatch.setattr("fd_airquality.models._load_granite", lambda *_: DummyBackbone())
    monkeypatch.setattr("fd_airquality.baselines.train_local", lambda *_: pytest.fail("zero-shot entrenó"))
    output = run_baseline(RunSettings(
        manifest=str(path), output_dir=str(tmp_path / "runs"), model="flowstate-r1",
        mlflow_tracking_uri="disabled",
    ), "zero-shot")
    history = pd.read_csv(output / "round_metrics.csv")
    assert list(history["split"]) == ["validation", "test"]
    assert json.loads((output / "run.json").read_text())["training_budget"]["stages"] == 0


def test_comparison_requires_same_manifest_and_complete_station_panel(prepared, tmp_path):
    path, _ = prepared
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"),
                           num_server_rounds=1, mlflow_tracking_uri="disabled")
    local = run_baseline(settings, "local")
    central = run_baseline(settings, "centralized")
    output = compare_runs([local, central], tmp_path / "comparison")
    comparison = pd.read_csv(output / "comparison.csv")
    assert set(comparison["mode"]) == {"local", "centralized"}
    assert len(pd.read_csv(output / "station_comparison.csv")) == 4
    metadata = json.loads((central / "run.json").read_text())
    metadata["manifest_sha256"] = "changed"
    (central / "run.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="manifests"):
        compare_runs([local, central], tmp_path / "invalid")
    metadata["manifest_sha256"] = json.loads((local / "run.json").read_text())["manifest_sha256"]
    (central / "run.json").write_text(json.dumps(metadata))
    station_path = central / "station_metrics.csv"
    stations = pd.read_csv(station_path)
    stations = stations.loc[~((stations["split"] == "test") & (stations["partition-id"] == 0))]
    stations.to_csv(station_path, index=False)
    with pytest.raises(ValueError, match="Panel de test"):
        compare_runs([local, central], tmp_path / "invalid")


def test_comparison_rejects_wrong_per_station_counts_even_with_valid_pooled_counts(prepared, tmp_path):
    path, _ = prepared
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"),
                           num_server_rounds=1, mlflow_tracking_uri="disabled")
    local = run_baseline(settings, "local")
    central = run_baseline(settings, "centralized")
    station_path = central / "station_metrics.csv"
    stations = pd.read_csv(station_path)
    stations.loc[(stations["split"] == "test") & (stations["partition-id"] == 0), "num-targets"] += 1
    stations.to_csv(station_path, index=False)
    with pytest.raises(ValueError, match="recuentos de test"):
        compare_runs([local, central], tmp_path / "invalid")


def test_baseline_is_reproducible(prepared, tmp_path):
    path, _ = prepared
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"),
                           num_server_rounds=1, mlflow_tracking_uri="disabled")
    a = run_baseline(settings, "centralized")
    b = run_baseline(replace(settings), "centralized")
    assert (a / "test_metrics.json").read_bytes() == (b / "test_metrics.json").read_bytes()


@pytest.mark.parametrize("model", ["ttm-r3", "patchtst-fm-r2", "flowstate-r1"])
@pytest.mark.parametrize("algorithm", ["lora", "ffa-lora"])
@pytest.mark.parametrize("mode", ["local", "centralized"])
def test_foundation_controls_use_internal_adapters_on_identical_global_test(global_prepared, tmp_path, monkeypatch, model, algorithm, mode):
    from test_models import DummyBackbone

    monkeypatch.setattr("fd_airquality.models._load_granite", lambda *_: DummyBackbone())
    path, manifest = global_prepared
    output = run_baseline(RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"), model=model,
        adaptation=algorithm, lora_targets="proj", num_server_rounds=1, batch_size=32,
        mlflow_tracking_uri="disabled"), mode)
    result = json.loads((output / "test_metrics.json").read_text())
    assert "macro-mase" in result["selection_metric"]
    assert result["test"]["num-examples"] == sum(c["num_test_windows"] for c in manifest["clients"])
    for checkpoint in output.glob("*best_model.pt"):
        payload = torch.load(checkpoint, weights_only=True)
        assert payload["model"]["adaptation"] == algorithm
        assert all(key.startswith("adapter.") for key in payload["model_state_dict"])


def test_comparison_labels_adapter_and_relative_zero_shot_improvements(global_prepared, tmp_path, monkeypatch):
    from test_models import DummyBackbone

    monkeypatch.setattr("fd_airquality.models._load_granite", lambda *_: DummyBackbone())
    path, _ = global_prepared
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"), model="flowstate-r1",
        lora_targets="proj", num_server_rounds=1, batch_size=32, mlflow_tracking_uri="disabled")
    zero = run_baseline(settings, "zero-shot")
    adapted = run_baseline(settings, "centralized")
    output = compare_runs([zero, adapted], tmp_path / "comparison")
    rows = pd.read_csv(output / "comparison.csv")
    assert set(rows["adaptation"]) == {"none", "lora"}
    assert rows.loc[rows["adaptation"] == "lora", "macro_mase_improvement_vs_zero_shot_pct"].notna().all()
    stations = pd.read_csv(output / "station_comparison.csv")
    assert "mae_improvement_vs_zero_shot_pct" in stations


def test_local_cuda_keeps_only_active_station_in_gpu_and_independent_checkpoints(
    prepared, tmp_path, monkeypatch,
):
    path, _ = prepared
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

    def train(model, dataset, *args):
        assert residents == {model.index}
        calls.append((model.index, "train"))
        with torch.no_grad():
            model.weight.add_(1)
        return {"num-examples": len(dataset), "train-mse-normalized": 1.0}

    def evaluate(model, dataset, *args):
        assert residents == {model.index}
        calls.append((model.index, dataset.split))
        count = len(dataset)
        targets = count * dataset.horizon
        error = 1.0 / model.weight.item()
        return {
            "num-examples": count, "num-targets": targets,
            "absolute-error-sum": targets * error, "squared-error-sum": targets * error**2,
            "baseline-absolute-error-sum": targets, "baseline-squared-error-sum": targets,
            "mae": error, "mse": error**2, "baseline-mae": 1.0,
            "mase-sum": count * error, "mase-count": count,
            "rmsse-sum": count * error, "rmsse-count": count,
            "mase": error, "rmsse": error,
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
    for index in range(2):
        assert calls.count((index, "train")) == 2
        assert calls.count((index, "validation")) == 3
        assert calls.count((index, "test")) == 1
        saved = torch.load(output / f"client_{index:03d}_best_model.pt", weights_only=True)
        assert saved["model_state_dict"]["weight"].item() == index + 3
        assert saved["round"] == 2


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
