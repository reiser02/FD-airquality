import json
from datetime import UTC, datetime
from unittest.mock import Mock

import pytest
from mlflow import MlflowClient

from fd_airquality.baselines import run_baseline
from fd_airquality.settings import RunSettings
from fd_airquality.tracking import RunTracker, backtest_tags, run_labels, tracking_uri, write_run


@pytest.mark.parametrize("failure", [False, True])
def test_mlflow_tracks_metrics_artifacts_and_status_without_measurements(prepared, tmp_path, monkeypatch, failure, mlflow_uri):
    path, _ = prepared
    output_dir = tmp_path / "backtest_20261104_091836" / "fold_001" / "runs" / "case_0001"
    settings = RunSettings(manifest=str(path), output_dir=str(output_dir), num_server_rounds=1,
                           mlflow_tracking_uri=mlflow_uri, mlflow_experiment="test-airquality")
    if failure:
        def fail(*_, **__):
            raise RuntimeError("training failed")

        monkeypatch.setattr("fd_airquality.baselines.train_local", fail)
        with pytest.raises(RuntimeError, match="training failed"):
            run_baseline(settings, "centralized")
    else:
        run_baseline(settings, "centralized")
    metadata_path, = output_dir.glob("*/mlflow.json")
    metadata = json.loads(metadata_path.read_text())
    client = MlflowClient(tracking_uri=mlflow_uri)
    run = client.get_run(metadata["run_id"])
    assert run.info.status == ("FAILED" if failure else "FINISHED")
    assert run.data.tags["model"] == "mlp"
    assert run.data.tags["adaptation"] == "full"
    assert run.data.params["data.context"] == "6"
    assert "manifest_sha256" in run.data.tags
    assert run.data.tags["backtest"] == "PM25 | 2026-11-04 09:18:36 UTC"
    assert run.data.tags["fold"] == "1"
    assert run.data.tags["candidate"] == "MLP | centralizado | completo"
    assert run.info.run_name == "MLP | centralizado | completo | PM25 | 2026-11-04 09:18:36 UTC | fold 1"
    assert run.data.params["seed"] == "42"
    assert not {"model", "model.name", "model.context", "adaptation", "strategy", "proximal_mu",
                "mlflow_tracking_uri", "mlflow_experiment", "mlflow_log_models", "heartbeat_seconds", "log_level",
                "data.validation_hours", "data.test_hours"} & run.data.params.keys()
    assert not any(key.startswith(("lora_", "fair_")) for key in run.data.params)
    saved_settings = json.loads((metadata_path.parent / "run.json").read_text())["settings"]
    assert saved_settings["mlflow_tracking_uri"] == mlflow_uri
    artifacts = client.list_artifacts(run.info.run_id)
    names = {artifact.path for artifact in artifacts}
    assert "run.json" in names
    assert "station_metrics.csv" not in names
    assert not any(name.endswith((".npz", ".npy")) or "series" in name for name in names)
    if not failure:
        assert {"test/macro-mase", "test/macro-rmsse", "selection/round", "runtime/wall-seconds"} <= run.data.metrics.keys()
        assert not any(key.startswith("station_") or key.endswith(("mae", "rmse", "-sum")) for key in run.data.metrics)
        assert "train/train-mse-normalized" not in run.data.metrics
        assert (metadata_path.parent / "station_metrics.csv").is_file()
        assert "checkpoints" in names
        assert client.list_artifacts(run.info.run_id, "checkpoints")[0].path.endswith("best_model.pt")


@pytest.mark.parametrize("mode", ["local", "zero-shot"])
def test_tracking_does_not_publish_per_station_selection_or_unused_parameters(prepared, tmp_path, mlflow_uri, mock_granite, mode):
    path, _ = prepared
    output = run_baseline(RunSettings(
        manifest=str(path), output_dir=str(tmp_path / "runs"), num_server_rounds=1,
        model="flowstate-r1", lora_targets="proj", mlflow_tracking_uri=mlflow_uri,
    ), mode)
    run_id = json.loads((output / "mlflow.json").read_text())["run_id"]
    data = MlflowClient(tracking_uri=mlflow_uri).get_run(run_id).data
    assert data.tags["adaptation"] == ("none" if mode == "zero-shot" else "lora")
    assert "model.revision" in data.params
    assert ("lora_rank" in data.params) == (mode == "local")
    assert "fair_iterations" not in data.params
    assert not any(key.startswith("station_") for key in data.metrics)
    assert ("selection/round" in data.metrics) == (mode == "zero-shot")
    if mode == "zero-shot":
        assert not {"learning_rate", "local_epochs", "num_server_rounds"} & data.params.keys()
    else:
        test = json.loads((output / "test_metrics.json").read_text())
        assert test["selected_round"] is test["validation_score"] is None
        assert "selected_rounds" not in test


def test_relative_sqlite_uri_is_resolved_before_flower_changes_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert tracking_uri(RunSettings(mlflow_tracking_uri="sqlite:///mlflow.db")) == f"sqlite:///{tmp_path / 'mlflow.db'}"


def test_tracking_preserves_aggregation_device_when_reported(tmp_path):
    tracker = RunTracker(RunSettings(), tmp_path, "federated", {})
    tracker.client = Mock()
    tracker.log_metrics("train", {"server-aggregation-cuda": 1, "refinement-objective-final": .5}, 3)
    points = tracker.client.log_batch.call_args.kwargs["metrics"]
    assert [(point.key, point.value, point.step) for point in points] == [("train/server-aggregation-cuda", 1., 3)]


def test_cpu_run_metadata_never_initializes_cuda_or_cudnn(prepared, tmp_path, monkeypatch):
    path, manifest = prepared

    def unexpected_probe(*_, **__):
        pytest.fail("CPU metadata must not probe CUDA or initialize cuDNN")

    monkeypatch.setattr("torch.backends.cudnn.version", unexpected_probe)
    for name in ("is_available", "device_count", "get_device_properties", "init", "_lazy_init"):
        monkeypatch.setattr(f"torch.cuda.{name}", unexpected_probe)
    settings = RunSettings(manifest=str(path), device="cpu", mlflow_tracking_uri="disabled")

    write_run(tmp_path, settings, manifest, "centralized")

    run = json.loads((tmp_path / "run.json").read_text())
    hardware = run["hardware"]
    assert hardware["cudnn"] is None
    assert "gpus" not in hardware
    assert run["arrival_protocol"]["period_hours"] == 0
    assert run["arrival_protocol"]["training_targets"] == "full_fold_training"


@pytest.mark.parametrize("directory,expected", [
    ("reports/backtest_20261104_091836/fold_001/runs/case_0029", {"backtest": "PM25 | 2026-11-04 09:18:36 UTC", "fold": "1"}),
    ("/missing/path/backtest_20261104_091836/fold_003/runs/case_10000/",
     {"backtest": "PM25 | 2026-11-04 09:18:36 UTC", "fold": "3"}),
    ("reports/backtest_20261104_091836_1/fold_003/runs/case_0029",
     {"backtest": "PM25 | 2026-11-04 09:18:36 UTC", "fold": "3"}),
    ("reports/backtest_20261304_091836/fold_001/runs/case_0029", {}),
    ("reports/backtest_20261104_091836/fold_001/runs/case_0029_copy", {}),
    ("reports/backtest_20261104_091836/fold_001/not-runs/case_0029", {}),
    ("reports/backtest_20261104_091836/runs/case_0029", {}),
    ("reports/non_federated", {}),
])
def test_backtest_tags_derive_the_group_and_fold_without_reading_paths(directory, expected):
    assert backtest_tags(directory, "PM25") == expected


@pytest.mark.parametrize("model,mode,adaptation,strategy,candidate", [
    ("patchtst-fm-r2", "federated", "lora-fair", "fedavg", "PatchTST-FM-R2 | federado | LoRA-FAIR | FedAvg"),
    ("nlinear", "federated", "full", "fedprox", "NLinear | federado | completo | FedProx"),
    ("nhits", "centralized", "full", "fedavg", "NHITS | centralizado | completo"),
    ("ttm-r3", "local", "ffa-lora", "fedavg", "TTM-R3 | local | FFA-LoRA"),
    ("flowstate-r1", "zero-shot", "none", "fedavg", "FlowState-R1 | zero-shot | sin adaptación"),
])
def test_standalone_run_labels_use_readable_model_mode_and_adaptation(model, mode, adaptation, strategy, candidate):
    settings = RunSettings(model=model, strategy=strategy, output_dir="reports/non_federated")
    started_at = datetime(2026, 11, 5, 12, 30, 0, 123456, tzinfo=UTC)
    assert run_labels(settings, mode, "PM25", adaptation, started_at=started_at) == {
        "candidate": candidate, "mlflow.runName": f"{candidate} | PM25 | 2026-11-05 12:30:00 UTC",
    }


def test_duplicate_display_names_remain_separate_and_filterable_by_seed_and_rank(prepared, tmp_path, mlflow_uri, mock_granite):
    path, manifest = prepared
    client = MlflowClient(tracking_uri=mlflow_uri)
    ids = {}
    for seed, rank in ((42, 4), (42, 8), (43, 4)):
        output_dir = tmp_path / "backtest_20261104_091836" / "fold_001" / "runs" / f"case_{len(ids) + 1:04d}"
        output = output_dir / "run"
        output.mkdir(parents=True)
        settings = RunSettings(manifest=str(path), output_dir=str(output_dir), seed=seed, lora_rank=rank,
                               model="flowstate-r1", lora_targets="proj", mlflow_tracking_uri=mlflow_uri)
        with RunTracker(settings, output, "federated", manifest) as tracker:
            ids[seed, rank] = tracker.run_id

    experiment = client.get_experiment_by_name("fd-airquality")
    query = ("tags.backtest = 'PM25 | 2026-11-04 09:18:36 UTC' AND tags.fold = '1' "
             "AND tags.candidate = 'FlowState-R1 | federado | LoRA | FedAvg'")
    runs = client.search_runs([experiment.experiment_id], filter_string=query)
    assert {run.info.run_id for run in runs} == set(ids.values()) and len(runs) == 3
    assert len({run.info.run_name for run in runs}) == 1
    for (seed, rank), run_id in ids.items():
        selected, = client.search_runs([experiment.experiment_id], filter_string=(
            query + f" AND params.seed = '{seed}' AND params.lora_rank = '{rank}'"
        ))
        assert selected.info.run_id == run_id
