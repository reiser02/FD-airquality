import json
from unittest.mock import Mock

import pytest
from mlflow import MlflowClient

from fd_airquality.baselines import run_baseline
from fd_airquality.settings import RunSettings
from fd_airquality.tracking import RunTracker, tracking_uri, write_run


@pytest.mark.parametrize("failure", [False, True])
def test_mlflow_tracks_metrics_artifacts_and_status_without_measurements(prepared, tmp_path, monkeypatch, failure, mlflow_uri):
    path, _ = prepared
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"), num_server_rounds=1,
                           mlflow_tracking_uri=mlflow_uri, mlflow_experiment="test-airquality")
    if failure:
        def fail(*_, **__):
            raise RuntimeError("training failed")

        monkeypatch.setattr("fd_airquality.baselines.train_local", fail)
        with pytest.raises(RuntimeError, match="training failed"):
            run_baseline(settings, "centralized")
    else:
        run_baseline(settings, "centralized")
    metadata_path, = list((tmp_path / "runs").glob("*/mlflow.json"))
    metadata = json.loads(metadata_path.read_text())
    client = MlflowClient(tracking_uri=mlflow_uri)
    run = client.get_run(metadata["run_id"])
    assert run.info.status == ("FAILED" if failure else "FINISHED")
    assert run.data.tags["model"] == "mlp"
    assert run.data.tags["adaptation"] == "full"
    assert run.data.params["data.context"] == "6"
    assert "manifest_sha256" in run.data.tags
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
