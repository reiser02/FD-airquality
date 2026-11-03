import json

import pytest
from mlflow import MlflowClient

from fd_airquality.baselines import run_baseline
from fd_airquality.settings import RunSettings
from fd_airquality.tracking import tracking_uri


@pytest.mark.parametrize("failure", [False, True])
def test_mlflow_tracks_metrics_artifacts_and_status_without_measurements(prepared, tmp_path, monkeypatch, failure, mlflow_uri):
    path, _ = prepared
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"), num_server_rounds=1,
                           mlflow_tracking_uri=mlflow_uri, mlflow_experiment="test-airquality")
    if failure:
        def fail(*_):
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
    assert run.data.params["model"] == "mlp"
    assert run.data.params["data.context"] == "6"
    assert "manifest_sha256" in run.data.tags
    artifacts = client.list_artifacts(run.info.run_id)
    names = {artifact.path for artifact in artifacts}
    assert "run.json" in names
    assert not any(name.endswith((".npz", ".npy")) or "series" in name for name in names)
    if not failure:
        assert "test/mae" in run.data.metrics
        assert "station_000/test/mae" in run.data.metrics
        assert "checkpoints" in names
        assert client.list_artifacts(run.info.run_id, "checkpoints")[0].path.endswith("best_model.pt")


def test_relative_sqlite_uri_is_resolved_before_flower_changes_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert tracking_uri(RunSettings(mlflow_tracking_uri="sqlite:///mlflow.db")) == f"sqlite:///{tmp_path / 'mlflow.db'}"
