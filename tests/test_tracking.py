import json
from datetime import UTC, datetime

import pytest
from mlflow import MlflowClient

from fd_airquality.baselines import run_baseline
from fd_airquality.settings import RunSettings
from fd_airquality.tracking import RunTracker, benchmark_tags, run_labels, tracking_uri, write_run


@pytest.mark.parametrize("failure", [False, True])
def test_mlflow_tracks_metrics_artifacts_and_status_without_measurements(prepared, tmp_path, monkeypatch, failure, mlflow_uri):
    path, _ = prepared
    output_dir = tmp_path / "benchmark_20261104_091836_270422" / "runs" / "case_0001"
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
    assert run.data.params["model"] == "mlp"
    assert run.data.params["data.context"] == "6"
    assert "manifest_sha256" in run.data.tags
    assert run.data.tags["benchmark"] == "PM25 | 2026-11-04 09:18:36 UTC"
    assert run.data.tags["candidate"] == "MLP | centralizado | completo"
    assert run.info.run_name == "MLP | centralizado | completo | PM25 | 2026-11-04 09:18:36 UTC"
    assert run.data.params["seed"] == "42" and run.data.params["lora_rank"] == "4"
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
    ("reports/benchmark_20261104_091836_270422/runs/case_0029", {"benchmark": "PM25 | 2026-11-04 09:18:36 UTC"}),
    ("/missing/path/fold_003/benchmark_20261104_091836_270422/runs/case_10000/",
     {"benchmark": "PM25 | 2026-11-04 09:18:36 UTC"}),
    ("reports/benchmark_20261104_091836/runs/case_0029", {"benchmark": "PM25 | 2026-11-04 09:18:36 UTC"}),
    ("reports/benchmark_20261304_091836_270422/runs/case_0029", {}),
    ("reports/benchmark_20261104_091836_270422/runs/case_0029_copy", {}),
    ("reports/benchmark_20261104_091836_270422/not-runs/case_0029", {}),
    ("reports/non_federated", {}),
])
def test_benchmark_tags_derive_the_matrix_label_without_reading_or_renaming_paths(directory, expected):
    assert benchmark_tags(directory, "PM25") == expected


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


def test_duplicate_display_names_remain_separate_and_filterable_by_seed_and_rank(prepared, tmp_path, mlflow_uri):
    path, manifest = prepared
    client = MlflowClient(tracking_uri=mlflow_uri)
    ids = {}
    for seed, rank in ((42, 4), (42, 8), (43, 4)):
        output_dir = tmp_path / "benchmark_20261104_091836_270422" / "runs" / f"case_{len(ids) + 1:04d}"
        output = output_dir / "run"
        output.mkdir(parents=True)
        settings = RunSettings(manifest=str(path), output_dir=str(output_dir), seed=seed, lora_rank=rank,
                               mlflow_tracking_uri=mlflow_uri)
        with RunTracker(settings, output, "federated", manifest) as tracker:
            ids[seed, rank] = tracker.run_id

    experiment = client.get_experiment_by_name("fd-airquality")
    query = "tags.benchmark = 'PM25 | 2026-11-04 09:18:36 UTC' AND tags.candidate = 'MLP | federado | completo | FedAvg'"
    runs = client.search_runs([experiment.experiment_id], filter_string=query)
    assert {run.info.run_id for run in runs} == set(ids.values()) and len(runs) == 3
    assert len({run.info.run_name for run in runs}) == 1
    for (seed, rank), run_id in ids.items():
        selected, = client.search_runs([experiment.experiment_id], filter_string=(
            query + f" AND params.seed = '{seed}' AND params.lora_rank = '{rank}'"
        ))
        assert selected.info.run_id == run_id
