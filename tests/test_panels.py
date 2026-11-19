import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from fd_airquality.artifacts import file_digest
from fd_airquality.baselines import run_baseline
from fd_airquality.cli import launch_simulation, run_benchmark
from fd_airquality.comparison import compare_runs
from fd_airquality.data import load_client_data, read_manifest
from fd_airquality.panels import prepare_runtime_panel
from fd_airquality.settings import RunSettings


@pytest.mark.parametrize("metric", ["macro-mase", "macro-rmsse"])
def test_runtime_excludes_missing_validation_reference_and_keeps_artifacts_immutable(selection_prepared, tmp_path, caplog, metric):
    path, source = selection_prepared
    original = path.read_bytes()
    settings = RunSettings(manifest=str(path), selection_metric=metric)
    effective, manifest = prepare_runtime_panel(settings, tmp_path / "panel")
    assert path.read_bytes() == original
    assert [client["station"] for client in manifest["clients"]] == ["station-b"]
    assert manifest["clients"][0]["partition_id"] == 0
    assert manifest["clients"][0]["source_partition_id"] == 1
    assert manifest["clients"][0]["sha256"] == source["clients"][1]["sha256"]
    assert load_client_data(effective.manifest, 0)[0] == manifest
    report = pd.read_csv(tmp_path / "panel" / "excluded_stations.csv")
    assert report.station.to_list() == ["station-a"]
    assert report.reason.to_list() == ["validation_reference_no_pairs"]
    assert report.reference_pairs.to_list() == [0]
    assert "station-a" in caplog.text and "Sin pares" in caplog.text


def test_runtime_distinguishes_zero_scale_from_missing_pairs(global_prepared, tmp_path):
    path, manifest = global_prepared
    manifest["clients"][0]["validation_mase_defined"] = False
    path.write_text(json.dumps(manifest))
    prepare_runtime_panel(RunSettings(manifest=str(path)), tmp_path / "panel")
    rows = pd.read_csv(tmp_path / "panel" / "excluded_stations.csv")
    assert rows.reason.to_list() == ["validation_mase_scale_zero"]
    assert rows.reference_pairs.iloc[0] > 0


@pytest.mark.parametrize("metric", ["mae", "macro-mae", "rmse", "macro-rmse"])
def test_selection_rejects_unscaled_errors(metric):
    with pytest.raises(ValueError, match="selection_metric debe ser macro-mase o macro-rmsse"):
        RunSettings(selection_metric=metric)


def test_no_surviving_station_still_writes_report_before_aborting(selection_prepared, tmp_path, caplog):
    path, manifest = selection_prepared
    manifest["clients"][1]["validation_mase_defined"] = False
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Ninguna estación utilizable"):
        prepare_runtime_panel(RunSettings(manifest=str(path)), tmp_path / "panel")
    report = pd.read_csv(tmp_path / "panel" / "excluded_stations.csv")
    assert set(report.station) == {"station-a", "station-b"}
    assert "0 estaciones" in caplog.text
    assert not (tmp_path / "panel" / "runtime_manifest.json").exists()


def test_effective_panel_is_deterministic_and_preserves_report_when_reused(selection_prepared, tmp_path):
    path, _ = selection_prepared
    settings = RunSettings(manifest=str(path))
    a, _ = prepare_runtime_panel(settings, tmp_path / "a")
    b, _ = prepare_runtime_panel(settings, tmp_path / "b")
    assert file_digest(Path(a.manifest)) == file_digest(Path(b.manifest))
    reused, _ = prepare_runtime_panel(a, tmp_path / "reused")
    assert reused == a
    assert (tmp_path / "a" / "excluded_stations.csv").read_bytes() == (tmp_path / "reused" / "excluded_stations.csv").read_bytes()
    with pytest.raises(ValueError, match="manifest original"):
        prepare_runtime_panel(RunSettings(manifest=a.manifest, selection_metric="macro-rmsse"), tmp_path / "other")


def test_filtered_baselines_share_a_panel_and_cannot_be_compared_with_another_panel(
    selection_prepared, prepared, tmp_path, mock_granite,
):
    path, _ = selection_prepared
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"),
                           num_server_rounds=1, mlflow_tracking_uri="disabled")
    foundation = replace(settings, model="flowstate-r1", lora_targets="proj")
    original_hash = file_digest(path)
    local = run_baseline(settings, "local")
    central = run_baseline(foundation, "centralized")
    zero = run_baseline(foundation, "zero-shot")
    for run in (local, central, zero):
        test = json.loads((run / "test_metrics.json").read_text())["test"]
        assert test["num-clients"] == test["macro-mase-stations"] == 1
        metadata = json.loads((run / "run.json").read_text())
        assert metadata["manifest_sha256"] == file_digest(run / "runtime_manifest.json")
        assert metadata["manifest_sha256"] != original_hash
        assert pd.read_csv(run / "excluded_stations.csv").station.tolist() == ["station-a"]

    output = compare_runs([local, central, zero], tmp_path / "comparison")
    comparison = pd.read_csv(output / "comparison.csv")
    assert set(comparison["mode"]) == {"centralized", "local", "zero-shot"}
    assert comparison["num-clients"].eq(1).all()
    assert comparison.loc[comparison["mode"] == "centralized",
                          "macro_mase_improvement_vs_zero_shot_pct"].notna().item()
    other = run_baseline(replace(foundation, manifest=str(prepared[0])), "zero-shot")
    with pytest.raises(ValueError, match="manifests/paneles"):
        compare_runs([central, other], tmp_path / "invalid")


def test_flower_launcher_uses_retained_station_count_and_saves_audit(selection_prepared, tmp_path, monkeypatch, capsys):
    path, _ = selection_prepared
    project = Path("pyproject.toml").read_bytes()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_bytes(project)
    launch_simulation(RunSettings(manifest=str(path)), dry_run=True)
    assert "num-supernodes=1" in capsys.readouterr().out
    report, = (tmp_path / ".flwr" / "launches").glob("*/excluded_stations.csv")
    manifest = read_manifest(report.parent / "runtime_manifest.json")
    assert len(manifest["clients"]) == 1
    assert pd.read_csv(report).station.to_list() == ["station-a"]


def test_benchmark_filters_once_for_all_candidates(selection_prepared, tmp_path, monkeypatch):
    path, _ = selection_prepared
    calls = []

    def run(settings, *args, **kwargs):
        calls.append(settings)
        assert [row["station"] for row in read_manifest(settings.manifest)["clients"]] == ["station-b"]
        return tmp_path / f"run{len(calls)}"

    monkeypatch.setattr("fd_airquality.cli.launch_simulation", run)
    monkeypatch.setattr("fd_airquality.baselines.run_baseline", run)
    monkeypatch.setattr("fd_airquality.comparison.compare_runs", lambda paths, output: output)
    output = run_benchmark(RunSettings(manifest=str(path), output_dir=str(tmp_path / "benchmarks"),
                                      mlflow_tracking_uri="disabled"),
                            ["mlp"], ["federated", "local", "centralized"])
    assert len(calls) == 3
    assert len({call.manifest for call in calls}) == 1
    assert pd.read_csv(output.parent / "excluded_stations.csv").station.to_list() == ["station-a"]
