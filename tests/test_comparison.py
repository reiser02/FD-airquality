import json
from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from fd_airquality.comparison import compare_runs
from fd_airquality.probabilistic import MEDIAN_ANCHORED_REARRANGEMENT


@pytest.fixture
def completed_run(tmp_path):
    def make(name, model="mlp", mode="centralized", adaptation="full", scale=1., score=2.):
        path = tmp_path / name
        path.mkdir()
        clients = [{"partition_id": i, "station": f"station-{i}", "num_test_windows": 2} for i in range(2)]
        run = {
            "mode": mode, "manifest_sha256": "same-prepared-panel", "versions": {},
            "manifest": {"geometry": {"horizon": 12}, "clients": clients},
            "settings": {"model": model, "strategy": "fedavg", "seed": 42, "flowstate_scale_factor": scale},
            "model": {"adaptation": adaptation, "revision": f"revision-{model}",
                      "point_forecast": "median" if model == "flowstate-r1" else "upstream_point_prediction"},
        }
        metrics = {"macro-mase": score, "macro-rmsse": score, "num-clients": 2, "num-targets": 48,
                   "macro-mase-stations": 2, "macro-rmsse-stations": 2}
        (path / "run.json").write_text(json.dumps(run))
        (path / "test_metrics.json").write_text(json.dumps({"test": metrics, "selected_round": 0}))
        pd.DataFrame([{"split": "test", "partition-id": i, "station": f"station-{i}",
                       "num-examples": 2, "num-targets": 24,
                       "mase": score, "rmsse": score}
                      for i in (1, 0)]).to_csv(path / "station_metrics.csv", index=False)
        return path
    return make


def test_all_systems_and_hyperparameters_are_compared_in_both_directions(completed_run, tmp_path):
    paths = [
        completed_run("mlp", mode="federated", score=5.),
        completed_run("classic", model="autoencoder", score=4.),
        completed_run("ttm", model="ttm-r3", mode="zero-shot", adaptation="none", score=3.),
        completed_run("patch", model="patchtst-fm-r2", adaptation="lora", score=2.),
        completed_run("flow", model="flowstate-r1", adaptation="lora-fair", scale=2., score=1.),
    ]
    output = compare_runs(paths, tmp_path / "comparison")
    rows = pd.read_csv(output / "comparison.csv")
    pairs = pd.read_csv(output / "pairwise_comparison.csv")
    assert len(rows) == 5
    assert set(zip(pairs.run_a, pairs.run_b)) == set(permutations(rows.run, 2))
    assert len(pairs) == 5 * 4 * 2
    assert set(pairs.metric) == {"macro-mase", "macro-rmsse"}
    assert not pairs.delta_a_minus_b.isna().any()
    a = rows.loc[rows.model == "flowstate-r1", "run"].item()
    b = rows.loc[rows.model == "mlp", "run"].item()
    match = pairs.loc[(pairs.run_a == a) & (pairs.run_b == b) & (pairs.metric == "macro-mase")].iloc[0]
    assert match.delta_a_minus_b == -4.
    assert match.improvement_a_vs_b_pct == 80.
    assert not list(output.glob("station_*"))
    assert rows.loc[rows.model == "flowstate-r1", "flowstate_scale_factor"].item() == 2.


def test_zero_shot_reference_matches_inference_scale_without_filtering_other_pairs(completed_run, tmp_path):
    zero1 = completed_run("zero1", model="flowstate-r1", mode="zero-shot", adaptation="none", scale=1., score=4.)
    zero2 = completed_run("zero2", model="flowstate-r1", mode="zero-shot", adaptation="none", scale=2., score=2.)
    adapted = completed_run("adapted", model="flowstate-r1", adaptation="lora", scale=2., score=1.)
    output = compare_runs([zero1, zero2, adapted], tmp_path / "comparison")
    rows = pd.read_csv(output / "comparison.csv")
    assert set(rows["adaptation"]) == {"lora", "none"}
    assert rows.loc[rows.adaptation == "lora", "macro_mase_improvement_vs_zero_shot_pct"].item() == 50.
    assert len(pd.read_csv(output / "pairwise_comparison.csv")) == 3 * 2 * 2


def test_comparison_requires_a_complete_station_panel(completed_run, tmp_path):
    local = completed_run("local", mode="local")
    central = completed_run("central")
    path = central / "station_metrics.csv"
    stations = pd.read_csv(path)
    stations.loc[stations["partition-id"] != 0].to_csv(path, index=False)
    with pytest.raises(ValueError, match="Panel de test"):
        compare_runs([local, central], tmp_path / "invalid")


def test_comparison_rejects_wrong_per_station_counts_even_with_valid_pooled_counts(completed_run, tmp_path):
    local = completed_run("local", mode="local")
    central = completed_run("central")
    path = central / "station_metrics.csv"
    stations = pd.read_csv(path)
    # Move a target between stations without changing their total.
    stations.loc[stations["partition-id"] == 0, "num-targets"] += 1
    stations.loc[stations["partition-id"] == 1, "num-targets"] -= 1
    stations.to_csv(path, index=False)
    with pytest.raises(ValueError, match="recuentos de test"):
        compare_runs([local, central], tmp_path / "invalid")


def test_mismatched_zero_shot_scale_only_disables_attribution_not_comparison(completed_run, tmp_path):
    zero = completed_run("zero", model="flowstate-r1", mode="zero-shot", adaptation="none", scale=1.)
    adapted = completed_run("adapted", model="flowstate-r1", adaptation="lora", scale=2.)
    output = compare_runs([zero, adapted], tmp_path / "comparison")
    rows = pd.read_csv(output / "comparison.csv")
    assert rows.loc[rows.adaptation == "lora", "macro_mase_improvement_vs_zero_shot_pct"].isna().all()
    assert len(pd.read_csv(output / "pairwise_comparison.csv")) == 4
    assert any("misma configuración de inferencia" in message
               for message in json.loads((output / "comparison.json").read_text())["warnings"])


def test_zero_reference_error_has_a_delta_but_no_infinite_percentage(completed_run, tmp_path):
    output = compare_runs([completed_run("perfect", score=0.), completed_run("other", score=1.)], tmp_path / "comparison")
    pairs = pd.read_csv(output / "pairwise_comparison.csv")
    zero_reference = pairs.loc[pairs.value_b == 0.]
    assert (zero_reference.delta_a_minus_b == 1.).all()
    assert zero_reference.improvement_a_vs_b_pct.isna().all()
    assert not np.isinf(pairs.improvement_a_vs_b_pct).any()


def test_seed_summary_averages_repeated_seeds_before_reporting_variability(completed_run, tmp_path):
    paths = [completed_run("seed42a", score=2.), completed_run("seed42b", score=4.),
             completed_run("seed43", score=6.)]
    metadata = json.loads((paths[2] / "run.json").read_text())
    metadata["settings"]["seed"] = 43
    (paths[2] / "run.json").write_text(json.dumps(metadata))
    output = compare_runs(paths, tmp_path / "comparison")
    summary = pd.read_csv(output / "seed_summary.csv")
    assert len(summary) == 1
    assert summary.iloc[0]["num_seeds"] == 2
    assert summary.iloc[0]["num_runs"] == 3
    assert summary.iloc[0]["macro-mase-mean"] == 4.5
    assert summary.iloc[0]["macro-mase-std"] == pytest.approx(np.std([3., 6.], ddof=1))


def test_probabilistic_comparison_labels_and_warns_about_changed_postprocessing(completed_run, tmp_path):
    paths = [completed_run("historical"), completed_run("rearranged")]
    for path, crossings in zip(paths, (19, 0), strict=True):
        result_path = path / "test_metrics.json"
        result = json.loads(result_path.read_text())
        result["test"].update(crps=2., **{"quantile-crossings": crossings})
        result_path.write_text(json.dumps(result))
    metadata_path = paths[1] / "run.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["probabilistic_protocol"] = {"quantile_postprocessing": MEDIAN_ANCHORED_REARRANGEMENT}
    metadata_path.write_text(json.dumps(metadata))

    output = compare_runs(paths, tmp_path / "comparison")
    rows = pd.read_csv(output / "comparison.csv")
    assert rows["quantile_postprocessing"].tolist() == ["none", MEDIAN_ANCHORED_REARRANGEMENT]
    assert rows["quantile-crossings"].tolist() == [19, 0]
    assert any("diferencia de CRPS" in warning
               for warning in json.loads((output / "comparison.json").read_text())["warnings"])
    assert len(pd.read_csv(output / "seed_summary.csv")) == 2


@pytest.mark.parametrize("changed", ["model", "application_sources", "versions"])
def test_seed_summary_never_pools_different_checkpoints_or_software(completed_run, tmp_path, changed):
    paths = [completed_run("first"), completed_run("second")]
    metadata = json.loads((paths[1] / "run.json").read_text())
    if changed == "model":
        metadata["model"]["revision"] = "another-pinned-revision"
    else:
        metadata[changed] = {"changed": "revision"}
    (paths[1] / "run.json").write_text(json.dumps(metadata))
    output = compare_runs(paths, tmp_path / "comparison")
    summary = pd.read_csv(output / "seed_summary.csv")
    assert len(summary) == 2
    assert summary["num_seeds"].eq(1).all()
    assert summary["macro-mase-std"].isna().all()


def test_compare_reads_historical_scores_without_republishing_raw_errors(completed_run, tmp_path):
    legacy, current = completed_run("legacy", model="darts-nlinear"), completed_run("current")
    old_result = legacy / "test_metrics.json"
    old = json.loads(old_result.read_text())
    old["test"].update({"mae": 5., "macro-mae": 5., "rmse": 8., "baseline-mae": 6.,
                        "block-20250101-count": 48, "block-20250101-absolute-error-sum": 240.})
    old_result.write_text(json.dumps(old))
    new_result = current / "test_metrics.json"
    new = json.loads(new_result.read_text())
    new["test"].update({"block-20250101-mase-target-count": 48, "block-20250101-mase-scaled-error-sum": 96.})
    new_result.write_text(json.dumps(new))
    output = compare_runs([legacy, current], tmp_path / "comparison")
    rows = pd.read_csv(output / "comparison.csv")
    assert not {"mae", "rmse", "macro-mae", "baseline-mae"} & set(rows)
    assert rows.loc[rows["model"] == "darts-nlinear", "missing_policy"].item() == "strict"
    assert pd.read_csv(output / "block_bootstrap_comparison.csv").empty
    assert json.loads(old_result.read_text()) == old


@pytest.mark.parametrize("allow,matches", [(False, True), (True, False), (True, True)])
def test_distinct_training_artifacts_need_opt_in_and_matching_evaluation(completed_run, tmp_path, allow, matches):
    paths = [completed_run("first"), completed_run("second")]
    for index, path in enumerate(paths):
        metadata = json.loads((path / "run.json").read_text())
        metadata["manifest_sha256"] = f"training-{index}"
        metadata["manifest"]["evaluation_sha256"] = "common-evaluation" if matches else f"evaluation-{index}"
        (path / "run.json").write_text(json.dumps(metadata))
    if not (allow and matches):
        with pytest.raises(ValueError, match="manifests/paneles/ventanas"):
            compare_runs(paths, tmp_path / "comparison", allow_training_differences=allow)
    else:
        output = compare_runs(paths, tmp_path / "comparison", allow_training_differences=True)
        metadata = json.loads((output / "comparison.json").read_text())
        assert metadata["allow_training_differences"]
        assert any("evaluación idéntica por firma" in warning for warning in metadata["warnings"])
