import json
from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from fd_airquality.comparison import compare_runs


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
        metrics = {"mae": score, "rmse": score, "macro-mase": score, "macro-rmsse": score,
                   "baseline-mae": 5., "num-clients": 2, "num-targets": 48,
                   "macro-mase-stations": 2, "macro-rmsse-stations": 2}
        (path / "run.json").write_text(json.dumps(run))
        (path / "test_metrics.json").write_text(json.dumps({"test": metrics, "selected_round": 0}))
        pd.DataFrame([{"split": "test", "partition-id": i, "station": f"station-{i}",
                       "num-examples": 2, "num-targets": 24,
                       "mae": score, "rmse": score, "mase": score, "rmsse": score}
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
    assert len(pairs) == 5 * 4 * 4
    assert not pairs.delta_a_minus_b.isna().any()
    a = rows.loc[rows.model == "flowstate-r1", "run"].item()
    b = rows.loc[rows.model == "mlp", "run"].item()
    match = pairs.loc[(pairs.run_a == a) & (pairs.run_b == b) & (pairs.metric == "macro-mase")].iloc[0]
    assert match.delta_a_minus_b == -4.
    assert match.improvement_a_vs_b_pct == 80.
    stations = pd.read_csv(output / "station_pairwise_comparison.csv")
    assert len(stations) == len(pairs) * 2
    assert set(stations["partition-id"]) == {0, 1}
    assert rows.loc[rows.model == "flowstate-r1", "flowstate_scale_factor"].item() == 2.


def test_zero_shot_reference_matches_inference_scale_without_filtering_other_pairs(completed_run, tmp_path):
    zero1 = completed_run("zero1", model="flowstate-r1", mode="zero-shot", adaptation="none", scale=1., score=4.)
    zero2 = completed_run("zero2", model="flowstate-r1", mode="zero-shot", adaptation="none", scale=2., score=2.)
    adapted = completed_run("adapted", model="flowstate-r1", adaptation="lora", scale=2., score=1.)
    output = compare_runs([zero1, zero2, adapted], tmp_path / "comparison")
    rows = pd.read_csv(output / "comparison.csv")
    assert rows.loc[rows.adaptation == "lora", "macro_mase_improvement_vs_zero_shot_pct"].item() == 50.
    assert len(pd.read_csv(output / "pairwise_comparison.csv")) == 3 * 2 * 4


def test_mismatched_zero_shot_scale_only_disables_attribution_not_comparison(completed_run, tmp_path):
    zero = completed_run("zero", model="flowstate-r1", mode="zero-shot", adaptation="none", scale=1.)
    adapted = completed_run("adapted", model="flowstate-r1", adaptation="lora", scale=2.)
    output = compare_runs([zero, adapted], tmp_path / "comparison")
    rows = pd.read_csv(output / "comparison.csv")
    assert rows.loc[rows.adaptation == "lora", "macro_mase_improvement_vs_zero_shot_pct"].isna().all()
    assert len(pd.read_csv(output / "pairwise_comparison.csv")) == 8
    assert any("misma configuración de inferencia" in message
               for message in json.loads((output / "comparison.json").read_text())["warnings"])


def test_zero_reference_error_has_a_delta_but_no_infinite_percentage(completed_run, tmp_path):
    output = compare_runs([completed_run("perfect", score=0.), completed_run("other", score=1.)], tmp_path / "comparison")
    pairs = pd.read_csv(output / "pairwise_comparison.csv")
    zero_reference = pairs.loc[pairs.value_b == 0.]
    assert (zero_reference.delta_a_minus_b == 1.).all()
    assert zero_reference.improvement_a_vs_b_pct.isna().all()
    assert not np.isinf(pairs.improvement_a_vs_b_pct).any()
