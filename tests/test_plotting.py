import json

import numpy as np
import pandas as pd
import pytest

from fd_airquality.cli import main
from fd_airquality.plotting import cross_pairs, load_comparison, matched_pairs, plot_comparison


@pytest.fixture
def comparison(tmp_path):
    source = tmp_path / "benchmark" / "comparison"
    source.mkdir(parents=True)
    records = []
    for model, adaptations in (("nlinear", ("full",)), ("ttm-r3", ("lora", "ffa-lora", "flora", "lora-fair"))):
        for mode in ("centralized", "federated"):
            for adaptation in adaptations:
                if mode == "centralized" and adaptation in {"flora", "lora-fair"}:
                    continue
                value = 2. if mode == "centralized" else 1.5
                records.append({
                    "run": f"{model}/{adaptation}/{mode}/run", "model": model, "mode": mode,
                    "adaptation": adaptation, "strategy": "fedavg" if mode == "federated" else "none",
                    "seed": 42, "lora_rank": 4, "revision": "pinned", "flowstate_scale_factor": None,
                    "missing_policy": "native", "selection_metric": "validation_macro-mase",
                    "num-clients": 2, "num-examples": 146,
                    "macro-mase": value, "macro-rmsse": value + 1, "mae": value + 2, "rmse": value + 3,
                })
    pd.DataFrame(records).to_csv(source / "comparison.csv", index=False)
    (source / "comparison.json").write_text(json.dumps({
        "manifest_sha256": "audited-panel", "evaluation_sha256": "audited-evaluation",
        "allow_training_differences": True, "warnings": ["Aviso del comparador"],
    }))
    return source


def test_pairs_preserve_adapter_identity_and_label_fl_only_controls(comparison):
    _, rows, _ = load_comparison(comparison / "comparison.csv")
    pairs, warnings = matched_pairs(rows)
    assert not warnings
    assert len(pairs) == 5
    assert np.allclose(pairs["macro-mase_improvement_pct"], 25.)
    assert np.allclose(pairs["macro-mase_delta"], -.5)
    for _, row in pairs.iterrows():
        if row.federated_adaptation in {"flora", "lora-fair"}:
            assert row.reference_kind == "lora_control"
            assert row.centralized_adaptation == "lora"
            assert "LoRA*" in row.label
        else:
            assert row.reference_kind == "same_adaptation"
            assert row.federated_adaptation == row.centralized_adaptation


@pytest.mark.parametrize("key,value", [
    ("seed", 43), ("lora_rank", 8), ("revision", "other"), ("flowstate_scale_factor", 2.),
    ("missing_policy", "ffill"), ("selection_metric", "validation_mae"),
])
def test_pairs_do_not_match_incompatible_configurations(comparison, key, value):
    _, rows, _ = load_comparison(comparison)
    rows.loc[(rows["mode"] == "centralized") & (rows["model"] == "ttm-r3"), key] = value
    pairs, warnings = matched_pairs(rows)
    assert len(pairs) == 1  # Only the full NLinear has a compatible control.
    assert len(warnings) == 4


def test_ambiguous_controls_are_not_selected_by_test_score(comparison):
    _, rows, _ = load_comparison(comparison)
    extra = rows.loc[(rows["mode"] == "centralized") & (rows["adaptation"] == "lora")].copy()
    extra["macro-mase"] = .01
    pairs, warnings = matched_pairs(pd.concat([rows, extra], ignore_index=True))
    assert len(pairs) == 2  # Full NLinear and FFA-LoRA, never the lowest-error LoRA control.
    assert len(warnings) == 3


def test_all_cross_model_pairs_and_zero_denominators(comparison):
    _, rows, _ = load_comparison(comparison)
    rows.loc[rows["mode"] == "centralized", "mae"] = 0.
    pairs = cross_pairs(rows)
    assert len(pairs) == 5 * 3 * 4
    assert pairs.loc[pairs.metric == "mae", "improvement_pct"].isna().all()
    assert not np.isinf(pairs.improvement_pct).any()
    matched, _ = matched_pairs(rows)
    assert matched["mae_improvement_pct"].isna().all()


def test_generation_is_offline_preserves_sources_and_exports_figures(comparison):
    original = {path.name: path.read_bytes() for path in comparison.iterdir()}
    output = plot_comparison(comparison.parent)
    manifest = json.loads((output / "plots.json").read_text())
    assert manifest["manifest_sha256"] == "audited-panel"
    assert manifest["evaluation_sha256"] == "audited-evaluation"
    assert manifest["allow_training_differences"]
    assert manifest["warnings"] == ["Aviso del comparador"]
    assert len(manifest["figures"]) == 6  # Four summaries plus one per model.
    for figure in manifest["figures"]:
        for extension in ("png", "pdf"):
            assert (output / f"{figure['stem']}.{extension}").stat().st_size > 1000
    assert "LoRA centralizado" in (output / "index.html").read_text()
    assert len(pd.read_csv(output / "matched_pairs.csv")) == 5
    assert len(pd.read_csv(output / "cross_model_pairs.csv")) == 60
    assert len(pd.read_csv(output / "plotted_metrics.csv")) == 8
    for name, content in original.items():
        assert (comparison / name).read_bytes() == content
    with pytest.raises(ValueError, match="ya existe"):
        plot_comparison(comparison)


def test_panel_without_scaled_scores_still_plots_mae_and_rmse(comparison):
    csv = comparison / "comparison.csv"
    rows = pd.read_csv(csv).query("model == 'nlinear'").drop(columns=["macro-mase", "macro-rmsse"])
    rows.to_csv(csv, index=False)
    output = plot_comparison(csv)
    metadata = json.loads((output / "plots.json").read_text())
    assert not any(figure["stem"] == "04_pares_entre_modelos" for figure in metadata["figures"])
    assert any("No hay MASE macro" in warning for warning in metadata["warnings"])
    pairs = pd.read_csv(output / "cross_model_pairs.csv")
    assert pairs.loc[pairs.metric.isin(["macro-mase", "macro-rmsse"]), "improvement_pct"].isna().all()
    assert pairs.loc[pairs.metric.isin(["mae", "rmse"]), "improvement_pct"].gt(0).all()


def test_missing_mode_and_duplicate_runs_are_rejected(comparison):
    csv = comparison / "comparison.csv"
    rows = pd.read_csv(csv)
    pd.concat([rows, rows.iloc[[0]]]).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="duplicadas"):
        load_comparison(comparison)
    rows.loc[rows["mode"] == "centralized"].to_csv(csv, index=False)
    with pytest.raises(ValueError, match="federados y centralizados"):
        load_comparison(comparison)


def test_plot_cli_only_invokes_plotting(comparison, monkeypatch):
    received = []
    monkeypatch.setattr("fd_airquality.cli.resolve_manifest", lambda *_: pytest.fail("Plot must not resolve a manifest"))
    monkeypatch.setattr("fd_airquality.plotting.plot_comparison", lambda *args: received.append(args))
    main(["plot", str(comparison), "--output", str(comparison / "custom")])
    assert received == [(comparison, comparison / "custom")]
