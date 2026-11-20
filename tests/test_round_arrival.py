import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest
import torch
from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict

from fd_airquality.arrival import (
    arrival_metrics,
    audit_normalization,
    normalization_artifact,
    round_bounds,
    round_cutoff,
    round_epoch_limit,
    round_validation_bounds,
    training_view,
)
from fd_airquality.data import DataConfig, WindowDataset, load_client_data, prepare
from fd_airquality.server_app import run_server
from fd_airquality.settings import RunSettings
from fd_airquality.strategy import AuditedStrategy


def test_rounds_receive_only_released_calendar_prefixes(arrival_prepared):
    path, manifest = arrival_prepared
    geometry = manifest["geometry"]
    _, artifact = load_client_data(str(path), 0)
    settings = RunSettings(num_server_rounds=3, arrival_period_hours=24)
    local = normalization_artifact(artifact, geometry, settings)
    original = artifact["values"].copy()
    views = [training_view(local, geometry, settings, number) for number in (1, 2, 3)]
    used_windows, used_targets = set(), set()
    for number, dataset in enumerate(views, 1):
        cutoff = round_cutoff(geometry, settings, number)
        assert len(dataset.values) < len(artifact["values"])
        assert dataset.timestamps.max() < cutoff.value
        first = artifact["timestamps"][dataset.starts + geometry["context"]]
        last = artifact["timestamps"][dataset.starts + geometry["context"] + geometry["horizon"] - 1]
        assert (last < cutoff.value).all()
        assert (last < pd.Timestamp(geometry["validation_start"]).value).all()
        if number > 1:
            assert (first >= (cutoff - pd.Timedelta(hours=24)).value).all()
            assert not used_windows.intersection(dataset.starts)
            assert dataset.arrival_lower == round_cutoff(geometry, settings, number - 1)
            assert artifact["timestamps"][dataset.starts[0]] < dataset.arrival_lower.value
        positions = dataset.starts[:, None] + dataset.context + np.arange(dataset.horizon)
        targets = set(artifact["timestamps"][positions][np.isfinite(artifact["values"][positions])])
        assert not used_targets.intersection(targets)
        used_windows.update(dataset.starts)
        used_targets.update(targets)
    assert len(views[0].values) < len(views[-1].values)
    np.testing.assert_array_equal(artifact["values"], original)


def test_default_blocks_reuse_context_but_never_previous_targets():
    times = pd.date_range("2025-01-01", periods=96, freq="h").asi8
    values = np.arange(96, dtype=np.float32)
    starts = np.arange(64)
    artifact = {"values": values, "timestamps": times, "mean": np.array(0.), "scale": np.array(1.),
                "train_starts": starts, "allow_missing": np.array(True)}
    geometry = {"context": 6, "horizon": 3, "split_mode": "global", "validation_start": "2025-01-04"}
    settings = RunSettings(num_server_rounds=3)
    views = [training_view(artifact, geometry, settings, number) for number in (1, 2, 3)]
    np.testing.assert_array_equal(views[0].starts, np.arange(16))
    np.testing.assert_array_equal(views[1].starts, np.arange(18, 40))
    np.testing.assert_array_equal(views[2].starts, np.arange(42, 64))
    context, targets = views[1][0]
    torch.testing.assert_close(context, torch.arange(18., 24.))
    torch.testing.assert_close(targets, torch.arange(24., 27.))
    assert views[1].timestamps[-1] == times[47]
    metrics = arrival_metrics(views[1])
    assert metrics["available-from-ns"] == times[24]
    assert views[1].target_time_bounds() == (times[24], times[47])
    # A target horizon crossing a block boundary is not replayed or backdated.
    used = set().union(*(set(view.starts) for view in views))
    assert set(starts) - used == {16, 17, 40, 41}
    np.testing.assert_array_equal(artifact["values"], values)
    # An explicit no-arrival control still trains with the whole fold.
    all_data = training_view(artifact, geometry, replace(settings, arrival_period_hours=0), 2)
    np.testing.assert_array_equal(all_data.starts, starts)
    assert all_data.arrival_lower is all_data.available_through is None
    # A block shorter than H must not fall back to old training windows.
    short = training_view(artifact, geometry, replace(settings, arrival_period_hours=1), 2)
    assert len(short) == 0
    assert arrival_metrics(short)["available-from-ns"] == times[70]


def test_epoch_limits_only_reduce_updates_with_new_arrivals(arrival_prepared):
    _, manifest = arrival_prepared
    geometry = manifest["geometry"]
    s = RunSettings(local_epochs=40, federated_update_epochs=20)
    assert [round_epoch_limit(geometry, s, r) for r in range(1, 6)] == [40, 20, 20, 20, 20]
    assert round_epoch_limit(geometry, replace(s, arrival_period_hours=0), 2) == 40
    assert round_epoch_limit(geometry | {"split_mode": "station"}, s, 2) == 40
    assert round_epoch_limit(geometry, replace(s, federated_update_epochs=0), 2) == 40
    assert round_epoch_limit(geometry, replace(s, local_epochs=1), 2) == 1


def test_availability_filters_full_round_pool_before_epoch_budget(arrival_prepared):
    path, manifest = arrival_prepared
    _, artifact = load_client_data(str(path), 0)
    geometry = manifest["geometry"] | {"max_train_windows": 7}
    strict = RunSettings(num_server_rounds=3, training_data_availability_threshold="[1.0,1.0]")
    permissive = replace(strict, training_data_availability_threshold="0.0")
    for stage in (1, 2, 3):
        a = training_view(artifact, geometry, strict, stage)
        b = training_view(artifact, geometry, permissive, stage)
        assert len(a) < len(b) and len(b) >= 8
        assert a.availability_excluded_windows > 0
        assert a.candidate_windows == b.candidate_windows
        for start in a.starts:
            assert np.isfinite(a.values[start:start + 9]).all()
    for name in ("validation", "test"):
        assert len(WindowDataset(artifact, name, 6, 3)) == manifest["clients"][0][f"num_{name}_windows"]


def test_fixed_normalization_is_masked_and_audited_per_fold(arrival_prepared, tmp_path):
    path, manifest = arrival_prepared
    _, artifact = load_client_data(str(path), 0)
    settings = RunSettings(num_server_rounds=3, normalization="fixed")
    local = normalization_artifact(artifact, manifest["geometry"], settings)
    first = round_cutoff(manifest["geometry"], settings, 1)
    observed = artifact["values"][artifact["timestamps"] < first.value]
    observed = observed[np.isfinite(observed)].astype(np.float64)
    assert local["mean"] == pytest.approx(observed.mean())
    assert local["scale"] == pytest.approx(observed.std())
    output = tmp_path / "fold"
    audit_normalization(local, settings, 0, output / "clients")
    assert json.loads((output / "clients/normalization_000.json").read_text())["sha256"] == local["normalization_sha256"]
    altered = artifact | {"values": artifact["values"] + 1}
    assert normalization_artifact(altered, manifest["geometry"], settings)["normalization_sha256"] != local["normalization_sha256"]
    with pytest.raises(ValueError, match="aún no disponibles"):
        normalization_artifact(artifact, manifest["geometry"], replace(
            settings, normalization_reference_end=manifest["geometry"]["validation_start"],
        ))


@pytest.mark.parametrize(("model", "strategy"), [
    ("mlp", "fedavg"), ("nlinear", "fedprox"), ("nhits", "fedavg"), ("autoencoder", "fedprox"),
])
def test_actual_flower_strategy_uses_declared_round_blocks_and_no_early_validation(
    arrival_prepared, tmp_path, flower_grid, model, strategy,
):
    path, manifest = arrival_prepared
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"),
                            model=model, strategy=strategy, num_server_rounds=3, hidden_size=8,
                            batch_size=32, pad_last_batch=0, probabilistic=int(model != "autoencoder"),
                            arrival_period_hours=48, local_epochs=4, federated_update_epochs=2,
                            early_stopping=0, lr_scheduler="none",
                            mlflow_tracking_uri="disabled")
    grid = flower_grid(settings, len(manifest["clients"]))
    output = run_server(grid, settings)
    history = pd.read_csv(output / "station_metrics.csv")
    train = history.loc[history.split == "train"]
    for _, station in train.groupby("station"):
        assert station.iloc[0]["num-examples"] > station.iloc[1:]["num-examples"].max()
        assert station.iloc[1:]["num-examples"].le(16).all()
        assert (station.iloc[1:]["available-from-ns"].to_numpy()
                == station.iloc[:-1]["available-through-ns"].to_numpy()).all()
        assert (station.iloc[1:]["train-first-target-ns"] >= station.iloc[1:]["available-from-ns"]).all()
        assert np.all(np.diff(station["available-through-ns"]) == pd.Timedelta(hours=48).value)
        assert (station["train-last-target-ns"] < station["available-through-ns"]).all()
        assert station["effective-batch-size"].eq(32).all()
        assert station["epochs-completed"].tolist() == [4, 2, 2]
        assert station.iloc[1:]["optimizer-steps"].tolist() == [2, 2]
        assert station["replacement-window-occurrences"].eq(0).all()
    validations = [config for kind, config in grid.requests if config.get("split") == "validation"]
    assert len(validations) == 2 and all(config["server-round"] == 3 for config in validations)
    result = json.loads((output / "test_metrics.json").read_text())
    assert result["selected_round"] == 3
    assert result["test"]["num-targets"] == sum(client["num_test_targets"] for client in manifest["clients"])
    if model != "autoencoder":
        assert np.isfinite(result["test"]["crps"])
        assert result["test"]["crps-count"] == result["test"]["num-targets"]
    last = torch.load(output / "last_model.pt", weights_only=True)
    assert last["state_role"] == "training"
    assert last["information_cutoff"] == manifest["geometry"]["validation_start"]
    metadata = json.loads((output / "run.json").read_text())
    arrival = metadata["arrival_protocol"]
    assert arrival["training_targets"] == "current_block_only"
    assert arrival["history"] == "new-only"
    assert arrival["block_schedule"] == "initial_prefix_then_calendar_increments"
    assert metadata["training_budget"]["federated_update_epochs"] == 2
    assert metadata["window_protocol"]["short_training_batch"] == "keep_without_replacement"


@pytest.mark.parametrize("model_name", ["mlp", "flowstate-r1"])
def test_new_blocks_continue_from_the_previous_aggregated_weights(
    arrival_prepared, tmp_path, monkeypatch, flower_grid, mock_granite, model_name,
):
    from fd_airquality.client_app import train_local
    from fd_airquality.models import shared_state

    path, _ = arrival_prepared
    incoming, aggregated = {}, {}

    def train(model, dataset, settings, device, seed, proximal_mu=0., **kwargs):
        number = (seed - settings.seed) // 100_003
        incoming.setdefault(number, []).append({
            key: value.detach().clone() for key, value in shared_state(model).items()
        })
        return train_local(model, dataset, settings, device, seed, proximal_mu, **kwargs)

    aggregate_train = AuditedStrategy.aggregate_train

    def aggregate(self, server_round, replies):
        arrays, metrics = aggregate_train(self, server_round, replies)
        aggregated[server_round] = {
            key: value.detach().clone() for key, value in arrays.to_torch_state_dict().items()
        }
        return arrays, metrics

    monkeypatch.setattr("fd_airquality.client_app.train_local", train)
    monkeypatch.setattr(AuditedStrategy, "aggregate_train", aggregate)
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"), model=model_name,
                           lora_targets="proj" if model_name == "flowstate-r1" else "auto",
                           missing_policy="ffill" if model_name == "flowstate-r1" else "auto",
                           num_server_rounds=3, hidden_size=8, batch_size=16, mlflow_tracking_uri="disabled")
    run_server(flower_grid(settings, 2), settings)
    assert set(incoming) == {1, 2, 3}
    for number in (2, 3):
        assert len(incoming[number]) == 2
        for state in incoming[number]:
            for key, expected in aggregated[number - 1].items():
                torch.testing.assert_close(state[key], expected, rtol=0, atol=0)
    assert any(not torch.equal(aggregated[1][key], value) for key, value in incoming[1][0].items())


@pytest.mark.parametrize("violation", ["old_target", "future_target", "wrong_block", "missing_interval"])
def test_server_rejects_training_outside_the_new_block(arrival_prepared, tmp_path, flower_identity, violation):
    path, manifest = arrival_prepared
    settings = RunSettings(manifest=str(path), num_server_rounds=3)
    strategy = AuditedStrategy(settings, manifest, tmp_path)
    lower, upper = round_bounds(manifest["geometry"], settings, 2)
    request = Message(content=RecordDict(), dst_node_id=10, message_type="train")
    replies = []
    for partition in range(2):
        metrics = {
            "partition-id": partition, "num-examples": 1, "num-targets": 1, "train-mse-normalized": 1.,
            "available-from-ns": lower.value, "available-through-ns": upper.value,
            "train-first-target-ns": lower.value, "train-last-target-ns": upper.value - 1,
        }
        replies.append(Message(content=RecordDict({
            "arrays": ArrayRecord([np.array([1.], dtype=np.float32)]), "metrics": MetricRecord(metrics),
        }), reply_to=request))
    metrics = replies[0].content["metrics"]
    if violation == "old_target":
        metrics["train-first-target-ns"] = lower.value - 1
        reason = "bloque anterior"
    elif violation == "future_target":
        metrics["train-last-target-ns"] = upper.value
        reason = "aún no ha llegado"
    elif violation == "wrong_block":
        metrics["available-from-ns"] = lower.value - 1
        reason = "bloque de llegada"
    else:
        del metrics["train-first-target-ns"]
        reason = "no acredita"
    with pytest.raises(ValueError, match=reason):
        strategy.aggregate_train(2, replies)


def test_server_checks_expanded_monitoring_reserve(arrival_prepared, tmp_path):
    _, manifest = arrival_prepared
    geometry = manifest["geometry"]
    settings = RunSettings(local_epochs=4, num_server_rounds=3, round_validation_max_hours=48)
    strategy = AuditedStrategy(settings, manifest, tmp_path)
    start, end = round_validation_bounds(geometry, settings)
    expanded = start - pd.Timedelta(hours=6)
    metrics = {"monitoring-from-ns": expanded.value, "monitoring-through-ns": end.value,
               "num-examples": 1, "train-last-target-ns": expanded.value - 1}
    strategy._check_monitoring_interval(metrics, geometry, 1, 0)
    with pytest.raises(ValueError, match="no respeta el histórico"):
        strategy._check_monitoring_interval(metrics | {"monitoring-through-ns": end.value + 1}, geometry, 1, 0)
    with pytest.raises(ValueError, match="cambió entre rondas"):
        strategy._check_monitoring_interval(metrics | {"monitoring-from-ns": start.value}, geometry, 2, 0)
    with pytest.raises(ValueError, match="utiliza targets de monitorización"):
        strategy._check_monitoring_interval(metrics | {"train-last-target-ns": expanded.value}, geometry, 1, 0)


def test_empty_early_round_is_explicit_and_not_averaged_as_an_update(arrival_prepared, tmp_path, flower_grid):
    path, _manifest = arrival_prepared
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"), num_server_rounds=3,
                            arrival_period_hours=96, hidden_size=8,
                           mlflow_tracking_uri="disabled")
    output = run_server(flower_grid(settings, 2), settings)
    history = pd.read_csv(output / "round_metrics.csv").query("split == 'train'")
    assert history.iloc[0]["num-targets"] == 0
    assert history.iloc[0]["idle-client"] == 2
    assert history.iloc[-1]["num-targets"] > 0


def test_empty_new_block_keeps_weights_instead_of_replaying_old_data(arrival_raw, tmp_path, monkeypatch, flower_grid):
    raw, config = arrival_raw
    for source in raw.glob("*/*.csv"):
        frame = pd.read_csv(source, index_col=0, parse_dates=True)
        frame.loc[(frame.index >= "2025-01-08") & (frame.index < "2025-01-09")] = np.nan
        frame.to_csv(source)
    path = prepare(raw, tmp_path / "no_last_block", "PM25", config)
    states = {}
    save = AuditedStrategy._save

    def record(self, path, arrays, number):
        if path.name == "last_model.pt":
            states[number] = {key: value.detach().clone() for key, value in arrays.to_torch_state_dict().items()}
        return save(self, path, arrays, number)

    monkeypatch.setattr(AuditedStrategy, "_save", record)
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"), num_server_rounds=3,
                           hidden_size=8, batch_size=32, pad_last_batch=0, mlflow_tracking_uri="disabled")
    output = run_server(flower_grid(settings, 2), settings)
    rounds = pd.read_csv(output / "round_metrics.csv").query("split == 'train'").set_index("round")
    assert rounds.loc[2, "num-targets"] > 0
    assert rounds.loc[3, "num-targets"] == rounds.loc[3, "optimizer-steps"] == 0
    assert rounds.loc[3, "idle-client"] == 2
    for key, expected in states[2].items():
        torch.testing.assert_close(states[3][key], expected, rtol=0, atol=0)


def test_masked_block_diagnostics_account_for_every_retained_window(arrival_prepared):
    path, manifest = arrival_prepared
    blocks = pd.read_csv(path.parent / "blocks.csv")
    assert "windows_crossing_missing_values" in set(blocks.kind)
    for client in manifest["clients"]:
        for split in ("train", "validation", "test"):
            assert blocks.loc[blocks.station == client["station"], f"num_{split}_windows"].sum() == client[f"num_{split}_windows"]


def test_fixed_training_scale_reference_is_explicit_and_never_uses_validation():
    from fd_airquality.data import split_station

    times = pd.date_range("2025-01-01", periods=400, freq="h")
    values = pd.Series(np.arange(400, dtype=float), index=times)
    config = DataConfig(context=6, horizon=3, stride=3, seasonality=3,
                        error_scale_reference="training", error_scale_hours=48,
                        min_validation_windows=2, min_test_windows=2,
                        validation_start="2025-01-09", test_start="2025-01-11", test_end="2025-01-13")
    split = split_station(values, config)
    changed = values.copy()
    changed.loc[config.validation_start:] *= 1000
    other = split_station(changed, config)
    assert split["validation_scale_end"] == split["test_scale_end"] == "2025-01-09 00:00:00"
    assert split["validation_mase_scale"] == split["test_mase_scale"] == other["test_mase_scale"]
    assert split["test_rmsse_scale"] == other["test_rmsse_scale"]
    assert config == DataConfig(**asdict(config))


def test_scaled_scores_compare_canonical_labels_across_normalizations(arrival_prepared, monkeypatch):
    from fd_airquality.task import evaluate_local

    path, manifest = arrival_prepared
    _, artifact = load_client_data(str(path), 0)
    scores = []
    for normalization in ("window", "fixed", "none"):
        settings = RunSettings(normalization=normalization, hidden_size=8, num_server_rounds=3)
        local = normalization_artifact(artifact, manifest["geometry"], settings)
        dataset = WindowDataset(local, "test", 6, 3)
        normalized_prediction = (10. - dataset.mean) / dataset.scale

        def constant_forecast(_model, inputs, _settings, value=normalized_prediction):
            return inputs.new_full((len(inputs), 3), value, dtype=torch.float64)

        monkeypatch.setattr("fd_airquality.task.forecast", constant_forecast)
        stats = evaluate_local(torch.nn.Identity(), dataset, settings, torch.device("cpu"), local, 3)
        scores.append((stats["mase"], stats["rmsse"]))
    for score in scores[1:]:
        assert score == pytest.approx(scores[0])


def test_pre_cutoff_inactivity_does_not_use_a_future_reappearance(tmp_path):
    folder = tmp_path / "raw" / "station"
    folder.mkdir(parents=True)
    times = pd.date_range("2025-01-01", "2025-04-03", freq="5min", inclusive="left")
    values = 10 + np.sin(np.arange(len(times)) / 23)
    values[(times >= "2025-01-02") & (times < "2025-03-15")] = np.nan
    pd.DataFrame({"PM25": values}, index=times).to_csv(folder / "PM25.csv")
    config = DataConfig(context=6, horizon=3, stride=3, window_policy="masked", seasonality=3,
                        hourly_repeat_filter=0, min_validation_windows=2, min_test_windows=2,
                        validation_start="2025-03-01", test_start="2025-03-29", test_end="2025-04-02")
    path = prepare(folder.parent, tmp_path / "prepared", "PM25", config)
    row = pd.read_csv(path.parent / "stations.csv").iloc[0]
    prior = pd.Timestamp(row["last_observation_before_validation"])
    assert prior == pd.Timestamp("2025-01-01 23:00:00")
    assert row["inactive_days_before_validation"] == pytest.approx(
        (pd.Timestamp(config.validation_start) - prior).total_seconds() / 86400,
    )
