import math
from copy import deepcopy
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
import torch
from flwr.app import MetricRecord, RecordDict
from torch.utils.data import ConcatDataset, TensorDataset

from fd_airquality.batching import CompleteBatchSampler, training_padding_counts
from fd_airquality.data import WindowDataset
from fd_airquality.settings import RunSettings
from fd_airquality.strategy import aggregate_training
from fd_airquality.task import evaluate_local, train_local


@pytest.mark.parametrize("size,batch_size", [(1, 8), (3, 8), (8, 8), (17, 8), (16, 8), (7, 1)])
def test_complete_batches_visit_every_window_before_filling_only_the_tail(size, batch_size):
    sampler = CompleteBatchSampler(size, batch_size, torch.Generator().manual_seed(13))
    batches = list(sampler)
    assert len(batches) == len(sampler) == math.ceil(size / batch_size)
    assert all(len(batch) == batch_size for batch in batches)
    flat = [index for batch in batches for index in batch]
    assert sorted(flat[:size]) == list(range(size))
    assert all(0 <= index < size for index in flat[size:])
    assert len(flat) - size == (-size) % batch_size


@pytest.mark.parametrize("size,batch_size", [(0, 8), (-1, 8), (3, 0), (3, -1)])
def test_sampler_rejects_empty_or_invalid_batches(size, batch_size):
    with pytest.raises(ValueError, match="positivos"):
        CompleteBatchSampler(size, batch_size, torch.Generator())


def test_sampler_is_reproducible_without_replaying_the_same_epoch_or_global_rng():
    sampler = CompleteBatchSampler(17, 8, torch.Generator().manual_seed(19))
    first, second = list(sampler), list(sampler)
    replay = CompleteBatchSampler(17, 8, torch.Generator().manual_seed(19))
    torch.manual_seed(1)
    state = torch.get_rng_state().clone()
    assert list(replay) == first
    assert list(replay) == second
    assert first != second
    assert list(CompleteBatchSampler(17, 8, torch.Generator().manual_seed(20))) != first
    torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)


@pytest.mark.parametrize("cap,batch_size", [(1, 8), (5, 3), (8, 4), (26, 8), (50, 7)])
def test_epoch_budget_resamples_the_full_pool_and_only_repeats_that_epochs_selection(cap, batch_size):
    size, draws = 31, min(31, cap)
    sampler = CompleteBatchSampler(size, batch_size, torch.Generator().manual_seed(17), max_windows=cap)
    replay = CompleteBatchSampler(size, batch_size, torch.Generator().manual_seed(17), max_windows=cap)
    seen = set()
    for _ in range(12):
        batches = list(sampler)
        flat = [index for batch in batches for index in batch]
        selected = set(flat[:draws])
        assert batches == list(replay)
        assert len(batches) == len(sampler) == math.ceil(draws / batch_size)
        assert all(len(batch) == batch_size for batch in batches)
        assert len(selected) == draws and selected <= set(range(size))
        assert set(flat[draws:]) <= selected
        seen.update(selected)
    if cap < size:
        assert len(seen) > cap


@pytest.mark.parametrize("cap", [0, -1, True, 2.5])
def test_sampler_rejects_invalid_epoch_budget(cap):
    with pytest.raises(ValueError, match="max_train_windows"):
        CompleteBatchSampler(13, 8, torch.Generator(), max_windows=cap)


@pytest.mark.parametrize("pad_last_batch", [0, 1])
def test_centralized_training_draws_a_quota_per_station_and_mixes_batches(pad_last_batch):
    inputs = torch.arange(23, dtype=torch.float32).reshape(-1, 1)
    actual = torch.ones(23, 1)
    boundaries = (0, 2, 8, 23)
    data = ConcatDataset([TensorDataset(inputs[a:b], actual[a:b]) for a, b in pairwise(boundaries)])
    model = torch.nn.Linear(1, 1)
    calls = []
    model.register_forward_pre_hook(lambda _, args: calls.append(args[0].flatten().tolist()))
    settings = RunSettings(batch_size=8, local_epochs=3, pad_last_batch=pad_last_batch,
                           early_stopping=0, lr_scheduler="none")
    metrics = train_local(model, data, settings, torch.device("cpu"), 23, max_train_windows=5)
    seen = set()
    assert len(calls) == metrics["optimizer-steps"] == 6
    for a, b in zip(calls[::2], calls[1::2], strict=True):
        assert len(a) == 8 and len(b) == (8 if pad_last_batch else 4)
        draws = a + b
        selected = set(draws[:12])
        assert [sum(low <= index < high for index in selected)
                for low, high in pairwise(boundaries)] == [2, 5, 5]
        assert set(draws[12:]) <= selected
        assert len({index < 8 for index in a}) > 1
        seen.update(selected)
    assert metrics["admissible-windows"] == 23
    assert metrics["unique-training-windows"] == len(seen) > metrics["num-examples"] == 12
    assert metrics["num-targets"] == 12
    assert metrics["replacement-window-occurrences"] == 12 * pad_last_batch


def test_target_support_uses_the_actual_epoch_draws_and_excludes_the_tail():
    size, cap, batch_size, epochs = 19, 5, 4, 3
    inputs = torch.arange(size, dtype=torch.float32).reshape(-1, 1)
    actual = torch.ones(size, 2)
    actual[::2, 1] = float("nan")
    model = torch.nn.Linear(1, 2)
    calls = []
    model.register_forward_pre_hook(lambda _, args: calls.extend(map(int, args[0][:, 0])))
    settings = RunSettings(batch_size=batch_size, local_epochs=epochs, early_stopping=0, lr_scheduler="none")
    metrics = train_local(model, TensorDataset(inputs, actual), settings, torch.device("cpu"), 23,
                          max_train_windows=cap)
    by_epoch = np.asarray(calls).reshape(epochs, -1)
    observed = torch.isfinite(actual).sum(dim=1).numpy()
    support = int(observed[by_epoch[:, :cap]].sum())
    assert len(set(by_epoch[:, :cap].flat)) == metrics["unique-training-windows"] > cap
    assert metrics["num-examples"] == cap and metrics["admissible-windows"] == size
    assert metrics["selected-target-occurrences"] == support
    assert metrics["num-targets"] == support / epochs
    assert metrics["sampled-target-occurrences"] == int(observed[by_epoch].sum()) > support


@pytest.mark.parametrize("size,batch_size", [(3, 8), (11, 8), (8, 8)])
@pytest.mark.parametrize("pad_last_batch", [0, 1])
def test_optimizer_respects_batch_tail_policy_and_federation_support(size, batch_size, pad_last_batch):
    torch.manual_seed(2)
    actual = torch.arange(size * 2, dtype=torch.float32).reshape(size, 2)
    actual[::2, 1] = float("nan")
    data = TensorDataset(torch.randn(size, 3), actual)
    settings = RunSettings(local_epochs=2, batch_size=batch_size, pad_last_batch=pad_last_batch,
                           early_stopping=0, lr_scheduler="none")
    model = torch.nn.Linear(3, 2)
    calls = []
    model.register_forward_pre_hook(lambda _, args: calls.append(args[0].clone()))
    metrics = train_local(model, data, settings, torch.device("cpu"), 23)
    expected_batches = 2 * math.ceil(size / batch_size)
    assert len(calls) == metrics["optimizer-steps"] == expected_batches
    expected_sizes = [batch_size] * math.ceil(size / batch_size)
    if not pad_last_batch:
        expected_sizes[-1] = size % batch_size or batch_size
    assert list(map(len, calls)) == expected_sizes * 2
    assert metrics["num-examples"] == size
    assert metrics["num-targets"] == int(torch.isfinite(actual).sum())
    assert metrics["sampled-window-occurrences"] == sum(expected_sizes) * 2
    assert metrics["replacement-window-occurrences"] == pad_last_batch * 2 * ((-size) % batch_size)
    assert metrics["sampled-target-occurrences"] >= 2 * metrics["num-targets"]
    if not pad_last_batch:
        assert metrics["sampled-target-occurrences"] == metrics["selected-target-occurrences"]
    # Only the true pre-replacement support is used for client aggregation.
    other = {"num-examples": 1, "num-targets": 1, "train-mse-normalized": 100.}
    pooled = aggregate_training([RecordDict({"metrics": MetricRecord(row)}) for row in (metrics, other)], "num-targets")
    assert pooled["num-targets"] == metrics["num-targets"] + 1
    assert pooled["train-mse-normalized"] == pytest.approx(
        (metrics["train-mse-normalized"] * metrics["num-targets"] + 100.) / pooled["num-targets"],
    )


@pytest.mark.parametrize("pad_last_batch", [0, 1])
def test_batch_tail_preserves_gradients_under_microbatch_accumulation(monkeypatch, pad_last_batch):
    data = TensorDataset(torch.randn(5, 4), torch.randn(5, 3))
    data.tensors[1][::2, 0] = float("nan")
    full = torch.nn.Linear(4, 3)
    micro = deepcopy(full)
    settings = RunSettings(batch_size=8, pad_last_batch=pad_last_batch)
    monkeypatch.setattr("fd_airquality.task._microbatch_size", lambda *_: 8)
    expected = train_local(full, data, settings, torch.device("cpu"), 19, .3)
    monkeypatch.setattr("fd_airquality.task._microbatch_size", lambda *_: 3)
    result = train_local(micro, data, settings, torch.device("cpu"), 19, .3)
    assert expected["replacement-window-occurrences"] == result["replacement-window-occurrences"] == 3 * pad_last_batch
    assert expected["sampled-target-occurrences"] == result["sampled-target-occurrences"]
    for key, value in full.state_dict().items():
        torch.testing.assert_close(micro.state_dict()[key], value)


def test_padding_counts_recurse_over_visited_centralized_windows():
    artifact = {"values": np.arange(6, dtype=np.float32), "timestamps": pd.date_range(
        "2025-01-01", periods=6, freq="h",
    ).asi8, "train_starts": np.array([-2, 1]), "mean": np.array(0.), "scale": np.array(1.),
        "allow_missing": np.array(True), "window_padding": np.array("both"), "train_target_end": np.array(5)}
    station = WindowDataset(artifact, "train", 3, 3)
    other = TensorDataset(torch.ones(1, 3), torch.tensor([[0., float("nan"), 1.]]))
    data = ConcatDataset([station, other])
    assert training_padding_counts(data) == (2, 2)
    assert training_padding_counts(data, np.array([False, True, True])) == (0, 2)


def test_empty_training_does_not_fabricate_windows():
    with pytest.raises(ValueError, match="No hay ventanas"):
        train_local(torch.nn.Linear(2, 1), TensorDataset(torch.empty(0, 2), torch.empty(0, 1)),
                    RunSettings(), torch.device("cpu"), 17)


@pytest.mark.parametrize("split", ["validation", "test"])
def test_evaluation_keeps_incomplete_batch_without_replacement(split):
    values = np.arange(14, dtype=np.float32)
    artifact = {"values": values, "timestamps": pd.date_range("2025-01-01", periods=14, freq="h").asi8,
                f"{split}_starts": np.arange(5), "mean": np.array(0.), "scale": np.array(1.),
                "allow_missing": np.array(True), "window_padding": np.array("both"),
                f"{split}_mase_scale": np.array(1.), f"{split}_rmsse_scale": np.array(1.),
                f"{split}_scale_pairs": np.array(5)}
    dataset = WindowDataset(artifact, split, 3, 2)
    model = torch.nn.Linear(3, 2)
    calls = []
    model.register_forward_pre_hook(lambda _, args: calls.append(args[0].clone()))
    metrics = evaluate_local(model, dataset, RunSettings(batch_size=4), torch.device("cpu"), artifact, 2)
    assert list(map(len, calls)) == [4, 1]
    torch.testing.assert_close(torch.cat(calls), torch.stack([dataset[index][0] for index in range(5)]))
    assert metrics["num-examples"] == 5 and metrics["num-targets"] == 10
