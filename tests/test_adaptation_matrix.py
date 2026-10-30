import pytest

from fd_airquality.cli import build_benchmark_matrix
from fd_airquality.models import GRANITE, validate_run_mode
from fd_airquality.settings import RunSettings


def test_full_foundation_matrix_has_12_federated_and_meaningful_controls():
    matrix = build_benchmark_matrix(list(GRANITE), ["federated", "local", "centralized", "zero-shot"],
                                    ["lora", "ffa-lora", "flora", "lora-fair"])
    assert len(matrix) == 27
    for model in GRANITE:
        rows = [row for row in matrix if row["model"] == model]
        assert {row["adaptation"] for row in rows if row["mode"] == "federated"} == {"lora", "ffa-lora", "flora", "lora-fair"}
        assert [row["adaptation"] for row in rows if row["mode"] == "zero-shot"] == ["none"]
        for mode in ("local", "centralized"):
            assert {row["adaptation"] for row in rows if row["mode"] == mode} == {"lora", "ffa-lora"}
    assert len({tuple(sorted(row.items())) for row in matrix}) == len(matrix)


def test_classic_models_are_not_assigned_lora_or_zero_shot():
    matrix = build_benchmark_matrix(["mlp", "autoencoder", "darts-nlinear"],
                                    ["federated", "local", "centralized", "zero-shot"], ["lora", "ffa-lora", "flora", "lora-fair"])
    assert len(matrix) == 9
    assert all(row["adaptation"] == "auto" and row["mode"] != "zero-shot" for row in matrix)


@pytest.mark.parametrize("adaptation,mode,strategy", [
    ("flora", "centralized", "fedavg"), ("lora-fair", "local", "fedavg"),
    ("flora", "federated", "fedprox"), ("lora-fair", "federated", "fedprox"),
])
def test_undefined_algorithm_combinations_fail_explicitly(adaptation, mode, strategy):
    with pytest.raises(ValueError):
        validate_run_mode(RunSettings(model="flowstate-r1", adaptation=adaptation, strategy=strategy), mode)


@pytest.mark.parametrize("kwargs", [
    {"adaptation": "unknown"}, {"lora_rank": 0}, {"lora_rank": True},
    {"lora_alpha": 0}, {"lora_dropout": 1}, {"fair_iterations": 0},
    {"fair_lambda": -1}, {"fair_learning_rate": float("nan")}, {"selection_metric": "test-mae"},
])
def test_invalid_adapter_hyperparameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        RunSettings(**kwargs)
