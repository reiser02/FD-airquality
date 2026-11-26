import pytest

from fd_airquality.backtesting import _candidate_plan, build_candidate_matrix
from fd_airquality.models import GRANITE, validate_run_mode
from fd_airquality.settings import FEDERATED_MODES, GRANITE_ADAPTATIONS, RUN_MODES, RunSettings


def test_full_foundation_matrix_includes_zero_shot_in_each_mode():
    matrix = build_candidate_matrix(list(GRANITE), RUN_MODES, GRANITE_ADAPTATIONS)
    assert len(matrix) == 48
    for model in GRANITE:
        rows = [row for row in matrix if row["model"] == model]
        for mode in FEDERATED_MODES:
            assert {row["adaptation"] for row in rows if row["mode"] == mode} == set(GRANITE_ADAPTATIONS)
        for mode in ("local", "centralized"):
            assert {row["adaptation"] for row in rows if row["mode"] == mode} == {"lora", "ffa-lora", "zero-shot"}
    assert len({tuple(sorted(row.items())) for row in matrix}) == len(matrix)


def test_classic_models_are_not_assigned_lora_or_zero_shot():
    matrix = build_candidate_matrix(["mlp", "autoencoder", "autoencoder-tiny", "nlinear"],
                                    RUN_MODES, GRANITE_ADAPTATIONS)
    assert len(matrix) == 16
    assert all(row["adaptation"] == "auto" for row in matrix)


def test_zero_shot_only_does_not_add_lora_or_repeat_over_training_hyperparameters():
    plan = _candidate_plan(RunSettings(), ["ttm-r3"], RUN_MODES, ["zero-shot"],
                           [42, 43], [4, 8], ["fedavg", "fedprox"])
    assert [(case["mode"], case["adaptation"]) for case in plan] == [
        (mode, "zero-shot") for mode in RUN_MODES
    ]


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
