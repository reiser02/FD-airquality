import json
from copy import deepcopy

import pytest
import torch

from fd_airquality.baselines import run_baseline
from fd_airquality.settings import RunSettings
from fd_airquality.task import configure_torch, resolve_device, validate_selection_coverage


def test_cpu_does_not_probe_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CPU no necesita CUDA"))
    assert resolve_device("cpu") == torch.device("cpu")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_configure_torch_does_not_override_determinism(monkeypatch, device):
    monkeypatch.setattr("fd_airquality.task.resolve_device", lambda _: torch.device(device))
    monkeypatch.setattr(torch, "use_deterministic_algorithms",
                        lambda *_: pytest.fail("No debe imponer una política de determinismo"))

    assert configure_torch(RunSettings(device=device), 42) == torch.device(device)


def test_unavailable_cuda_warns_and_returns_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(UserWarning, match="CUDA no disponible; se usará CPU"):
        assert configure_torch(RunSettings(device="cuda"), 42) == torch.device("cpu")


@pytest.mark.parametrize("failure", ["allocation", "kernel", "synchronization"])
def test_cuda_probe_failure_warns_and_returns_cpu(monkeypatch, failure):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    class Probe:
        def add_(self, _):
            if failure == "kernel":
                raise RuntimeError("no kernel image is available for execution on the device")

    def allocate(*args, **kwargs):
        assert kwargs["device"] == "cuda"
        if failure == "allocation":
            raise RuntimeError("CUDA allocation failed")
        return Probe()

    def synchronize():
        if failure == "synchronization":
            raise RuntimeError("CUDA synchronization failed")

    monkeypatch.setattr(torch, "ones", allocate)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    with pytest.warns(UserWarning, match="CUDA no utilizable.*se usará CPU"):
        assert resolve_device("cuda") == torch.device("cpu")


def test_usable_cuda_executes_kernel_and_synchronizes(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    tensor = torch.zeros(1)
    calls = []
    monkeypatch.setattr(torch, "ones", lambda *args, **kwargs: calls.append(kwargs["device"]) or tensor)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("synchronized"))
    assert resolve_device("cuda") == torch.device("cuda")
    assert calls == ["cuda", "synchronized"]
    assert tensor.item() == 1.


def test_baseline_completes_on_cpu_and_records_effective_device_after_cuda_failure(prepared, tmp_path, monkeypatch):
    path, _ = prepared
    original_ones = torch.ones

    def ones(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            raise RuntimeError("no kernel image is available for execution on the device")
        return original_ones(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "ones", ones)
    # AdamW also queries graph capture when availability is mocked as True.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"), device="cuda",
                           num_server_rounds=1, mlflow_tracking_uri="disabled")
    with pytest.warns(UserWarning, match="CUDA no utilizable.*se usará CPU"):
        output = run_baseline(settings, "centralized")
    run = json.loads((output / "run.json").read_text())
    assert run["settings"]["device"] == "cpu"
    assert (output / "test_metrics.json").is_file()


@pytest.mark.parametrize("mode", ["local", "federated", "centralized", "zero-shot"])
@pytest.mark.parametrize("coverage", [[True, False], [False, False], [True, True]])
def test_effective_panel_requires_defined_selection_in_every_mode(mode, coverage):
    manifest = {"schema_version": 4, "clients": [{"validation_mase_defined": value} for value in coverage]}
    if not all(coverage):
        with pytest.raises(ValueError, match="descartar estas estaciones"):
            validate_selection_coverage(manifest, RunSettings(), mode)
    else:
        validate_selection_coverage(manifest, RunSettings(), mode)
    validate_selection_coverage(manifest, RunSettings(selection_metric="mae"), mode)


@pytest.mark.parametrize("model,device,batch,expected", [
    ("patchtst-fm-r2", "cuda", 128, 8), ("patchtst-fm-r2", "cuda:1", 3, 3),
    ("patchtst-fm-r2", "cpu", 128, 128), ("ttm-r3", "cuda", 128, 128),
    ("mlp", "cpu", 128, 128), ("flowstate-r1", "cuda", 128, 128),
])
def test_microbatch_limit_only_applies_to_patchtst_cuda(model, device, batch, expected):
    from fd_airquality.task import _microbatch_size

    settings = RunSettings(model=model, batch_size=batch)
    assert _microbatch_size(settings, torch.device(device)) == expected
    assert settings.batch_size == batch


@pytest.mark.parametrize("proximal_mu", [0.0, 0.7])
@pytest.mark.parametrize("freeze_a", [False, True])
def test_microbatch_accumulation_matches_full_batch_with_uneven_tails(
    monkeypatch, proximal_mu, freeze_a,
):
    from torch import nn
    from torch.utils.data import TensorDataset

    from fd_airquality.adapters import LoRAFactors, LoRALinear
    from fd_airquality.task import train_local

    settings = RunSettings(model="patchtst-fm-r2", batch_size=11, local_epochs=2,
                           learning_rate=0.01, weight_decay=0.03)
    configure_torch(settings, 17)
    original = LoRALinear(nn.Linear(4, 2), LoRAFactors(
        4, 2, rank=2, alpha=4, freeze_a=freeze_a, seed=19,
    )).double()
    dataset = TensorDataset(torch.randn(29, 4, dtype=torch.float64) * 10,
                            torch.randn(29, 2, dtype=torch.float64))
    full, accumulated = deepcopy(original), deepcopy(original)
    steps, norms = [], []
    original_step = torch.optim.AdamW.step
    original_clip = nn.utils.clip_grad_norm_

    def step(optimizer, *args, **kwargs):
        steps.append(optimizer)
        return original_step(optimizer, *args, **kwargs)

    def clip(*args, **kwargs):
        norm = original_clip(*args, **kwargs)
        norms.append(float(norm))
        return norm

    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    monkeypatch.setattr(nn.utils, "clip_grad_norm_", clip)
    monkeypatch.setattr("fd_airquality.task._microbatch_size", lambda *_: 11)
    reference = train_local(full, dataset, settings, torch.device("cpu"), 23, proximal_mu)
    monkeypatch.setattr("fd_airquality.task._microbatch_size", lambda *_: 4)
    actual = train_local(accumulated, dataset, settings, torch.device("cpu"), 23, proximal_mu)

    # 29 windows -> logical batches 11, 11, 7. Physical batches include tails
    # of 3; weighting each microbatch equally would fail this comparison.
    assert reference["optimizer-steps"] == actual["optimizer-steps"] == 6
    assert len(steps) == len(norms) == 12
    assert max(norms) > 1  # Clipping really occurs, only after accumulation.
    assert actual["effective-batch-size"] == 11
    assert actual["microbatch-size"] == 4
    assert actual["train-mse-normalized"] == pytest.approx(reference["train-mse-normalized"], rel=1e-10)
    for key, value in full.state_dict().items():
        torch.testing.assert_close(accumulated.state_dict()[key], value, rtol=1e-9, atol=1e-11)
    assert accumulated.base.weight.grad is None
    assert accumulated.base.bias.grad is None
    if freeze_a:
        assert accumulated.factors.A.grad is None


def test_evaluation_microbatches_preserve_window_order_and_metrics(prepared, monkeypatch):
    from torch import nn

    from fd_airquality.data import WindowDataset, load_client_data
    from fd_airquality.task import evaluate_local

    path, manifest = prepared
    geometry = manifest["geometry"]
    _, artifact = load_client_data(str(path), 0)
    dataset = WindowDataset(artifact, "validation", geometry["context"], geometry["horizon"])
    model = nn.Linear(geometry["context"], geometry["horizon"])
    calls = []
    model.register_forward_pre_hook(lambda _, args: calls.append(args[0].clone()))
    settings = RunSettings(model="patchtst-fm-r2", batch_size=128)
    monkeypatch.setattr("fd_airquality.task._microbatch_size", lambda *_: 128)
    reference = evaluate_local(model, dataset, settings, torch.device("cpu"), artifact, geometry["seasonality"])
    expected_order = torch.cat(calls)
    calls.clear()
    monkeypatch.setattr("fd_airquality.task._microbatch_size", lambda *_: 2)
    actual = evaluate_local(model, dataset, settings, torch.device("cpu"), artifact, geometry["seasonality"])
    assert max(len(inputs) for inputs in calls) <= 2
    torch.testing.assert_close(torch.cat(calls), expected_order, rtol=0, atol=0)
    assert actual == pytest.approx(reference, rel=1e-6, abs=1e-7)
