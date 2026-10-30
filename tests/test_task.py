import json

import pytest
import torch

from fd_airquality.baselines import run_baseline
from fd_airquality.settings import RunSettings
from fd_airquality.task import configure_torch, resolve_device, validate_selection_coverage


def test_cpu_does_not_probe_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CPU no necesita CUDA"))
    assert resolve_device("cpu") == torch.device("cpu")


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
