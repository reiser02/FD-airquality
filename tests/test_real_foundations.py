import os

import pytest
import torch
from torch.utils.data import TensorDataset

from fd_airquality.federated_adapters import aggregate_adapter_updates
from fd_airquality.models import (
    GRANITE,
    checkpoint_payload,
    load_shared_state,
    make_model,
    restore_checkpoint,
    shared_state,
)
from fd_airquality.settings import RunSettings
from fd_airquality.task import configure_torch, train_local


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("FD_AIRQUALITY_REAL_MODELS") != "1", reason="Explicit opt-in: loads pinned real foundation checkpoints")
@pytest.mark.parametrize("name,algorithm", [(name, algorithm) for name in GRANITE for algorithm in ("lora", "ffa-lora", "flora", "lora-fair")])
def test_real_checkpoint_gradients_aggregation_and_restore(name, algorithm, tmp_path):
    config = RunSettings(model=name, adaptation=algorithm, lora_rank=4, lora_alpha=8.,
                         batch_size=1, local_epochs=1, fair_iterations=5, mlflow_tracking_uri="disabled")
    device = configure_torch(config, 42)
    geometry = {"context": 90, "horizon": 12}
    model = make_model(geometry, config)
    if name == "patchtst-fm-r2":
        assert any(".ffn1." in path for path in model.layer_paths)
        assert any(".ffn2." in path for path in model.layer_paths)
    original = {k: v.detach().clone() for k, v in shared_state(model).items()}
    probe = torch.sin(torch.arange(90, dtype=torch.float32) / 5).unsqueeze(0)
    target = torch.cos(torch.arange(12, dtype=torch.float32) / 5).unsqueeze(0)
    with torch.no_grad():
        initial = model(probe).clone()
    assert torch.isfinite(initial).all()
    metrics = train_local(model, TensorDataset(probe.repeat(2, 1), target.repeat(2, 1)), config, device, 42)
    assert all(factor.B.grad is not None and torch.isfinite(factor.B.grad).all() for factor in model.factors.values())
    update = {k: v.detach().cpu().clone() for k, v in shared_state(model).items() if k.startswith("adapter.")}
    assert any(not torch.equal(v, original[k]) for k, v in update.items())
    assert all(p.grad is None for key, p in model.backbone.named_parameters() if "factors" not in key)
    state, _ = aggregate_adapter_updates([update, update], [1, 2], config, previous=original, server_round=1)
    load_shared_state(model, state)
    model.eval()
    checkpoint = tmp_path / "model.pt"
    torch.save(checkpoint_payload(geometry, config, state, 1), checkpoint)
    restored = restore_checkpoint(checkpoint).eval()
    with torch.no_grad():
        forecast = model(probe)
        torch.testing.assert_close(restored(probe), forecast, rtol=1e-5, atol=1e-5)
    assert forecast.shape == (1, 12) and torch.isfinite(forecast).all()
    assert not torch.equal(initial, forecast)
    assert metrics["trainable-parameters"] > 0 and metrics["train-seconds"] > 0


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("FD_AIRQUALITY_REAL_MODELS") != "1", reason="Explicit opt-in: real native missing-data interfaces")
@pytest.mark.parametrize("name", list(GRANITE))
@pytest.mark.parametrize("padded", [False, True])
def test_real_native_missing_histories_and_masked_targets(name, padded):
    from fd_airquality.missing import forecast_distribution

    settings = RunSettings(model=name, batch_size=4 if padded else 2, local_epochs=1, probabilistic=1,
                            mlflow_tracking_uri="disabled")
    configure_torch(settings, 42)
    model = make_model({"context": 90, "horizon": 12}, settings)
    inputs = torch.sin(torch.arange(90, dtype=torch.float32) / 7).repeat(2, 1)
    inputs[0, [0, 10, 11, 89]] = float("nan")
    inputs[1, [23, 24]] = float("nan")
    actual = torch.cos(torch.arange(12, dtype=torch.float32) / 5).repeat(2, 1)
    actual[0, [1, 4]] = float("nan")
    if padded:
        inputs[0, :89] = float("nan")
        inputs[0, 89] = .25
        inputs[1, :7] = float("nan")
        actual[:, 1:] = float("nan")
    metrics = train_local(model, TensorDataset(inputs, actual), settings, torch.device("cpu"), 42)
    assert metrics["num-targets"] == (2 if padded else 22)
    assert metrics["replacement-window-occurrences"] == (2 if padded else 0)
    model.eval()
    with torch.no_grad():
        prediction, quantiles, levels = forecast_distribution(model, inputs, settings)
    assert prediction.shape == (2, 12) and torch.isfinite(prediction).all()
    if name != "ttm-r3":
        assert quantiles.shape == (2, 12, len(levels))
        assert torch.isfinite(quantiles).all()
