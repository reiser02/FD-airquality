import pytest
import torch
from torch import nn

from fd_airquality.models import (
    GRANITE,
    ForecastAutoencoder,
    load_shared_state,
    make_model,
    shared_state,
)
from fd_airquality.settings import RunSettings
from fd_airquality.task import configure_torch, train_local


@pytest.mark.parametrize("name", ["mlp", "autoencoder", "nlinear", "nhits"])
def test_trainable_models_have_forecast_shapes_and_gradients(name):
    settings = RunSettings(model=name, hidden_size=8, latent_size=3)
    configure_torch(settings, 17)
    model = make_model({"context": 6, "horizon": 2}, settings)
    inputs = torch.randn(4, 6)
    forecast = model(inputs)
    assert forecast.shape == (4, 2)
    forecast.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    clone = make_model({"context": 6, "horizon": 2}, settings)
    load_shared_state(clone, shared_state(model))
    torch.testing.assert_close(model(inputs), clone(inputs))
    if name != "autoencoder":
        assert type(model.module).__module__.startswith("neuralforecast.")


def test_autoencoder_reconstruction_uses_context_only(prepared):
    from fd_airquality.data import WindowDataset, load_client_data

    path, _ = prepared
    _, artifact = load_client_data(str(path), 0)
    settings = RunSettings(model="autoencoder", hidden_size=8, latent_size=3)
    device = configure_torch(settings, 19)
    model = make_model({"context": 6, "horizon": 2}, settings)
    assert isinstance(model, ForecastAutoencoder)
    before = model.reconstruction[-1].weight.detach().clone()
    train_local(model, WindowDataset(artifact, "train", 6, 2), settings, device, 19)
    assert not torch.equal(before, model.reconstruction[-1].weight)
    assert model.encoder[-1].out_features == 3


@pytest.mark.parametrize("name", list(GRANITE))
def test_granite_exchanges_only_adapter_and_never_trains_backbone(name, mock_granite):
    settings = RunSettings(model=name, hidden_size=4, lora_targets="proj")
    context = 90
    configure_torch(settings, 7)
    server = make_model({"context": context, "horizon": 12}, settings, pretrained=False)
    configure_torch(settings, 7)
    client = make_model({"context": context, "horizon": 12}, settings)
    for key in shared_state(server):
        torch.testing.assert_close(shared_state(server)[key], shared_state(client)[key], rtol=0, atol=0)
    load_shared_state(client, shared_state(server))
    client.train()
    assert not client.backbone.training
    inputs = torch.randn(2, context)
    predictions = client(inputs)
    torch.testing.assert_close(predictions, inputs[:, -1:].repeat(1, 12))
    predictions.square().mean().backward()
    assert client.backbone.frozen.grad is None
    assert client.factors["0"].B.grad is not None
    assert not any("backbone" in key or "frozen" in key for key in shared_state(client))
    if name == "flowstate-r1":
        assert client.backbone.calls[0]["prediction_type"] == "median"
        assert client.backbone.calls[0]["scale_factor"] == 1.0


def test_granite_does_not_change_initialization_when_loading_weights(monkeypatch, granite_template):
    def load(*_):
        torch.randn(100)
        return granite_template

    monkeypatch.setattr("fd_airquality.models._load_granite", load)
    settings = RunSettings(model="flowstate-r1", lora_targets="proj")
    configure_torch(settings, 7)
    server = make_model({"context": 72, "horizon": 12}, settings, pretrained=False)
    configure_torch(settings, 7)
    rng_state = torch.get_rng_state()
    client = make_model({"context": 72, "horizon": 12}, settings)
    assert torch.equal(torch.get_rng_state(), rng_state)
    for key in shared_state(server):
        torch.testing.assert_close(shared_state(server)[key], shared_state(client)[key], rtol=0, atol=0)


@pytest.mark.parametrize("model,context,horizon", [
    ("ttm-r3", 512, 12), ("ttm-r3", 90, 31),
    ("patchtst-fm-r2", 6, 2), ("flowstate-r1", 4096, 12),
])
def test_invalid_model_geometry_fails_before_download(model, context, horizon, monkeypatch):
    monkeypatch.setattr("fd_airquality.models._load_granite", lambda *_: pytest.fail("No debe descargar"))
    with pytest.raises(ValueError):
        make_model({"context": context, "horizon": horizon}, RunSettings(model=model))


def test_immutable_revision_and_finite_hyperparameters():
    for kwargs in ({"hf_revision": "main"}, {"model": "unknown"}, {"reconstruction_weight": float("nan")},
                   {"flowstate_scale_factor": 0}, {"mlflow_log_models": 2}):
        with pytest.raises(ValueError):
            RunSettings(**kwargs)


def test_flowstate_does_not_silently_shorten_scaled_context():
    from fd_airquality.models import validate_model_geometry

    with pytest.raises(ValueError, match="recortaría"):
        validate_model_geometry({"context": 512, "horizon": 12}, RunSettings(model="flowstate-r1", flowstate_scale_factor=10))
    with pytest.raises(ValueError, match="recortaría"):
        validate_model_geometry({"context": 2048, "horizon": 48}, RunSettings(model="flowstate-r1"))


def test_alternative_patchtst_revision_cannot_silently_crop_context(mock_granite):
    with pytest.raises(ValueError, match="recortaría"):
        make_model({"context": 1024, "horizon": 12}, RunSettings(model="patchtst-fm-r2"))


def test_ttm_uses_decomposed_r3_class_and_rejects_partial_checkpoint(monkeypatch, granite_template):
    import tsfm_public

    from fd_airquality.models import _load_granite

    called = []

    class Loader:
        @classmethod
        def from_pretrained(cls, identifier, **kwargs):
            called.append((identifier, kwargs))
            return granite_template, {"missing_keys": ["trend_forecaster.weight"]}

    monkeypatch.setattr(tsfm_public, "TinyTimeMixerForDecomposedPrediction", Loader)
    _load_granite.cache_clear()
    with pytest.raises(RuntimeError, match="pesos aleatorios"):
        _load_granite("ttm-r3", GRANITE["ttm-r3"][1])
    assert called[0][0].endswith("granite-timeseries-ttm-r3")
    assert called[0][1]["revision"] == GRANITE["ttm-r3"][1]
    assert called[0][1]["use_safetensors"] is True


@pytest.mark.parametrize("device,native_bf16,enabled", [
    ("cpu", False, None), ("mps", False, True),
    ("cuda:0", False, False), ("cuda:1", True, True),
])
def test_patchtst_autocast_requires_native_bf16_on_the_requested_gpu(
    device, native_bf16, enabled, monkeypatch,
):
    from contextlib import contextmanager, nullcontext

    from fd_airquality.models import _patchtst_autocast_context

    requested = torch.device(device)
    active_devices, checks, contexts = [], [], []

    @contextmanager
    def cuda_device(value):
        active_devices.append(value)
        yield
        active_devices.pop()

    def supports_bf16(*, including_emulation):
        assert active_devices == [requested]
        checks.append(including_emulation)
        return native_bf16

    def autocast(**kwargs):
        contexts.append(kwargs)
        return nullcontext()

    monkeypatch.setattr(torch.cuda, "device", cuda_device)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", supports_bf16)
    monkeypatch.setattr(torch, "autocast", autocast)
    with _patchtst_autocast_context(requested):
        pass
    assert not active_devices
    assert checks == ([False] if requested.type == "cuda" else [])
    assert contexts == ([] if enabled is None else [{
        "device_type": requested.type, "dtype": torch.bfloat16, "enabled": enabled,
    }])


def test_patchtst_fp32_fallback_overrides_outer_autocast_without_cutting_gradients(monkeypatch):
    from contextlib import nullcontext

    from fd_airquality.models import _patchtst_autocast_context

    real_autocast = torch.autocast
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **_: False)
    # Exercise real nesting and autograd on CPU without initializing a GPU.
    monkeypatch.setattr(torch, "autocast", lambda **kw: real_autocast(
        device_type="cpu", dtype=kw["dtype"], enabled=kw["enabled"],
    ))
    layer = nn.Linear(4, 2)
    inputs = torch.randn(3, 4, requires_grad=True)
    with real_autocast("cpu", dtype=torch.bfloat16), _patchtst_autocast_context(torch.device("cuda:0")):
        prediction = layer(inputs)
    assert prediction.dtype == torch.float32
    prediction.square().mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()


@pytest.mark.parametrize("pinned", [True, False])
def test_patchtst_loader_scopes_reconstruction_to_pinned_checkpoint(monkeypatch, granite_template, pinned):
    import tsfm_public
    from tsfm_public.models.patchtst_fm import modeling_patchtst_fm

    from fd_airquality.models import _load_granite, _patchtst_autocast_context
    from fd_airquality.patchtst import _overlap_add

    called = []
    granite_template.backbone = nn.Identity()
    revision = GRANITE["patchtst-fm-r2"][1] if pinned else "a" * 40

    class Loader:
        @classmethod
        def from_pretrained(cls, identifier, **kwargs):
            assert modeling_patchtst_fm.get_autocast_context is _patchtst_autocast_context
            called.append((identifier, kwargs))
            return granite_template, {}

    monkeypatch.setattr(tsfm_public, "PatchTSTFMForPrediction", Loader)
    # Restore the upstream module's helper after the loader assigns it.
    monkeypatch.setattr(modeling_patchtst_fm, "get_autocast_context", lambda _: None)
    _load_granite.cache_clear()
    try:
        model = _load_granite("patchtst-fm-r2", revision)
        assert not model.training
        assert not any(parameter.requires_grad for parameter in model.parameters())
        if pinned:
            assert model.backbone._overlap_add.__func__ is _overlap_add
        else:
            assert not hasattr(model.backbone, "_overlap_add")
    finally:
        _load_granite.cache_clear()
    assert called == [(GRANITE["patchtst-fm-r2"][0], {
        "revision": revision,
        "use_safetensors": True, "output_loading_info": True,
    })]
