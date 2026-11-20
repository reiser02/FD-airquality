from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fd_airquality.patchtst import install_overlap_add


def _reconstruction(width):
    from tsfm_public.models.patchtst_fm.modeling_patchtst_fm import PatchTSTFMModel

    class Reconstruction(nn.Module):
        _overlap_add = PatchTSTFMModel._overlap_add

    module = Reconstruction()
    module.config = SimpleNamespace(d_patch=width)
    window = torch.linspace(0, 1, width).reshape(1, 1, width)
    module.register_buffer("patch_loss_window", window)
    return module


@pytest.mark.parametrize("patches", [1, 3, 1023])
def test_vectorized_overlap_matches_upstream_forward_and_backward(patches, monkeypatch):
    width, stride = 16, 8
    module = _reconstruction(width)
    upstream = module._overlap_add
    rng = torch.Generator().manual_seed(71)
    prediction = torch.randn(2, 3, patches * width, generator=rng).transpose(1, 2).requires_grad_()
    length = (patches + 1) * stride
    expected = upstream(prediction, original_length=length, stride=stride)
    weights = torch.randn(expected.shape, generator=rng)
    expected_grad, = torch.autograd.grad(expected, prediction, weights)

    def fallback(*_args, **_kwargs):
        pytest.fail("FP32 with the supported geometry must not use the upstream loop")

    monkeypatch.setattr(type(module), "_overlap_add", fallback)
    install_overlap_add(module)
    clone = deepcopy(module)
    assert clone._overlap_add.__self__ is clone
    result = clone._overlap_add(prediction, original_length=length, stride=stride)
    grad, = torch.autograd.grad(result, prediction, weights)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    torch.testing.assert_close(grad, expected_grad, atol=0, rtol=0)


@pytest.mark.parametrize(("width", "length", "stride", "dtype"), [
    (16, 40, 8, torch.bfloat16),
    (16, 41, 8, torch.float32),
    (12, 40, 4, torch.float32),
])
def test_patchtst_retains_upstream_for_other_geometries_and_precisions(width, length, stride, dtype, monkeypatch):
    module = _reconstruction(width)
    upstream = type(module)._overlap_add
    patches = (length - width) // stride + 1
    prediction = torch.randn(2, patches * width, 3, dtype=dtype)
    expected = upstream(module, prediction, length, stride)
    calls = []

    def record(self, values, original_length, stride):
        calls.append((original_length, stride))
        return upstream(self, values, original_length, stride)

    monkeypatch.setattr(type(module), "_overlap_add", record)
    install_overlap_add(module)
    result = module._overlap_add(prediction, original_length=length, stride=stride)
    assert calls == [(length, stride)]
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
