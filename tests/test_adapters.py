import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from fd_airquality.adapters import LoRAFactors, LoRALinear, layer_seed, target_linears
from fd_airquality.federated_adapters import aggregate_adapter_updates
from fd_airquality.models import (
    checkpoint_payload,
    load_shared_state,
    make_model,
    restore_checkpoint,
    shared_state,
)
from fd_airquality.settings import RunSettings


def factors(freeze_a=False):
    return LoRAFactors(8, 6, rank=2, alpha=4., dropout=0., freeze_a=freeze_a, seed=17)


@pytest.mark.parametrize("freeze_a", [False, True])
def test_internal_lora_has_correct_formula_and_only_adapter_gradients(freeze_a):
    base = nn.Linear(8, 6)
    adapter = factors(freeze_a)
    layer = LoRALinear(base, adapter)
    inputs = torch.randn(3, 8)
    torch.testing.assert_close(layer(inputs), base(inputs))
    with torch.no_grad():
        adapter.B.fill_(0.1)
    expected = base(inputs) + inputs @ adapter.A.T @ adapter.B.T * 2
    torch.testing.assert_close(layer(inputs), expected)
    layer(inputs).square().mean().backward()
    assert base.weight.grad is None and base.bias.grad is None
    assert adapter.B.grad is not None
    assert (adapter.A.grad is None) == freeze_a


def test_frozen_a_is_seeded_independently_of_backbone_and_client_rng():
    first = factors(True)
    torch.manual_seed(999)
    torch.randn(1000)
    second = factors(True)
    torch.testing.assert_close(first.A, second.A, rtol=0, atol=0)
    assert first.A.requires_grad is False


def test_patchtst_targets_include_both_real_conformer_feedforward_branches():
    backbone = nn.Module()
    backbone.backbone = nn.Module()
    block = nn.Module()
    block.attn = nn.Module()
    block.attn.qkv = nn.Linear(8, 24)
    block.attn.proj = nn.Linear(8, 8)
    for branch in ("ffn1", "ffn2"):
        module = nn.Module()
        module.net = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8))
        setattr(block, branch, module)
    backbone.backbone.blocks = nn.ModuleList([block])
    targets = target_linears(backbone, "patchtst-fm-r2", "auto")
    assert len(targets) == 6
    assert any(".ffn1." in path for path in targets)
    assert any(".ffn2." in path for path in targets)


def updates():
    one = {"adapter.proj.lora_A": torch.tensor([[1., 0., 2.], [0., 1., 1.]]),
           "adapter.proj.lora_B": torch.tensor([[1., 0.], [0., 1.], [2., 1.], [1., 2.]])}
    two = {"adapter.proj.lora_A": torch.tensor([[0., 2., 1.], [1., 0., 2.]]),
           "adapter.proj.lora_B": torch.tensor([[0., 2.], [1., 0.], [0., 1.], [2., 1.]])}
    return [one, two]


def settings(algorithm, **kwargs):
    return RunSettings(model="flowstate-r1", adaptation=algorithm, lora_rank=2, lora_alpha=4., **kwargs)


def test_vanilla_lora_weights_factors_by_window_count():
    local = updates()
    state, _ = aggregate_adapter_updates(local, [1, 3], settings("lora"), previous=local[0], server_round=1)
    for key in local[0]:
        torch.testing.assert_close(state[key], (local[0][key] + 3 * local[1][key]) / 4)


def test_ffa_only_aggregates_b_and_rejects_trainable_a_payload():
    local = [{k: v for k, v in x.items() if k.endswith("lora_B")} for x in updates()]
    state, _ = aggregate_adapter_updates(local, [1, 3], settings("ffa-lora"), previous=local[0], server_round=1)
    assert set(state) == {"adapter.proj.lora_B"}
    torch.testing.assert_close(state["adapter.proj.lora_B"], (local[0]["adapter.proj.lora_B"] + 3 * local[1]["adapter.proj.lora_B"]) / 4)
    with pytest.raises(ValueError, match="claves|estado|FFA"):
        aggregate_adapter_updates(updates(), [1, 3], settings("ffa-lora"), previous=local[0], server_round=1)


def test_flora_stacks_exact_products_once_and_preserves_history():
    local = updates()
    config = settings("flora")
    state, _ = aggregate_adapter_updates(local, [1, 3], config, previous=local[0], server_round=1)
    a = state["history.000001.proj.lora_A"]
    b = state["history.000001.proj.lora_B"]
    expected = sum(w * x["adapter.proj.lora_B"] @ x["adapter.proj.lora_A"] for w, x in zip([.25, .75], local))
    torch.testing.assert_close(b @ a, expected)
    assert a.shape == (4, 3) and b.shape == (4, 4)
    assert torch.count_nonzero(state["adapter.proj.lora_B"]) == 0
    again, _ = aggregate_adapter_updates(local, [1, 3], config, previous=state, server_round=2)
    torch.testing.assert_close(again["history.000001.proj.lora_A"], a, rtol=0, atol=0)
    assert "history.000002.proj.lora_A" in again
    assert not torch.equal(again["adapter.proj.lora_A"], state["adapter.proj.lora_A"])
    again["history.000001.proj.lora_A"].zero_()
    assert torch.count_nonzero(state["history.000001.proj.lora_A"]) > 0


def test_lora_fair_refines_b_without_accessing_data():
    local = updates()
    config = settings("lora-fair", fair_iterations=100, fair_learning_rate=.01, fair_lambda=.001)
    state, metrics = aggregate_adapter_updates(local, [1, 3], config, previous=local[0], server_round=1)
    torch.testing.assert_close(state["adapter.proj.lora_A"], (local[0]["adapter.proj.lora_A"] + 3 * local[1]["adapter.proj.lora_A"]) / 4)
    assert metrics["refinement-objective-final"] <= metrics["refinement-objective-initial"]
    assert torch.isfinite(state["adapter.proj.lora_B"]).all()
    assert not torch.equal(state["adapter.proj.lora_B"], (local[0]["adapter.proj.lora_B"] + 3 * local[1]["adapter.proj.lora_B"]) / 4)


def test_fair_does_not_create_an_update_for_zero_client_updates():
    local = updates()
    for x in local:
        x["adapter.proj.lora_B"].zero_()
    state, _ = aggregate_adapter_updates(local, [1, 3], settings("lora-fair", fair_iterations=5), previous=local[0], server_round=1)
    assert torch.count_nonzero(state["adapter.proj.lora_B"]) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fair_preserves_cpu_objective_and_sgd_with_fractional_weights(dtype):
    local = [{key: value.to(dtype) for key, value in state.items()} for state in updates()]
    config = settings("lora-fair", fair_iterations=7)
    akey, bkey = "adapter.proj.lora_A", "adapter.proj.lora_B"
    counts = [2.5, 3.5]
    weights = [count / sum(counts) for count in counts]
    a = sum(weight * state[akey] for weight, state in zip(weights, local, strict=True))
    b = sum(weight * state[bkey] for weight, state in zip(weights, local, strict=True))
    target = sum(weight * state[bkey] @ state[akey]
                 for weight, state in zip(weights, local, strict=True))
    # Independent reference, using the CPU implementation before the device change.
    generator = torch.Generator(device="cpu").manual_seed(layer_seed(config.seed + 1, akey))
    delta = torch.empty_like(b)
    torch.nn.init.xavier_uniform_(delta, generator=generator)
    delta.requires_grad_(True)
    optimizer = torch.optim.SGD([delta], lr=config.fair_learning_rate)

    def objective():
        return (1 - F.cosine_similarity(target, (b + delta) @ a, dim=0, eps=1e-8).mean()
                + config.fair_lambda * delta.square().sum())

    initial = float(objective().detach())
    for _ in range(config.fair_iterations):
        optimizer.zero_grad(set_to_none=True)
        objective().backward()
        optimizer.step()
    final = float(objective().detach())

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        state, metrics = aggregate_adapter_updates(local, counts, config, previous=local[0], server_round=1)
    torch.testing.assert_close(state[akey], a, rtol=0, atol=0)
    torch.testing.assert_close(state[bkey], (b + delta).detach(), rtol=0, atol=0)
    assert metrics["refinement-objective-initial"] == initial
    assert metrics["refinement-objective-final"] == final
    assert metrics["server-aggregation-cuda"] == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("zero", [False, True])
def test_fair_refines_on_resolved_device_and_returns_detached_cpu_state(monkeypatch, caplog, device, zero):
    local = updates()
    if zero:
        for state in local:
            state["adapter.proj.lora_B"].zero_()
    pristine = copy.deepcopy(local)
    moves, cache_calls, steps = [], [], []
    original_to, original_step = torch.Tensor.to, torch.optim.SGD.step

    def to(tensor, *args, **kwargs):
        if args and isinstance(args[0], torch.device) and args[0].type == "cuda":
            moves.append(tuple(tensor.shape))
            args = (torch.device("cpu"), *args[1:])
        return original_to(tensor, *args, **kwargs)

    def step(optimizer, *args, **kwargs):
        steps.append(True)
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to)
    monkeypatch.setattr(torch.optim.SGD, "step", step)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cache_calls.append(True))
    config = settings("lora-fair", device=device, fair_iterations=7)
    with caplog.at_level("INFO", logger="fd_airquality"):
        state, metrics = aggregate_adapter_updates(local, [2.5, 3.5], config, previous=local[0], server_round=2)

    assert all(value.device.type == "cpu" and not value.requires_grad for value in state.values())
    assert metrics["server-aggregation-cuda"] == int(device == "cuda")
    assert len(steps) == (0 if zero else config.fair_iterations)
    assert cache_calls == ([True] if device == "cuda" else [])
    # Mean A/B and two clients' factors move before forming dense products.
    assert moves == ([(2, 3), (4, 2), (4, 2), (2, 3), (4, 2), (2, 3)] if device == "cuda" else [])
    for original, untouched in zip(pristine, local, strict=True):
        torch.testing.assert_close(original, untouched, rtol=0, atol=0)
    assert "stage=aggregation" in caplog.text and f"algorithm=lora-fair device={device}" in caplog.text
    assert "layers=0/1" in caplog.text and "layers=1/1" in caplog.text


def test_fair_releases_cuda_cache_on_failure_without_cpu_retry(monkeypatch):
    calls = []

    def fail(*args):
        calls.append(args[-1].type)
        raise RuntimeError("simulated CUDA failure")

    monkeypatch.setattr("fd_airquality.federated_adapters._fair_layer", fail)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("cleanup"))
    local = updates()
    with pytest.raises(RuntimeError, match="simulated CUDA failure"):
        aggregate_adapter_updates(local, [1, 3], settings("lora-fair", device="cuda"),
                                  previous=local[0], server_round=1)
    assert calls == ["cuda", "cleanup"]


@pytest.mark.parametrize("algorithm", ["lora", "ffa-lora", "flora"])
def test_other_aggregations_stay_on_cpu_with_cuda_training(monkeypatch, algorithm):
    local = updates()
    if algorithm == "ffa-lora":
        local = [{key: value for key, value in state.items() if key.endswith("lora_B")} for state in local]
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: pytest.fail("Only FAIR uses CUDA aggregation"))
    state, metrics = aggregate_adapter_updates(local, [1, 3], settings(algorithm, device="cuda"),
                                              previous=local[0], server_round=1)
    assert all(value.device.type == "cpu" for value in state.values())
    assert "server-aggregation-cuda" not in metrics


@pytest.mark.parametrize("kind", ["count", "nonfinite", "shape", "keys"])
def test_adapter_aggregation_rejects_invalid_updates(kind):
    local = updates()
    counts = [1, 3]
    if kind == "count":
        counts = [1, 0]
    elif kind == "nonfinite":
        local[1]["adapter.proj.lora_A"][0, 0] = float("nan")
    elif kind == "shape":
        local[1]["adapter.proj.lora_B"] = torch.ones(5, 2)
    else:
        local[1]["untrusted"] = torch.ones(1)
    with pytest.raises(ValueError):
        aggregate_adapter_updates(local, counts, settings("lora"), previous=local[0], server_round=1)


@pytest.mark.parametrize("algorithm", ["lora", "ffa-lora", "flora", "lora-fair"])
def test_models_restore_internal_adapters_and_never_mutate_cached_backbone(algorithm, mock_granite):
    template = mock_granite
    pristine = copy.deepcopy(template.state_dict())
    config = settings(algorithm, lora_targets="proj", fair_iterations=5)
    geometry = {"context": 90, "horizon": 12}
    server = make_model(geometry, config, pretrained=False)
    client = make_model(geometry, config)
    load_shared_state(client, shared_state(server))
    inputs = torch.randn(2, 90)
    initial = client(inputs).detach()
    optimizer = torch.optim.SGD([p for p in client.parameters() if p.requires_grad], lr=.01)
    for _ in range(2):
        optimizer.zero_grad()
        client(inputs).square().mean().backward()
        optimizer.step()
    update = {k: v for k, v in shared_state(client).items() if k.startswith("adapter.")}
    state, _ = aggregate_adapter_updates([update, update], [1, 3], config, previous=shared_state(server), server_round=1)
    load_shared_state(client, state)
    payload = checkpoint_payload(geometry, config, state, 1)
    restored = restore_checkpoint(payload)
    torch.testing.assert_close(restored(inputs), client(inputs))
    assert not torch.equal(initial, client(inputs))
    if algorithm == "ffa-lora":
        assert payload["fixed_adapter_state"]
        assert not any(k.endswith("lora_A") for k in state)
    zero = make_model(geometry, RunSettings(model="flowstate-r1", adaptation="none"))
    torch.testing.assert_close(zero(inputs), initial)
    for key, value in pristine.items():
        torch.testing.assert_close(template.state_dict()[key], value, rtol=0, atol=0)


def test_unknown_or_empty_target_selection_fails_without_silent_fallback(mock_granite):
    with pytest.raises(ValueError, match="capas|target"):
        make_model({"context": 90, "horizon": 12}, settings("lora", lora_targets="absent"))


def test_clients_share_only_immutable_storage_not_parameter_or_module_objects(mock_granite):
    first = make_model({"context": 90, "horizon": 12}, settings("lora", lora_targets="proj"))
    second = make_model({"context": 90, "horizon": 12}, settings("lora", lora_targets="proj"))
    assert first.backbone is not second.backbone
    assert first.backbone.proj.base.weight is not second.backbone.proj.base.weight
    assert first.backbone.proj.base.weight.data_ptr() == second.backbone.proj.base.weight.data_ptr()
    assert first.factors["0"].A.data_ptr() != second.factors["0"].A.data_ptr()
    first.to(dtype=torch.float64)
    assert second.backbone.proj.base.weight.dtype == mock_granite.proj.weight.dtype == torch.float32
