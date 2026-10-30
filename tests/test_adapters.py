import copy

import pytest
import torch
from torch import nn

from fd_airquality.adapters import LoRAFactors, LoRALinear, target_linears
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
def test_models_restore_internal_adapters_and_never_mutate_cached_backbone(algorithm, monkeypatch):
    from test_models import DummyBackbone

    template = DummyBackbone()
    pristine = copy.deepcopy(template.state_dict())
    monkeypatch.setattr("fd_airquality.models._load_granite", lambda *_: template)
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


def test_unknown_or_empty_target_selection_fails_without_silent_fallback(monkeypatch):
    from test_models import DummyBackbone

    monkeypatch.setattr("fd_airquality.models._load_granite", lambda *_: DummyBackbone())
    with pytest.raises(ValueError, match="capas|target"):
        make_model({"context": 90, "horizon": 12}, settings("lora", lora_targets="absent"))


def test_clients_share_only_immutable_storage_not_parameter_or_module_objects(monkeypatch):
    from test_models import DummyBackbone

    template = DummyBackbone()
    monkeypatch.setattr("fd_airquality.models._load_granite", lambda *_: template)
    first = make_model({"context": 90, "horizon": 12}, settings("lora", lora_targets="proj"))
    second = make_model({"context": 90, "horizon": 12}, settings("lora", lora_targets="proj"))
    assert first.backbone is not second.backbone
    assert first.backbone.proj.base.weight is not second.backbone.proj.base.weight
    assert first.backbone.proj.base.weight.data_ptr() == second.backbone.proj.base.weight.data_ptr()
    assert first.factors["0"].A.data_ptr() != second.factors["0"].A.data_ptr()
    first.to(dtype=torch.float64)
    assert second.backbone.proj.base.weight.dtype == template.proj.weight.dtype == torch.float32
