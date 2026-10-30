"""FedIT, freeze-A, exact stacking FLoRA, and ICCV-2025 LoRA-FAIR aggregation.

References: arXiv:2403.12313, 2409.05976, 2411.14961v3; LoRA-FAIR
server.py at jmbian/LoRA-FAIR@6afb36c4d8ff9f7b5dce5fd4be7728d4f9467d5a.
FAIR follows the released column-wise cosine loss and squared L2 regularizer.
All clients use identical ranks/scaling in this benchmark. No server data needed.
"""

from __future__ import annotations

import time
from collections.abc import Mapping

import torch
from torch.nn import functional as F

from fd_airquality.adapters import layer_seed
from fd_airquality.settings import RunSettings


def tensor_bytes(state: Mapping[str, torch.Tensor]) -> int:
    """Tensor payload bytes, excluding Flower serialization and transport framing."""
    return sum(value.numel() * value.element_size() for value in state.values())


def _validate(updates, counts, previous, algorithm):
    if not updates or len(updates) != len(counts) or any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in counts):
        raise ValueError("Recuentos de entrenamiento inválidos.")
    expected = {key for key in previous if key.startswith("adapter.")}
    if not expected:
        raise ValueError("Estado LoRA vacío.")
    if algorithm == "ffa-lora" and any(key.endswith("lora_A") for key in expected):
        raise ValueError("FFA intercambia solo B, nunca A.")
    for state in updates:
        if set(state) != expected:
            raise ValueError("Claves del estado de adaptación incompatibles.")
        for key, value in state.items():
            if (not key.endswith((".lora_A", ".lora_B")) or value.ndim != 2 or not value.is_floating_point()
                    or value.dtype != previous[key].dtype or value.shape != previous[key].shape
                    or not torch.isfinite(value).all()):
                raise ValueError(f"Forma/tipo/valor de adaptación incompatible: {key}")
    if algorithm != "ffa-lora":
        for key in expected:
            counterpart = key.removesuffix("lora_B") + "lora_A" if key.endswith("lora_B") else key.removesuffix("lora_A") + "lora_B"
            if counterpart not in expected:
                raise ValueError("Pareja A/B incompleta.")
            if key.endswith("lora_A"):
                bkey = key.removesuffix("lora_A") + "lora_B"
                if bkey not in expected or previous[bkey].shape[1] != previous[key].shape[0]:
                    raise ValueError("Pareja A/B incompatible.")


def _fair_b(a, b, target, settings, seed):
    if not torch.count_nonzero(target):
        return b.clone(), 0.0, 0.0
    generator = torch.Generator(device="cpu").manual_seed(seed)
    delta = torch.empty_like(b, device="cpu")
    torch.nn.init.xavier_uniform_(delta, generator=generator)
    delta.requires_grad_(True)
    optimizer = torch.optim.SGD([delta], lr=settings.fair_learning_rate)

    def objective():
        reconstructed = (b + delta) @ a
        return 1 - F.cosine_similarity(target, reconstructed, dim=0, eps=1e-8).mean() + settings.fair_lambda * delta.square().sum()

    initial = float(objective().detach())
    for _ in range(settings.fair_iterations):
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        if not torch.isfinite(loss):
            raise ValueError("Refinamiento LoRA-FAIR no finito.")
        loss.backward()
        optimizer.step()
    final = float(objective().detach())
    if not torch.isfinite(delta).all():
        raise ValueError("Refinamiento LoRA-FAIR produjo pesos no finitos.")
    return (b + delta).detach(), initial, final


def aggregate_adapter_updates(updates: list[dict], counts: list[int], settings: RunSettings,
                              *, previous: dict, server_round: int) -> tuple[dict, dict]:
    """Aggregate current local factors; FLoRA history is server-owned and cumulative.

    History factors use the original local alpha/r even after stacking increases
    their width. Applying alpha/(sum ranks), or p to both A and B, is incorrect.
    """
    started = time.perf_counter()
    algorithm = "lora" if settings.adaptation == "auto" else settings.adaptation
    if algorithm not in {"lora", "ffa-lora", "flora", "lora-fair"}:
        raise ValueError("Algoritmo de agregación LoRA desconocido.")
    _validate(updates, counts, previous, algorithm)
    updates = [{key: value.detach().cpu() for key, value in state.items()} for state in updates]
    weights = [n / sum(counts) for n in counts]
    averaged = {key: sum(weight * state[key] for weight, state in zip(weights, updates, strict=True)) for key in sorted(updates[0])}
    diagnostics = {}
    if algorithm == "flora":
        previous_rounds = {int(key.split(".")[1]) for key in previous if key.startswith("history.")}
        if server_round < 1 or previous_rounds != set(range(1, server_round)):
            raise ValueError("Ronda FLoRA repetida o inválida; no se aplica dos veces una actualización.")
        history = {key: value.detach().cpu().clone() for key, value in previous.items() if key.startswith("history.")}
        for key in sorted(averaged):
            if not key.endswith("lora_A"):
                continue
            bkey = key.removesuffix("lora_A") + "lora_B"
            path = key.removeprefix("adapter.").removesuffix(".lora_A")
            history[f"history.{server_round:06d}.{path}.lora_A"] = torch.cat([weight * state[key] for weight, state in zip(weights, updates, strict=True)], dim=0)
            history[f"history.{server_round:06d}.{path}.lora_B"] = torch.cat([state[bkey] for state in updates], dim=1)
            generator = torch.Generator().manual_seed(layer_seed(settings.seed + 100_003 * server_round, path))
            averaged[key] = torch.randn(averaged[key].shape, generator=generator, dtype=averaged[key].dtype) * .02
            averaged[bkey] = torch.zeros_like(averaged[bkey])
        averaged.update(history)
    elif algorithm == "lora-fair":
        initial = final = 0.0
        for key in sorted(averaged):
            if not key.endswith("lora_A"):
                continue
            bkey = key.removesuffix("lora_A") + "lora_B"
            target = sum(weight * state[bkey] @ state[key] for weight, state in zip(weights, updates, strict=True))
            averaged[bkey], before, after = _fair_b(averaged[key], averaged[bkey], target, settings,
                                                  layer_seed(settings.seed + server_round, key))
            initial += before
            final += after
        diagnostics.update({"refinement-objective-initial": initial, "refinement-objective-final": final})
    diagnostics["server-aggregation-seconds"] = time.perf_counter() - started
    diagnostics["global-state-tensor-bytes"] = tensor_bytes(averaged)
    return averaged, diagnostics
