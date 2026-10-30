"""Internal, real-valued low-rank updates independent of any LM-specific wrapper."""

from __future__ import annotations

from hashlib import sha256

import torch
from torch import nn
from torch.nn import functional as F

ADAPTER_IMPLEMENTATION = "internal_lora_v1"


def layer_seed(seed: int, name: str) -> int:
    """Stable initialization independent of backbone loading and client RNG state."""
    return (seed + int.from_bytes(sha256(name.encode()).digest()[:8], "little")) % (2**63 - 1)


class LoRAFactors(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, rank: int, alpha: float,
                 dropout: float = 0.0, freeze_a: bool = False, seed: int = 0):
        super().__init__()
        if not 1 <= rank <= min(in_features, out_features):
            raise ValueError(f"El rango {rank} excede las dimensiones ({out_features}, {in_features}); no se recorta.")
        self.scale, self.dropout = alpha / rank, dropout
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.A = nn.Parameter(torch.randn(rank, in_features, generator=generator) * 0.02, requires_grad=not freeze_a)
        self.B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = F.dropout(inputs, p=self.dropout, training=self.training)
        return F.linear(F.linear(inputs, self.A), self.B) * self.scale


class LoRALinear(nn.Module):
    """Frozen original linear plus a local LoRA and optional exact FLoRA history.

    Accumulated updates are separate buffers. Never mutate the shared pretrained
    Parameter: another client or zero-shot instance may share that storage.
    """

    def __init__(self, base: nn.Linear, factors: LoRAFactors):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.factors = factors
        self.register_buffer("merged_delta", None, persistent=False)
        self.in_features, self.out_features = base.in_features, base.out_features

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        prediction = self.base(inputs)
        if self.merged_delta is not None:
            prediction = prediction + F.linear(inputs, self.merged_delta)
        return prediction + self.factors(inputs)


def target_linears(backbone: nn.Module, name: str, targets: str) -> dict[str, nn.Linear]:
    explicit = [item.strip() for item in targets.split(",")] if targets != "auto" else None

    def selected(path):
        if explicit is not None:
            return any(path == item or path.endswith("." + item) for item in explicit)
        if name == "ttm-r3":
            return path.startswith(("trend_forecaster.", "residual_forecaster.")) and path.endswith(("mlp.fc1", "mlp.fc2"))
        if name == "patchtst-fm-r2":
            feedforward = path.startswith("backbone.blocks.") and bool({"mlp", "ffn1", "ffn2"}.intersection(path.split(".")))
            return path.endswith(("attn.qkv", "attn.proj")) or feedforward
        if name == "flowstate-r1":
            return path.endswith((".out", "decoder.lin", "mlp.w1", "mlp.w2", "mlp.w3", "output_gate"))
        return False

    result = {path: module for path, module in backbone.named_modules() if isinstance(module, nn.Linear) and selected(path)}
    if not result:
        raise ValueError(f"La selección de capas target={targets!r} no encontró lineales para {name}.")
    if explicit is not None:
        for item in explicit:
            if not any(path == item or path.endswith("." + item) for path in result):
                raise ValueError(f"La selección target no encontró {item!r}.")
    return dict(sorted(result.items()))
