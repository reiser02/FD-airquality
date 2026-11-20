"""Model-specific handling of incomplete histories; labels are never imputed."""

from __future__ import annotations

import json

import torch

POLICIES = {"strict", "ffill", "linear", "mean", "zero", "native"}
DEFAULT_POLICIES = {
    "mlp": "native", "autoencoder": "ffill", "autoencoder-tiny": "ffill",
    "nlinear": "native", "nhits": "native",
    "nlinear-point": "native", "linear-6lags": "ffill", "seasonal-naive": "native", "holt": "native",
    "ttm-r3": "native", "patchtst-fm-r2": "native", "flowstate-r1": "native",
}


def resolve_missing_policy(model: str, policy: str = "auto", overrides: str = "{}") -> str:
    mapping = json.loads(overrides)
    if not isinstance(mapping, dict) or set(mapping) - DEFAULT_POLICIES.keys():
        raise ValueError("missing-policies debe asociar nombres de modelos conocidos a políticas.")
    if any(not isinstance(value, str) or value not in POLICIES for value in mapping.values()):
        raise ValueError(f"Políticas de ausencias admitidas: {sorted(POLICIES)}")
    native_models = {name for name, default in DEFAULT_POLICIES.items() if default == "native"}
    if any(value == "native" and name not in native_models
           for name, value in mapping.items()):
        raise ValueError("La política native requiere un modelo con tratamiento nativo de ausencias.")
    resolved = mapping.get(model, DEFAULT_POLICIES[model]) if policy == "auto" else policy
    if not isinstance(resolved, str) or resolved not in POLICIES or (resolved == "native" and model not in native_models):
        raise ValueError(f"Política de ausencias {resolved!r} no soportada por {model}.")
    return resolved


def impute_context(inputs: torch.Tensor, policy: str) -> torch.Tensor:
    """Use only this forecast's history, never labels or another batch member.

    Forward fill initializes a missing prefix to the training mean (zero after
    normalization). Linear interpolation may use both sides of an interior gap:
    both sides are in the already available context, not after the origin.
    """
    if inputs.ndim != 2 or torch.isinf(inputs).any():
        raise ValueError("El contexto debe ser bidimensional y no contener infinitos.")
    observed = torch.isfinite(inputs)
    if observed.all():
        return inputs
    if policy == "strict":
        raise ValueError("La política strict no admite NaN en el contexto.")
    if policy == "native":
        return inputs
    if policy == "zero":
        return torch.nan_to_num(inputs, nan=0.0)
    if policy == "mean":
        mean = torch.where(observed, inputs, 0).sum(1, keepdim=True) / observed.sum(1, keepdim=True).clamp_min(1)
        return torch.where(observed, inputs, mean)
    positions = torch.arange(inputs.shape[1], device=inputs.device).expand_as(inputs)
    previous = torch.where(observed, positions, -1).cummax(1).values
    left = inputs.gather(1, previous.clamp_min(0)).nan_to_num()
    if policy == "ffill":
        return torch.where(previous >= 0, left, 0.0)
    if policy != "linear":
        raise ValueError(f"Imputador desconocido: {policy}")
    following = torch.where(observed, positions, inputs.shape[1]).flip(1).cummin(1).values.flip(1)
    right = inputs.gather(1, following.clamp_max(inputs.shape[1] - 1)).nan_to_num()
    fraction = (positions - previous) / (following - previous).clamp_min(1)
    interpolated = left + fraction * (right - left)
    # No backward fill of a missing prefix. Trailing gaps use the last observation.
    return torch.where(previous < 0, 0.0, torch.where(following == inputs.shape[1], left, interpolated))


def forecast(model, inputs: torch.Tensor, settings) -> torch.Tensor:
    policy = resolve_missing_policy(settings.model, settings.missing_policy, settings.missing_policies)
    if getattr(model, "handles_missing_context", False):
        return model(inputs, missing_policy=policy)
    values = impute_context(inputs, policy)
    if policy == "native" and torch.isnan(values).any():
        return model(values, missing_policy="native")
    return model(values)


def forecast_distribution(model, inputs: torch.Tensor, settings):
    policy = resolve_missing_policy(settings.model, settings.missing_policy, settings.missing_policies)
    if getattr(model, "handles_missing_context", False) and hasattr(model, "forecast_distribution"):
        return model.forecast_distribution(inputs, missing_policy=policy)
    if getattr(model, "handles_missing_context", False):
        return model(inputs, missing_policy=policy), None, None
    values = impute_context(inputs, policy)
    if hasattr(model, "forecast_distribution"):
        return model.forecast_distribution(values, missing_policy="native" if policy == "native" else "strict")
    return model(values), None, None
