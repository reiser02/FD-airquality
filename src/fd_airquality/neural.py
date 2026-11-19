"""Thin NeuralForecast/Flower bridge; architectures, losses and scalers are upstream.

Window origins and the optimizer belong to our external temporal protocol. We
call the installed package's window normalization, parsing, forward and loss,
not NeuralForecast.fit (which would own both window sampling and the optimizer).
"""

from __future__ import annotations

import pandas as pd
import torch
from torch import nn

from fd_airquality.availability import availability_threshold
from fd_airquality.missing import impute_context

NEURALFORECAST_REVISION = "69e431413b22cbb1a74d35db2fa7994092ce0ea4"
NEURAL_MODELS = ("mlp", "nlinear", "nhits")
QUANTILE_LEVELS = tuple(index / 20 for index in range(1, 20))


class NeuralForecaster(nn.Module):
    handles_missing_context = True

    def __init__(self, context: int, horizon: int, settings):
        super().__init__()
        from neuralforecast.losses.pytorch import MSE, MQLoss
        from neuralforecast.models import MLP, NHITS, NLinear

        self.context, self.horizon = context, horizon
        self.quantile_levels = QUANTILE_LEVELS if settings.probabilistic else None
        loss = MQLoss(quantiles=list(QUANTILE_LEVELS)) if self.quantile_levels else MSE()
        options = {
            "h": horizon, "input_size": context, "loss": loss,
            "scaler_type": settings.scaler_type if settings.normalization == "window" else "identity",
            "training_data_availability_threshold": list(availability_threshold(settings)),
            "random_seed": settings.seed, "start_padding_enabled": False,
            "windows_batch_size": None, "enable_checkpointing": False,
            "logger": False, "enable_progress_bar": False,
        }
        if settings.model == "mlp":
            self.module = MLP(**options, hidden_size=settings.hidden_size)
        elif settings.model == "nlinear":
            self.module = NLinear(**options)
        elif settings.model == "nhits":
            self.module = NHITS(**options, mlp_units=3 * [[settings.hidden_size, settings.hidden_size]])
        else:
            raise ValueError(f"Modelo NeuralForecast desconocido: {settings.model}")

    def _prepare(self, inputs, actual=None, *, missing_policy="native"):
        if inputs.ndim != 2 or inputs.shape[1] != self.context or torch.isinf(inputs).any():
            raise ValueError("Contexto NeuralForecast incompatible o infinito.")
        if not torch.isfinite(inputs).any(dim=1).all():
            raise ValueError("Un contexto no tiene ninguna observación para normalizar.")
        if missing_policy == "strict" and not torch.isfinite(inputs).all():
            raise ValueError("La política strict no admite NaN en el contexto.")
        if actual is None:
            actual = inputs.new_full((len(inputs), self.horizon), float("nan"))
        if actual.shape != (len(inputs), self.horizon) or torch.isinf(actual).any():
            raise ValueError("Targets NeuralForecast incompatibles o infinitos.")
        values = torch.cat([inputs, actual], dim=1)
        observed = torch.isfinite(values)
        # Preserve the original mask before the finite placeholder. The package
        # masks the entire horizon when estimating TemporalNorm statistics.
        temporal = torch.stack([values.nan_to_num(), observed.to(values.dtype)], dim=2).unsqueeze(-1)
        columns = pd.Index(["y", "available_mask"])
        windows = {"temporal": temporal, "temporal_cols": columns, "static": None, "static_cols": None}
        windows = self.module._normalization(windows, y_idx=0)
        parsed = self.module._parse_windows({"y_idx": 0, "temporal_cols": columns}, windows)
        insample, insample_mask, target, target_mask, hist, futr, stat = parsed
        if missing_policy != "native":
            incomplete = insample[..., 0].masked_fill(~insample_mask[..., 0].bool(), float("nan"))
            insample = impute_context(incomplete, missing_policy).unsqueeze(-1)
        batch = {"insample_y": insample, "insample_mask": insample_mask,
                 "futr_exog": futr, "hist_exog": hist, "stat_exog": stat}
        return batch, target, target_mask

    def _predict_normalized(self, batch):
        output = self.module.loss.domain_map(self.module(batch))
        expected = ((len(batch["insample_y"]), self.horizon, 1, len(self.quantile_levels))
                    if self.quantile_levels else (len(batch["insample_y"]), self.horizon, 1))
        if output.shape != expected or not torch.isfinite(output).all():
            raise ValueError("Forma de predicción NeuralForecast incompatible o valores no finitos.")
        return output

    def training_objective(self, inputs, actual, *, missing_policy="native"):
        batch, target, mask = self._prepare(inputs, actual, missing_policy=missing_policy)
        output = self._predict_normalized(batch)
        if not mask.any():
            raise ValueError("No hay targets observados para la pérdida NeuralForecast.")
        # Use the package's loss unchanged. Version 3.1.4 MQLoss sums over
        # quantiles; divide explicitly to fix the documented optimization scale.
        loss = self.module.loss(y=target, y_hat=output, mask=mask)
        if self.quantile_levels:
            loss = loss / len(self.quantile_levels)
        point = output[..., self.quantile_levels.index(.5)] if self.quantile_levels else output
        observed = mask.bool()
        squared = (point[observed] - target[observed]).square().sum()
        return loss, squared

    def forecast_distribution(self, inputs, *, missing_policy="native"):
        batch, _, _ = self._prepare(inputs, missing_policy=missing_policy)
        output = self._predict_normalized(batch)
        output = self.module._inv_normalization(output, y_idx=0)
        if self.quantile_levels:
            quantiles = output[:, :, 0, :]
            return quantiles[..., self.quantile_levels.index(.5)], quantiles, self.quantile_levels
        return output[..., 0], None, None

    def forward(self, inputs, *, missing_policy="native"):
        return self.forecast_distribution(inputs, missing_policy=missing_policy)[0]

    def export_state(self):
        # Fixed quantile levels are loss configuration, never federated weights.
        return {name: value for name, value in self.named_parameters() if value.requires_grad}

    def import_state(self, state):
        expected = self.export_state()
        if set(expected) != set(state):
            raise ValueError("Parámetros NeuralForecast incompatibles.")
        with torch.no_grad():
            for name, value in state.items():
                if (value.shape != expected[name].shape or value.dtype != expected[name].dtype
                        or not torch.isfinite(value).all()):
                    raise ValueError(f"Tensor NeuralForecast incompatible: {name}")
                expected[name].copy_(value)
