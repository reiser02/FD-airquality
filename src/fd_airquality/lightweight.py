"""Small point forecasters; uncertainty is evaluated separately from fitting."""

import torch
from torch import nn

from fd_airquality.missing import impute_context

LINEAR_LAGS = (1, 2, 3, 24, 48, 72)
STATION_CONTROLS = ("seasonal-naive", "holt")
HOLT_GRID = tuple((a, b) for a in (.2, .5, .8) for b in (.05, .2, .5))


def _context(inputs, policy):
    if inputs.ndim != 2 or torch.isinf(inputs).any() or not torch.isfinite(inputs).any(dim=1).all():
        raise ValueError("Contexto inválido o sin ninguna observación.")
    return impute_context(inputs, policy)


class SeasonalNaive(nn.Module):
    """Repeat the last calendar cycle, with an explicit causal fallback for gaps."""

    handles_missing_context = True

    def __init__(self, horizon: int, season: int):
        super().__init__()
        self.horizon, self.season = horizon, season

    def forward(self, inputs, *, missing_policy="native"):
        values = _context(inputs, missing_policy)
        steps = torch.arange(self.horizon, device=values.device)
        width = values.shape[1]
        if width < self.season:
            raise ValueError("SeasonalNaive requiere al menos un ciclo en el contexto.")
        positions = torch.arange(width, device=values.device)
        last = torch.where(torch.isfinite(values), positions, -1).amax(dim=1, keepdim=True)
        result = values.gather(1, last).expand(-1, self.horizon)
        for cycle in (3, 2, 1):
            positions = width - cycle * self.season + steps % self.season
            older = values[:, positions.clamp_min(0)]
            result = torch.where((positions >= 0) & torch.isfinite(older), older, result)
        return result


class Holt(nn.Module):
    """Additive level/trend; missing hours advance the state without an update."""

    handles_missing_context = True

    def __init__(self, horizon: int):
        super().__init__()
        self.horizon = horizon
        self.register_buffer("coefficients", torch.tensor([.5, .2]))

    def set_coefficients(self, alpha: float, beta: float) -> None:
        self.coefficients.copy_(self.coefficients.new_tensor((alpha, beta)))

    def forward(self, inputs, *, missing_policy="native"):
        values = _context(inputs, missing_policy)
        alpha, beta = self.coefficients
        level = values.new_zeros(len(values))
        trend = torch.zeros_like(level)
        initialized = torch.zeros_like(level, dtype=torch.bool)
        for value in values.unbind(1):
            observed = torch.isfinite(value)
            advance = level + trend
            updated = alpha * torch.where(observed, value, advance) + (1 - alpha) * advance
            trend = torch.where(initialized & observed, beta * (updated - level) + (1 - beta) * trend, trend)
            level = torch.where(initialized, updated, torch.where(observed, value, level))
            initialized = initialized | observed
        lead = torch.arange(1, self.horizon + 1, device=values.device)
        return level[:, None] + trend[:, None] * lead


class LagLinear(nn.Module):
    """One direct linear map to all leads, initialized to repeat the last input."""

    handles_missing_context = True

    def __init__(self, horizon: int, scaler_type: str):
        super().__init__()
        from neuralforecast.common._scalers import TemporalNorm

        self.scaler = TemporalNorm(scaler_type=scaler_type, dim=1)
        self.linear = nn.Linear(len(LINEAR_LAGS), horizon)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.weight[:, 0] = 1
            self.linear.bias.zero_()

    def _predict(self, inputs, policy):
        mask = torch.isfinite(inputs)
        if torch.isinf(inputs).any() or not mask.any(dim=1).all():
            raise ValueError("Contexto de regresión inválido o sin observaciones.")
        normalized = self.scaler.transform(inputs.nan_to_num(), mask=mask)
        values = impute_context(normalized.masked_fill(~mask, float("nan")), policy)
        return self.linear(values[:, [-lag for lag in LINEAR_LAGS]])

    def forward(self, inputs, *, missing_policy="ffill"):
        return self.scaler.inverse_transform(self._predict(inputs, missing_policy))

    def training_objective(self, inputs, actual, *, missing_policy="ffill"):
        prediction = self._predict(inputs, missing_policy)
        if prediction.shape != actual.shape or not torch.isfinite(prediction).all():
            raise ValueError("Predicción lineal incompatible o no finita.")
        mask = torch.isfinite(actual)
        target = self.scaler.scaler(actual.nan_to_num(), self.scaler.x_shift, self.scaler.x_scale)
        squared = (prediction[mask] - target[mask]).square()
        return squared.mean(), squared.sum()
