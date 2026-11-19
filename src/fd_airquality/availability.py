"""Training-only availability, with NeuralForecast's exact univariate minima."""

from __future__ import annotations

import json
from copy import copy

import numpy as np


def parse_threshold(value) -> tuple[float, float]:
    value = json.loads(value) if isinstance(value, str) else value
    pair = [value, value] if type(value) is float else value
    if (not isinstance(pair, (tuple, list)) or len(pair) != 2
            or any(type(item) is not float or not np.isfinite(item) or not 0 <= item <= 1
                   for item in pair)):
        raise ValueError("El umbral requiere un float en [0,1] o dos floats [contexto, targets].")
    return tuple(pair)


def availability_threshold(settings) -> tuple[float, float]:
    from fd_airquality.models import MODEL_NAMES

    mapping = json.loads(settings.training_availability)
    if not isinstance(mapping, dict) or set(mapping) - set(MODEL_NAMES):
        raise ValueError("training_availability debe asociar modelos conocidos a umbrales.")
    for value in mapping.values():
        parse_threshold(value)
    default = parse_threshold(settings.training_data_availability_threshold)
    return parse_threshold(mapping[settings.model]) if settings.model in mapping else default


def availability_minima(context: int, horizon: int, threshold) -> tuple[int, int]:
    alpha, beta = parse_threshold(threshold)
    return max(1, int(context * alpha)), max(1, int(horizon * beta))


def filter_training_windows(dataset, settings):
    """Never apply training thresholds to the validation/test scoring support."""
    if dataset.split != "train":
        raise ValueError("Los umbrales solo filtran entrenamiento.")
    context_min, target_min = availability_minima(
        dataset.context, dataset.horizon, availability_threshold(settings),
    )
    starts = dataset.starts
    context_counts, target_counts = dataset.observation_counts()
    keep = (context_counts >= context_min) & (target_counts >= target_min)
    result = copy(dataset)
    result.starts = starts[keep]
    result.candidate_windows = len(starts)
    result.availability_excluded_windows = int((~keep).sum())
    result.availability_minima = context_min, target_min
    return result
