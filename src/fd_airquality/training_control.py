"""Training budgets and epoch-level controls; never stop the federated arrival calendar."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import Dataset

from fd_airquality.lightweight import STATION_CONTROLS
from fd_airquality.models import load_shared_state, shared_state
from fd_airquality.settings import FEDERATED_MODES


def training_stage_count(settings, mode: str) -> int:
    """Count training fits, not validation checkpoints."""
    if mode == "zero-shot" or settings.model in STATION_CONTROLS:
        return 0
    return settings.num_server_rounds if mode in FEDERATED_MODES else 1


def controls_enabled(settings) -> bool:
    # A single epoch has no later epoch to stop or whose LR can be reduced.
    return settings.local_epochs > 1 and bool(settings.early_stopping or settings.lr_scheduler != "none")


class ContextSupervisionDataset(Dataset):
    """Old and monitoring values may give context, but not reconstruction labels."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        inputs, actual = self.dataset[index]
        arrival = getattr(self.dataset, "arrival_lower", None)
        monitor = getattr(self.dataset, "monitoring_interval", None)
        allowed = np.ones(len(inputs), dtype=bool)
        if arrival is not None or monitor is not None:
            times = self.dataset.context_timestamps(index)
            if arrival is not None:
                allowed &= times >= arrival.value
            if monitor is not None:
                lower, upper = monitor
                allowed &= (times < lower.value) | (times >= upper.value)
        return inputs, actual, torch.from_numpy(allowed)


class TrainingControl:
    def __init__(self, model, optimizer, settings):
        self.model, self.optimizer, self.settings = model, optimizer, settings
        self.best_score, self.best_epoch, self.bad_epochs = math.inf, 0, 0
        self.best_state = None
        self.reductions = 0
        self.scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=settings.lr_scheduler_factor,
            patience=settings.lr_scheduler_patience, threshold=settings.early_stopping_min_delta,
            threshold_mode="abs", min_lr=settings.min_learning_rate,
        ) if settings.lr_scheduler == "plateau" else None)

    def observe(self, epoch: int, score: float) -> bool:
        if not math.isfinite(score):
            raise ValueError("La métrica de monitorización de entrenamiento no es finita.")
        if score < self.best_score - self.settings.early_stopping_min_delta:
            self.best_score, self.best_epoch, self.bad_epochs = score, epoch, 0
            self.best_state = {key: value.detach().cpu().clone() for key, value in shared_state(self.model).items()}
        else:
            self.bad_epochs += 1
        if self.scheduler is not None:
            before = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(score)
            self.reductions += int(self.optimizer.param_groups[0]["lr"] < before)
        return bool(self.settings.early_stopping and self.bad_epochs >= self.settings.early_stopping_patience)

    def restore(self) -> None:
        if self.settings.early_stopping and self.best_state is not None:
            load_shared_state(self.model, self.best_state)
