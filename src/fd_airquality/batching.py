"""Training-only epoch sampling and optional batch completion."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Sampler

from fd_airquality.data import WindowDataset


class CompleteBatchSampler(Sampler[list[int]]):
    """Draw a fresh subset per group, with optional resampling to complete the tail."""

    def __init__(
        self, size: int, batch_size: int, generator: torch.Generator, *,
        max_windows: int | None = None, group_sizes: Sequence[int] | None = None,
        pad_last_batch: bool = True,
    ):
        if size < 1 or batch_size < 1:
            raise ValueError("El muestreo requiere ventanas y batch_size positivos.")
        if max_windows is not None and (type(max_windows) is not int or max_windows < 1):
            raise ValueError("max_train_windows debe ser un entero positivo.")
        self.size, self.batch_size, self.generator = size, batch_size, generator
        self.pad_last_batch = pad_last_batch
        self.group_sizes = (size,) if group_sizes is None else tuple(group_sizes)
        if any(n < 0 for n in self.group_sizes) or sum(self.group_sizes) != size:
            raise ValueError("Los grupos de entrenamiento no cubren el conjunto de ventanas.")
        self.group_samples = tuple(min(n, max_windows) if max_windows is not None else n
                                   for n in self.group_sizes)
        self.epoch_size = sum(self.group_samples)
        self.selected_indices = torch.empty(0, dtype=torch.long)

    def __len__(self):
        return (self.epoch_size + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.epoch_size == self.size:
            order = torch.randperm(self.size, generator=self.generator)
        elif len(self.group_sizes) == 1:
            order = torch.randperm(self.size, generator=self.generator)[:self.epoch_size]
        else:
            groups = []
            offset = 0
            for size, draws in zip(self.group_sizes, self.group_samples, strict=True):
                if draws == size:
                    groups.append(torch.arange(offset, offset + size))
                else:
                    groups.append(offset + torch.randperm(size, generator=self.generator)[:draws])
                offset += size
            # Centralized: a per-station quota, then mixed optimizer batches.
            order = torch.cat(groups)[torch.randperm(self.epoch_size, generator=self.generator)]
        self.selected_indices = order
        for first in range(0, self.epoch_size, self.batch_size):
            batch = order[first:first + self.batch_size].tolist()
            missing = self.batch_size - len(batch)
            if missing and self.pad_last_batch:
                fills = torch.randint(self.epoch_size, (missing,), generator=self.generator)
                batch.extend((fills if self.epoch_size == self.size else order[fills]).tolist())
            yield batch


def training_padding_counts(dataset, selected_mask: np.ndarray | None = None) -> tuple[int, int]:
    """Virtual slots in the union of windows visited, never in repeated draws."""
    if isinstance(dataset, WindowDataset):
        return dataset.padding_counts(dataset.starts[selected_mask] if selected_mask is not None else None)
    if isinstance(dataset, ConcatDataset):
        context = targets = offset = 0
        for child in dataset.datasets:
            end = offset + len(child)
            a, b = training_padding_counts(child, selected_mask[offset:end] if selected_mask is not None else None)
            context, targets = context + a, targets + b
            offset = end
        return context, targets
    return 0, 0
