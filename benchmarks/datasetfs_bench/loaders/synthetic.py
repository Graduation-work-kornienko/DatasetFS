"""Synthetic — the compute-only ceiling. Generates random tensors in-process
with NO storage I/O, so its throughput is the upper bound any real format can
approach. The gap between a format's bar and this one is its storage+decode tax.

Per-index deterministic RNG keeps runs reproducible across seeds and workers.
"""
from __future__ import annotations

from typing import ClassVar

import torch
from torch.utils.data import DataLoader, Dataset

from .base import BaseLoader
from ._common import imagefolder_collate


class _SyntheticDataset(Dataset):
    """A small pre-generated pool of random samples, cycled. Per-item cost is a
    single tensor index — NO allocation, NO RNG in the hot path — so the loader
    is as close to free as possible and its throughput is the true compute
    ceiling. (Generating per-index with a fresh torch.Generator was ~3× slower
    than ImageFolder and defeated the purpose.)"""

    def __init__(self, n: int, image_size: int, n_classes: int, seed: int, pool: int = 256):
        self.n = n
        g = torch.Generator().manual_seed(seed)
        self._imgs = torch.rand(pool, 3, image_size, image_size, generator=g)
        self._labels = torch.randint(0, max(1, n_classes), (pool,), generator=g)
        self._pool = pool

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        j = i % self._pool
        return self._imgs[j], int(self._labels[j])


class SyntheticLoader(BaseLoader):
    name: ClassVar[str] = "synthetic"

    def setup(self) -> None:
        # No data on disk; sized to resemble a real epoch.
        self._n = int(self.spec.get("synthetic_samples", 10000))
        self._n_classes = len(self.label_to_idx)

    def make_loader(self) -> DataLoader:
        ds = _SyntheticDataset(self._n, self.image_size, self._n_classes, self.seed or 0)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=imagefolder_collate,  # yields (img_tensor, int_label)
            persistent_workers=False,
            pin_memory=False,
        )

    def teardown(self) -> None:
        pass
