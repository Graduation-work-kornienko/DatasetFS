"""Format-agnostic training loop. Consumes any DataLoader that yields
(images, targets) and produces per-epoch EpochStats."""
from __future__ import annotations

import time
from typing import Callable

import torch
import torch.nn as nn

from ..metrics.training import EpochStats, TimedLoaderIter


def train_one_epoch(
    model: nn.Module,
    loader_iter,
    optim: torch.optim.Optimizer,
    loss_fn: Callable,
    epoch_idx: int,
    *,
    max_batches: int | None = None,
) -> EpochStats:
    """Train for one epoch, returning timing stats.

    `loader_iter`: an already-instantiated iterator over the DataLoader.
    We wrap it to capture per-batch fetch latency.
    """
    model.train()
    timed = TimedLoaderIter(loader_iter)

    n_batches = 0
    n_samples = 0
    compute_times: list[float] = []
    losses: list[float] = []

    t_start = time.perf_counter()
    time_to_first: float | None = None

    for batch_idx, (images, targets) in enumerate(timed):
        if time_to_first is None:
            time_to_first = time.perf_counter() - t_start

        t_compute_start = time.perf_counter()
        optim.zero_grad()
        out = model(images)
        loss = loss_fn(out, targets)
        loss.backward()
        optim.step()
        compute_times.append(time.perf_counter() - t_compute_start)

        losses.append(float(loss.item()))
        n_batches += 1
        n_samples += images.shape[0]

        if max_batches is not None and n_batches >= max_batches:
            break

    wall = time.perf_counter() - t_start

    return EpochStats(
        epoch=epoch_idx,
        n_batches=n_batches,
        n_samples=n_samples,
        wall_seconds=wall,
        time_to_first_batch=time_to_first or 0.0,
        fetch_latency_seconds=timed.fetch_latencies,
        compute_seconds=compute_times,
    )
