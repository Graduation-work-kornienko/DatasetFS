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
    loader,
    optim: torch.optim.Optimizer,
    loss_fn: Callable,
    epoch_idx: int,
    *,
    max_batches: int | None = None,
    warmup_batches: int = 0,
) -> EpochStats:
    """Train for one epoch, returning timing stats.

    `loader`: a DataLoader (NOT a pre-built iterator). We call `iter()` here,
    INSIDE the timed region, so DataLoader worker-spawn cost is attributed
    uniformly across formats — map-style loaders spawn during iter(), iterable
    loaders during the first next(); measuring from before iter() keeps the
    comparison fair (see EpochStats.steady_* for the rationale).

    `warmup_batches`: number of leading batches excluded from the steady-state
    throughput window (drops the one-time spawn + priming ramp). They are still
    trained on and still counted in the whole-epoch wall number.
    """
    model.train()

    t_start = time.perf_counter()
    it = iter(loader)                       # worker spawn happens here/at first next()
    timed = TimedLoaderIter(it)

    n_batches = 0
    n_samples = 0
    compute_times: list[float] = []
    losses: list[float] = []

    time_to_first: float | None = None
    t_steady_start: float | None = None
    steady_n_samples = 0

    for batch_idx, (images, targets) in enumerate(timed):
        if time_to_first is None:
            time_to_first = time.perf_counter() - t_start
        if batch_idx == warmup_batches:
            # Steady window opens just before we process the first post-warmup batch.
            t_steady_start = time.perf_counter()

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
        if batch_idx >= warmup_batches:
            steady_n_samples += images.shape[0]

        if max_batches is not None and n_batches >= max_batches:
            break

    t_end = time.perf_counter()
    wall = t_end - t_start
    steady_wall = (t_end - t_steady_start) if t_steady_start is not None else 0.0

    return EpochStats(
        epoch=epoch_idx,
        n_batches=n_batches,
        n_samples=n_samples,
        wall_seconds=wall,
        time_to_first_batch=time_to_first or 0.0,
        fetch_latency_seconds=timed.fetch_latencies,
        compute_seconds=compute_times,
        steady_n_samples=steady_n_samples,
        steady_wall_seconds=steady_wall,
    )
