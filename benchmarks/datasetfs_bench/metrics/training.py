"""Per-batch + per-epoch timing metrics. Phase 2 scope: enough to compute
throughput (samples/sec), TTFB, and latency percentiles. Phase 3 expands
this with psutil + daemon /metrics."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import median
from typing import Iterator

import numpy as np


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    return float(np.percentile(xs, p))


@dataclass
class EpochStats:
    """One epoch's worth of timing summary."""

    epoch: int
    n_batches: int
    n_samples: int
    wall_seconds: float
    time_to_first_batch: float        # from iter start (incl. worker spawn) → first batch
    fetch_latency_seconds: list[float] = field(default_factory=list)
    compute_seconds: list[float] = field(default_factory=list)
    zero_grad_seconds: list[float] = field(default_factory=list)
    forward_backward_seconds: list[float] = field(default_factory=list)
    optimizer_step_seconds: list[float] = field(default_factory=list)
    # Steady-state window: samples/time AFTER the first `warmup_batches` batches
    # are dropped. This excludes the one-time DataLoader worker-spawn + pipeline
    # priming cost UNIFORMLY across formats. Without it the wall-based number
    # unfairly penalizes IterableDataset loaders (DatasetFS/WebDataset/TFRecord),
    # whose workers spawn lazily on the first next() — inside the timed region —
    # while map-style loaders (ImageFolder/LMDB/HDF5) spawn during iter(), which
    # historically fell outside it. (Diagnosed 2026-06-03 via profiling/ttfb_probe.py.)
    steady_n_samples: int = 0
    steady_wall_seconds: float = 0.0
    warmup_batches: int = 0

    def _steady_fetch_seconds(self) -> list[float]:
        return self.fetch_latency_seconds[self.warmup_batches:]

    def _steady_compute_seconds(self) -> list[float]:
        return self.compute_seconds[self.warmup_batches:]

    @property
    def samples_per_second(self) -> float:
        return self.n_samples / self.wall_seconds if self.wall_seconds > 0 else 0.0

    @property
    def steady_samples_per_second(self) -> float:
        """Throughput over the post-warmup steady window. Falls back to the
        whole-epoch number if the epoch was too short to have a steady window."""
        if self.steady_wall_seconds > 0 and self.steady_n_samples > 0:
            return self.steady_n_samples / self.steady_wall_seconds
        return self.samples_per_second

    @property
    def stall_fraction(self) -> float:
        """Fraction of total batch time spent WAITING for data vs computing
        on it. High = data loader is the bottleneck."""
        total_fetch = sum(self.fetch_latency_seconds)
        total_compute = sum(self.compute_seconds)
        denom = total_fetch + total_compute
        return total_fetch / denom if denom > 0 else 0.0

    @property
    def batch_wait_fraction(self) -> float:
        return self.stall_fraction

    @property
    def steady_batch_wait_fraction(self) -> float:
        total_fetch = sum(self._steady_fetch_seconds())
        total_compute = sum(self._steady_compute_seconds())
        denom = total_fetch + total_compute
        return total_fetch / denom if denom > 0 else 0.0


    @property
    def forward_backward_fraction(self) -> float:
        total_fetch = sum(self.fetch_latency_seconds)
        total_compute = sum(self.compute_seconds)
        denom = total_fetch + total_compute
        return sum(self.forward_backward_seconds) / denom if denom > 0 else 0.0


    @property
    def optimizer_fraction(self) -> float:
        total_fetch = sum(self.fetch_latency_seconds)
        total_compute = sum(self.compute_seconds)
        denom = total_fetch + total_compute
        return sum(self.optimizer_step_seconds) / denom if denom > 0 else 0.0

    def summary(self) -> dict[str, float]:
        return {
            "epoch": self.epoch,
            "n_batches": self.n_batches,
            "n_samples": self.n_samples,
            "wall_seconds": self.wall_seconds,
            "samples_per_second": self.samples_per_second,
            "steady_samples_per_second": self.steady_samples_per_second,
            "steady_n_samples": self.steady_n_samples,
            "steady_wall_seconds": self.steady_wall_seconds,
            "time_to_first_batch": self.time_to_first_batch,
            "stall_fraction": self.stall_fraction,
            "fetch_p50": _percentile(self.fetch_latency_seconds, 50),
            "fetch_p95": _percentile(self.fetch_latency_seconds, 95),
            "fetch_p99": _percentile(self.fetch_latency_seconds, 99),
            "compute_p50": _percentile(self.compute_seconds, 50),
            "batch_wait_total_s": sum(self.fetch_latency_seconds),
            "compute_total_s": sum(self.compute_seconds),
            "steady_batch_wait_total_s": sum(self._steady_fetch_seconds()),
            "steady_compute_total_s": sum(self._steady_compute_seconds()),
            "zero_grad_total_s": sum(self.zero_grad_seconds),
            "forward_backward_total_s": sum(self.forward_backward_seconds),
            "optimizer_step_total_s": sum(self.optimizer_step_seconds),
            "batch_wait_fraction": self.batch_wait_fraction,
            "steady_batch_wait_fraction": self.steady_batch_wait_fraction,
            "forward_backward_fraction": self.forward_backward_fraction,
            "optimizer_fraction": self.optimizer_fraction,
            "zero_grad_p50": _percentile(self.zero_grad_seconds, 50),
            "forward_backward_p50": _percentile(self.forward_backward_seconds, 50),
            "optimizer_step_p50": _percentile(self.optimizer_step_seconds, 50),
        }


class TimedLoaderIter:
    """Wraps a DataLoader iterator and measures the time spent **waiting**
    for each batch (i.e., the gap between consumer requesting next() and the
    loader returning it). This is the metric that captures whether the data
    pipeline is keeping up with the consumer."""

    def __init__(self, loader_iter: Iterator):
        self._it = loader_iter
        self.fetch_latencies: list[float] = []
        self._last_yield_time: float | None = None

    def __iter__(self):
        return self

    def __next__(self):
        t0 = time.perf_counter()
        try:
            batch = next(self._it)
        except StopIteration:
            raise
        dt = time.perf_counter() - t0
        self.fetch_latencies.append(dt)
        self._last_yield_time = time.perf_counter()
        return batch
