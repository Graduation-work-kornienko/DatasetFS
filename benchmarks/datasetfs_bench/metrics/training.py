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
    time_to_first_batch: float        # from iter start → first batch yielded
    fetch_latency_seconds: list[float] = field(default_factory=list)
    compute_seconds: list[float] = field(default_factory=list)

    @property
    def samples_per_second(self) -> float:
        return self.n_samples / self.wall_seconds if self.wall_seconds > 0 else 0.0

    @property
    def stall_fraction(self) -> float:
        """Fraction of total batch time spent WAITING for data vs computing
        on it. High = data loader is the bottleneck."""
        total_fetch = sum(self.fetch_latency_seconds)
        total_compute = sum(self.compute_seconds)
        denom = total_fetch + total_compute
        return total_fetch / denom if denom > 0 else 0.0

    def summary(self) -> dict[str, float]:
        return {
            "epoch": self.epoch,
            "n_batches": self.n_batches,
            "n_samples": self.n_samples,
            "wall_seconds": self.wall_seconds,
            "samples_per_second": self.samples_per_second,
            "time_to_first_batch": self.time_to_first_batch,
            "stall_fraction": self.stall_fraction,
            "fetch_p50": _percentile(self.fetch_latency_seconds, 50),
            "fetch_p95": _percentile(self.fetch_latency_seconds, 95),
            "fetch_p99": _percentile(self.fetch_latency_seconds, 99),
            "compute_p50": _percentile(self.compute_seconds, 50),
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
