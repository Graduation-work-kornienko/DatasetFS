"""Scrape the Go daemon's /metrics endpoint and compute deltas per run cell.

Counters are monotonic — we record before/after and report the delta.
Gauges (active_pipelines, uptime) are snapshot at end-of-cell.
Histograms (load_latency) are summarised by their final percentiles.
"""
from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from typing import Any

import requests


def snapshot(url: str = "http://localhost:51409", timeout: float = 2.0) -> dict:
    """Return a dict of {counters, gauges, histograms}, or {} on failure."""
    try:
        r = requests.get(f"{url}/metrics", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[daemon-metrics] scrape failed: {e}", flush=True)
        return {}


def cell_summary(before: dict, after: dict) -> dict[str, Any]:
    """Build a flat dict suitable for one row in summary.csv.

    - For counters: report `daemon_<name>_delta` (after - before).
    - For gauges: report `daemon_<name>` (the after value).
    - For histograms: report p50/p95/p99/max/count for every daemon histogram
      AT END of the cell (since each tracker is a ring buffer, this is
      approximate across very long runs but accurate for single-cell durations).
    """
    out: dict[str, Any] = {}
    if not after:
        return out

    bc = before.get("counters", {}) if before else {}
    for k, v in after.get("counters", {}).items():
        out[f"daemon_{k}_delta"] = v - bc.get(k, 0)

    for k, v in after.get("gauges", {}).items():
        out[f"daemon_{k}"] = v

    bh = before.get("histograms", {}) if before else {}
    for name, h in after.get("histograms", {}).items():
        if not h:
            continue
        prefix = f"daemon_{name}"
        out[f"{prefix}_p50"] = h.get("p50_seconds", 0)
        out[f"{prefix}_p95"] = h.get("p95_seconds", 0)
        out[f"{prefix}_p99"] = h.get("p99_seconds", 0)
        out[f"{prefix}_max"] = h.get("max_seconds", 0)
        out[f"{prefix}_count"] = h.get("count", 0) - (bh.get(name, {}) or {}).get("count", 0)

    return out


def flatten_snapshot(metrics: dict) -> dict[str, Any]:
    """Flatten one /metrics response for daemon_timeseries.csv."""
    out: dict[str, Any] = {}
    if not metrics:
        return out
    for k, v in metrics.get("counters", {}).items():
        out[f"counter_{k}"] = v
    for k, v in metrics.get("gauges", {}).items():
        out[f"gauge_{k}"] = v
    for name, h in metrics.get("histograms", {}).items():
        if not h:
            continue
        prefix = f"hist_{name}"
        out[f"{prefix}_p50_seconds"] = h.get("p50_seconds", 0)
        out[f"{prefix}_p95_seconds"] = h.get("p95_seconds", 0)
        out[f"{prefix}_p99_seconds"] = h.get("p99_seconds", 0)
        out[f"{prefix}_max_seconds"] = h.get("max_seconds", 0)
        out[f"{prefix}_count"] = h.get("count", 0)
    return out


class DaemonSampler:
    """Background sampler for the daemon /metrics endpoint.

    `cell_summary` is enough for summary.csv, but timeline plots and debugging
    need raw samples over time. This sampler keeps those samples in memory; runs
    are short enough that avoiding incremental file I/O keeps integration simple.
    """

    def __init__(self, url: str = "http://localhost:51409", interval_s: float = 0.5,
                 context: dict[str, Any] | None = None):
        self.url = url
        self.interval_s = interval_s
        self.context = dict(context or {})
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            t = time.perf_counter() - self._t0
            row = {**self.context, "t": t, "wall_time_s": time.time()}
            row.update(flatten_snapshot(snapshot(self.url, timeout=min(2.0, self.interval_s))))
            self.samples.append(row)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._thread = None


def write_rows_union(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write heterogeneous metric rows to CSV using unioned columns."""
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
