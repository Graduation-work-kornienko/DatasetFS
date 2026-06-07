"""Process-tree + system-level metrics via psutil. Provides the SYMMETRIC
measurement layer that lets us compare DatasetFS vs WebDataset vs ImageFolder
on the same axes (CPU%, RSS, disk I/O).

Usage:
    sampler = SystemSampler()
    sampler.start()
    ... do work ...
    sampler.stop()
    aggregate = sampler.summary()  # {cpu_pct_mean, rss_max_mb, disk_read_mb, ...}
"""
from __future__ import annotations

import statistics
import threading
import time

import psutil


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    idx = (p / 100.0) * (len(xs) - 1)
    low = int(idx)
    high = min(low + 1, len(xs) - 1)
    frac = idx - low
    return xs[low] * (1 - frac) + xs[high] * frac


class SystemSampler:
    """Background thread sampling psutil at fixed interval.

    Records system-wide CPU%, RAM, disk I/O. Optionally tracks specific PIDs
    (e.g., Python + daemon) for process-tree memory + per-process CPU.
    """

    def __init__(self, interval_s: float = 0.2, track_pids: list[int] | None = None,
                 track_labels: dict[str, int] | None = None):
        self.interval_s = interval_s
        self.track_pids = list(track_pids or [])
        self.track_labels = dict(track_labels or {})
        for pid in self.track_labels.values():
            if pid not in self.track_pids:
                self.track_pids.append(pid)
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        # Prime cpu_percent (first call returns 0)
        psutil.cpu_percent(interval=None)
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        # Resolve tracked Process objects once; on disappearance we skip them.
        tracked: dict[int, psutil.Process] = {}
        for pid in self.track_pids:
            try:
                proc = psutil.Process(pid)
                proc.cpu_percent(interval=None)
                for child in proc.children(recursive=True):
                    try:
                        child.cpu_percent(interval=None)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                tracked[pid] = proc
            except psutil.NoSuchProcess:
                pass

        while not self._stop.wait(self.interval_s):
            t = time.perf_counter() - self._t0
            sample = {"t": t, "cpu_percent": psutil.cpu_percent(interval=None)}
            try:
                vm = psutil.virtual_memory()
                sample["mem_used_bytes"] = vm.used
                sample["mem_available_bytes"] = vm.available
            except Exception:
                pass
            try:
                io = psutil.disk_io_counters()
                if io is not None:
                    sample["disk_read_bytes"] = io.read_bytes
                    sample["disk_write_bytes"] = io.write_bytes
            except Exception:
                pass

            # Per-tracked-process RSS + CPU sum (e.g. Python tree + daemon).
            rss_sum = 0
            cpu_sum = 0.0
            labeled: dict[str, dict[str, float]] = {}
            for pid, proc in tracked.items():
                try:
                    rss_sum += proc.memory_info().rss
                    cpu_sum += proc.cpu_percent(interval=None)
                    # Include children (DataLoader workers spawn from main py)
                    for child in proc.children(recursive=True):
                        try:
                            rss_sum += child.memory_info().rss
                            cpu_sum += child.cpu_percent(interval=None)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            for label, pid in self.track_labels.items():
                proc = tracked.get(pid)
                if proc is None:
                    continue
                try:
                    rss = proc.memory_info().rss
                    cpu = proc.cpu_percent(interval=None)
                    for child in proc.children(recursive=True):
                        try:
                            rss += child.memory_info().rss
                            cpu += child.cpu_percent(interval=None)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    labeled[label] = {"rss_bytes": rss, "cpu_percent": cpu}
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            if tracked:
                sample["tracked_rss_bytes"] = rss_sum
                sample["tracked_cpu_percent"] = cpu_sum
            for label, vals in labeled.items():
                sample[f"{label}_rss_bytes"] = vals["rss_bytes"]
                sample[f"{label}_cpu_percent"] = vals["cpu_percent"]

            self.samples.append(sample)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._thread = None

    def summary(self) -> dict:
        """Aggregates suitable for one summary.csv row.

        - cpu_pct_mean / p95: time-averaged system CPU
        - mem_max_bytes: peak system memory in use
        - disk_read_bytes: cumulative delta (after - before)
        - tracked_rss_max_bytes: peak RSS of our process tree
        """
        if not self.samples:
            return {}

        cpus = [s["cpu_percent"] for s in self.samples]
        mems = [s["mem_used_bytes"] for s in self.samples if "mem_used_bytes" in s]
        reads = [s["disk_read_bytes"] for s in self.samples if "disk_read_bytes" in s]
        writes = [s["disk_write_bytes"] for s in self.samples if "disk_write_bytes" in s]
        rss = [s["tracked_rss_bytes"] for s in self.samples if "tracked_rss_bytes" in s]
        tracked_cpu = [s["tracked_cpu_percent"] for s in self.samples if "tracked_cpu_percent" in s]
        duration = max(
            self.samples[-1].get("t", 0) - self.samples[0].get("t", 0),
            1e-9,
        )

        out: dict = {
            "cpu_pct_mean": statistics.mean(cpus) if cpus else 0,
            "cpu_pct_p95": _percentile(cpus, 95) if cpus else 0,
            "n_samples": len(self.samples),
        }
        if mems:
            out["mem_max_bytes"] = max(mems)
            out["mem_mean_bytes"] = statistics.mean(mems)
        if reads:
            read_delta = reads[-1] - reads[0]
            out["disk_read_bytes"] = read_delta
            out["disk_read_bytes_per_s"] = read_delta / duration
        if writes:
            write_delta = writes[-1] - writes[0]
            out["disk_write_bytes"] = write_delta
            out["disk_write_bytes_per_s"] = write_delta / duration
        if rss:
            out["tracked_rss_max_bytes"] = max(rss)
            out["tracked_rss_mean_bytes"] = statistics.mean(rss)
        if tracked_cpu:
            out["tracked_cpu_pct_mean"] = statistics.mean(tracked_cpu)
            out["tracked_cpu_pct_p95"] = _percentile(tracked_cpu, 95)
        labels = set()
        for s in self.samples:
            for key in s:
                if key.endswith("_cpu_percent") and key not in ("cpu_percent", "tracked_cpu_percent"):
                    labels.add(key[:-len("_cpu_percent")])
        for label in sorted(labels):
            lcpu = [s[f"{label}_cpu_percent"] for s in self.samples if f"{label}_cpu_percent" in s]
            lrss = [s[f"{label}_rss_bytes"] for s in self.samples if f"{label}_rss_bytes" in s]
            if lcpu:
                out[f"{label}_cpu_pct_mean"] = statistics.mean(lcpu)
                out[f"{label}_cpu_pct_p95"] = _percentile(lcpu, 95)
            if lrss:
                out[f"{label}_rss_max_bytes"] = max(lrss)
                out[f"{label}_rss_mean_bytes"] = statistics.mean(lrss)
        return out
