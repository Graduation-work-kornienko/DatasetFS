"""Scrape the Go daemon's /metrics endpoint and compute deltas per run cell.

Counters are monotonic — we record before/after and report the delta.
Gauges (active_pipelines, uptime) are snapshot at end-of-cell.
Histograms (load_latency) are summarised by their final percentiles.
"""
from __future__ import annotations

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
    - For histograms: report p50/p95/p99/count of `load_latency` AT END of
      the cell (since the tracker is a ring buffer, this is approximate
      across very long runs but accurate for single-cell durations).
    """
    out: dict[str, Any] = {}
    if not after:
        return out

    bc = before.get("counters", {}) if before else {}
    for k, v in after.get("counters", {}).items():
        out[f"daemon_{k}_delta"] = v - bc.get(k, 0)

    for k, v in after.get("gauges", {}).items():
        out[f"daemon_{k}"] = v

    h = after.get("histograms", {}).get("load_latency", {})
    if h:
        out["daemon_load_latency_p50"] = h.get("p50_seconds", 0)
        out["daemon_load_latency_p95"] = h.get("p95_seconds", 0)
        out["daemon_load_latency_p99"] = h.get("p99_seconds", 0)
        out["daemon_load_latency_max"] = h.get("max_seconds", 0)
        out["daemon_load_latency_count"] = h.get("count", 0)

    return out
