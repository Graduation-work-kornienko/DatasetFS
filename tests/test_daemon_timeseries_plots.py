from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")

from benchmarks.datasetfs_bench.reporting.daemon_timeseries_plots import plot_daemon_timeseries


def test_daemon_timeseries_plot(tmp_path):
    ts_path = tmp_path / "daemon_timeseries.csv"
    rows = [
        {
            "loader": "datasetfs",
            "seed": "1",
            "t": "0.0",
            "gauge_active_pipelines": "1",
            "counter_load_requests_total": "10",
            "counter_shm_write_bytes_total": "1024",
            "hist_load_latency_p95_seconds": "0.01",
        },
        {
            "loader": "datasetfs",
            "seed": "1",
            "t": "1.0",
            "gauge_active_pipelines": "1",
            "counter_load_requests_total": "30",
            "counter_shm_write_bytes_total": "4096",
            "hist_load_latency_p95_seconds": "0.02",
        },
    ]
    with open(ts_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    out = plot_daemon_timeseries(tmp_path)

    assert out.exists()
    assert out.stat().st_size > 0
