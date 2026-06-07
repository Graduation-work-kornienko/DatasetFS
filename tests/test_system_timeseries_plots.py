from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")

from benchmarks.datasetfs_bench.reporting.system_timeseries_plots import plot_system_timeseries


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_system_timeseries_plot(tmp_path):
    _write_csv(
        tmp_path / "system_timeseries.csv",
        [
            {
                "t": "0.0",
                "cpu_percent": "20.0",
                "tracked_rss_bytes": str(100 * 1024 * 1024),
                "python_rss_bytes": str(60 * 1024 * 1024),
                "daemon_rss_bytes": str(40 * 1024 * 1024),
                "tracked_cpu_percent": "25.0",
                "python_cpu_percent": "15.0",
                "daemon_cpu_percent": "10.0",
                "disk_read_bytes": "1000",
                "disk_write_bytes": "2000",
            },
            {
                "t": "1.0",
                "cpu_percent": "35.0",
                "tracked_rss_bytes": str(120 * 1024 * 1024),
                "python_rss_bytes": str(70 * 1024 * 1024),
                "daemon_rss_bytes": str(50 * 1024 * 1024),
                "tracked_cpu_percent": "40.0",
                "python_cpu_percent": "28.0",
                "daemon_cpu_percent": "12.0",
                "disk_read_bytes": str(2 * 1024 * 1024),
                "disk_write_bytes": str(3 * 1024 * 1024),
            },
        ],
    )
    _write_csv(tmp_path / "train_events.csv", [{"train_run": "0", "start_s": "0.25", "end_s": "0.75"}])

    out = plot_system_timeseries(tmp_path)

    assert out.exists()
    assert out.stat().st_size > 0
