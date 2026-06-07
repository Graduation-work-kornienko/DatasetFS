from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")

from benchmarks.datasetfs_bench.reporting.remote_plots import plot_remote


def test_remote_plot(tmp_path):
    rows = [
        {
            "loader": "datasetfs-remote",
            "seed": "0",
            "warmup": "False",
            "steady_samples_per_second": "120.0",
            "time_to_first_batch": "1.5",
            "daemon_remote_shard_wait_latency_p95": "0.25",
            "daemon_remote_cache_hits_total_delta": "8",
            "daemon_remote_cache_misses_total_delta": "2",
        },
        {
            "loader": "webdataset-remote",
            "seed": "0",
            "warmup": "False",
            "steady_samples_per_second": "100.0",
            "time_to_first_batch": "2.0",
            "daemon_remote_shard_wait_latency_p95": "",
            "daemon_remote_cache_hits_total_delta": "",
            "daemon_remote_cache_misses_total_delta": "",
        },
    ]
    with open(tmp_path / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    out = plot_remote(tmp_path)

    assert out.exists()
    assert out.stat().st_size > 0
