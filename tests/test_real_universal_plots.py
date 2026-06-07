from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")

from benchmarks.datasetfs_bench.reporting.real_universal_plots import plot_real_universal


def test_real_universal_plot(tmp_path):
    rows = [
        {
            "name": "wiki",
            "format": "datasetfs",
            "modality": "text",
            "status": "ok",
            "samples_per_s": "320.0",
            "sys_cpu_pct_mean": "35.0",
            "sys_daemon_rss_max_bytes": str(96 * 1024 * 1024),
            "dataset_total_bytes": str(2 * 1024 ** 3),
        },
        {
            "name": "audio",
            "format": "parquet",
            "modality": "audio_text",
            "status": "ok",
            "samples_per_s": "180.0",
            "sys_cpu_pct_mean": "55.0",
            "sys_daemon_rss_max_bytes": str(128 * 1024 * 1024),
            "dataset_total_bytes": str(3 * 1024 ** 3),
        },
    ]
    with open(tmp_path / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    out = plot_real_universal(tmp_path)

    assert out.exists()
    assert out.stat().st_size > 0
