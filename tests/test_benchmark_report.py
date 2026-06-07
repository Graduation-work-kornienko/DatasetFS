from __future__ import annotations

import csv

from benchmarks.datasetfs_bench.reporting.report import generate_report


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_generate_benchmark_report(tmp_path):
    _write_csv(
        tmp_path / "summary.csv",
        [
            {
                "loader": "datasetfs",
                "seed": "0",
                "warmup": "False",
                "steady_samples_per_second": "120.0",
                "time_to_first_batch": "0.25",
                "stall_fraction": "0.1",
                "sys_cpu_pct_mean": "42.0",
                "sys_daemon_rss_max_bytes": str(128 * 1024 * 1024),
            },
            {
                "loader": "datasetfs",
                "seed": "1",
                "warmup": "False",
                "steady_samples_per_second": "140.0",
                "time_to_first_batch": "0.30",
                "stall_fraction": "0.2",
                "sys_cpu_pct_mean": "44.0",
                "sys_daemon_rss_max_bytes": str(160 * 1024 * 1024),
            },
        ],
    )
    _write_csv(
        tmp_path / "missing.csv",
        [{"name": "flickr30k", "modality": "image_text", "prepare": "python -m scripts.datasets.prepare_real_universal flickr30k"}],
    )
    (tmp_path / "daemon_timeseries.png").write_bytes(b"png")

    out = generate_report(tmp_path)
    text = out.read_text(encoding="utf-8")

    assert "# Benchmark Report" in text
    assert "| datasetfs | 2 | 130.00" in text
    assert "flickr30k" in text
    assert "[daemon_timeseries.png](daemon_timeseries.png)" in text


def test_generate_mutation_report(tmp_path):
    _write_csv(
        tmp_path / "summary.csv",
        [
            {
                "mode": "mixed",
                "mutation_rate_s": "5.0",
                "samples_per_s": "250.0",
                "consistency_violations": "0",
                "mutations_succeeded": "2",
                "mutations_failed": "0",
                "mutation_latency_mean_ms": "7.5",
                "cpu_pct_mean": "20.0",
                "tracked_rss_max_bytes": str(320 * 1024 * 1024),
            }
        ],
    )

    out = generate_report(tmp_path)
    text = out.read_text(encoding="utf-8")

    assert "| mode | mutations/s" in text
    assert "| mixed | 5.0 | 1 | 250.00 | 0 | 2/0 | 7.50" in text
