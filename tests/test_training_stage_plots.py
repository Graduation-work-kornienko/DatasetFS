from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from benchmarks.datasetfs_bench.reporting.plots import plot_training_stage_breakdown
from benchmarks.datasetfs_bench.reporting.report import generate_report


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_training_stage_breakdown_plot_and_report(tmp_path: Path):
    _write_csv(
        tmp_path / "summary.csv",
        [
            {
                "loader": "datasetfs",
                "seed": "0",
                "warmup": "False",
                "steady_samples_per_second": "100.0",
                "time_to_first_batch": "0.1",
                "batch_wait_fraction": "0.25",
                "forward_backward_fraction": "0.60",
                "optimizer_fraction": "0.10",
                "sys_cpu_pct_mean": "50.0",
                "sys_daemon_rss_max_bytes": str(64 * 1024 * 1024),
            }
        ],
    )

    plot = plot_training_stage_breakdown(tmp_path)
    report = generate_report(tmp_path)

    assert plot.name == "training_stage_breakdown.png"
    assert plot.exists()
    text = report.read_text(encoding="utf-8")
    assert "wait mean, %" in text
    assert "fwd/bwd mean, %" in text
    assert "training_stage_breakdown.png" in text
