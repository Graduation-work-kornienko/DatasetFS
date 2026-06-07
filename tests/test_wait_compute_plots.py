from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from benchmarks.datasetfs_bench.reporting.report import generate_report
from benchmarks.datasetfs_bench.reporting.wait_compute_plots import plot_wait_compute


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_wait_compute_plot_with_detailed_stage_columns(tmp_path: Path):
    _write_csv(
        tmp_path / "summary.csv",
        [
            {
                "loader": "datasetfs",
                "format": "raw",
                "seed": "0",
                "warmup": "False",
                "batch_wait_fraction": "0.30",
                "forward_backward_fraction": "0.55",
                "optimizer_fraction": "0.10",
            },
            {
                "loader": "imagefolder",
                "seed": "0",
                "warmup": "False",
                "batch_wait_fraction": "0.10",
                "forward_backward_fraction": "0.70",
                "optimizer_fraction": "0.10",
            },
        ],
    )
    out = plot_wait_compute(tmp_path)

    assert out.name == "wait_compute_breakdown.png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_wait_compute_plot_with_legacy_stall_fraction(tmp_path: Path):
    _write_csv(
        tmp_path / "summary.csv",
        [
            {
                "loader": "datasetfs",
                "seed": "0",
                "warmup": "False",
                "stall_fraction": "0.25",
            }
        ],
    )
    out = plot_wait_compute(tmp_path)
    report = generate_report(tmp_path)

    assert out.exists()
    assert "wait_compute_breakdown.png" in report.read_text(encoding="utf-8")
