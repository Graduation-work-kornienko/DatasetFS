from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from benchmarks.datasetfs_bench.reporting.pipeline_memory_plots import plot_pipeline_memory
from benchmarks.datasetfs_bench.reporting.report import generate_report


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_pipeline_memory_plot_and_report(tmp_path: Path):
    _write_csv(
        tmp_path / "memory_timeseries.csv",
        [
            {
                "scenario": "pipeline_memory",
                "mode": "no_mutation",
                "cycle": "0",
                "samples": "9",
                "drain_elapsed_s": "0.010",
                "daemon_rss_mib": "16.0",
            },
            {
                "scenario": "pipeline_memory",
                "mode": "no_mutation",
                "cycle": "1",
                "samples": "9",
                "drain_elapsed_s": "0.011",
                "daemon_rss_mib": "16.1",
            },
            {
                "scenario": "pipeline_memory",
                "mode": "bounded_replace",
                "cycle": "0",
                "samples": "9",
                "drain_elapsed_s": "0.020",
                "daemon_rss_mib": "20.0",
            },
            {
                "scenario": "pipeline_memory",
                "mode": "bounded_replace",
                "cycle": "1",
                "samples": "9",
                "drain_elapsed_s": "0.021",
                "daemon_rss_mib": "20.2",
            },
        ],
    )
    _write_csv(
        tmp_path / "summary.csv",
        [
            {
                "scenario": "pipeline_memory",
                "mode": "no_mutation",
                "cycles": "2",
                "warmup_cycles": "0",
                "files": "9",
                "replacements_per_cycle": "0",
                "rss_min_mib": "16.0",
                "rss_max_mib": "16.1",
                "rss_growth_mib": "0.1",
                "rss_slope_mib_per_cycle": "0.1",
                "drain_elapsed_mean_s": "0.0105",
            },
            {
                "scenario": "pipeline_memory",
                "mode": "bounded_replace",
                "cycles": "2",
                "warmup_cycles": "0",
                "files": "9",
                "replacements_per_cycle": "3",
                "rss_min_mib": "20.0",
                "rss_max_mib": "20.2",
                "rss_growth_mib": "0.2",
                "rss_slope_mib_per_cycle": "0.2",
                "drain_elapsed_mean_s": "0.0205",
            },
        ],
    )

    plot = plot_pipeline_memory(tmp_path)
    report = generate_report(tmp_path)

    assert plot.name == "pipeline_memory.png"
    assert plot.exists()
    assert plot.stat().st_size > 0
    text = report.read_text(encoding="utf-8")
    assert "| mode | cycles | warmup | files" in text
    assert "pipeline_memory.png" in text
    assert "memory_timeseries.csv" in text
