from __future__ import annotations

import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from benchmarks.datasetfs_bench.reporting.mutation_plots import plot_mutation
from benchmarks.datasetfs_bench.reporting.report import generate_report
from benchmarks.datasetfs_bench.runner.mutation_bench import (
    PlannedMutation,
    prepare_flat_webdataset,
    _replace_webdataset_files,
    _write_rows_union,
)


def _write_imagefolder(root: Path) -> None:
    for cls in ["a", "b"]:
        (root / cls).mkdir(parents=True, exist_ok=True)
    for i in range(4):
        cls = "a" if i % 2 == 0 else "b"
        (root / cls / f"img{i}.jpg").write_bytes(f"jpeg-{i}".encode())


def test_webdataset_format_mutation_rewrites_selected_members(tmp_path: Path):
    src = tmp_path / "imagefolder"
    _write_imagefolder(src)
    out = tmp_path / "webdataset"
    names = prepare_flat_webdataset(src, out, max_files=None, shard_target_bytes=20, seed=1)

    target = names[0]
    _replace_webdataset_files(out, [PlannedMutation(target, b"changed")])

    found = False
    for shard in out.glob("*.tar"):
        with tarfile.open(shard, "r") as tf:
            for member in tf.getmembers():
                if member.name == target:
                    assert tf.extractfile(member).read() == b"changed"
                    found = True
    assert found


def test_format_mutation_plot_and_report(tmp_path: Path):
    _write_rows_union(
        tmp_path / "summary.csv",
        [
            {
                "scenario": "format_mutation",
                "format": "datasetfs",
                "operation": "replace",
                "changed_files": "1",
                "repeat": "0",
                "elapsed_s": "0.01",
                "mean_operation_ms": "10.0",
                "operations_succeeded": "1",
                "operations_failed": "0",
                "bytes_written": "1024",
            },
            {
                "scenario": "format_mutation",
                "format": "webdataset",
                "operation": "replace",
                "changed_files": "1",
                "repeat": "0",
                "elapsed_s": "0.20",
                "mean_operation_ms": "200.0",
                "operations_succeeded": "1",
                "operations_failed": "0",
                "bytes_written": "1024",
            },
        ],
    )

    plot = plot_mutation(tmp_path)
    report = generate_report(tmp_path)

    assert plot.name == "mutation_format_compare.png"
    assert plot.exists()
    text = report.read_text(encoding="utf-8")
    assert "| format | changed files | rows | mean op, ms" in text
    assert "mutation_format_compare.png" in text
