"""Plots for pure-read small-file benchmark."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ORDER = ["imagefolder", "webdataset", "datasetfs"]
COLORS = {
    "imagefolder": "#6b7280",
    "webdataset": "#2563eb",
    "datasetfs": "#dc2626",
}
XTICKS = [1_000, 10_000, 100_000, 1_000_000]


def _rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str) -> float:
    return float(row.get(key) or 0.0)


def _plot_metric(run_dir: Path, metric: str, ylabel: str, filename: str, title: str) -> Path:
    rows = _rows(run_dir / "summary.csv")
    by_loader: dict[str, dict[int, float]] = {loader: {} for loader in ORDER}
    avg_sizes: list[float] = []
    for row in rows:
        loader = row["loader"]
        if loader not in by_loader:
            continue
        n = int(float(row["sample_count"]))
        by_loader[loader][n] = _f(row, metric)
        avg_sizes.append(_f(row, "avg_file_bytes") / 1024)

    counts = sorted({n for values in by_loader.values() for n in values})
    if not counts:
        raise ValueError("no rows to plot")

    fig, ax = plt.subplots(figsize=(10, 6))
    for loader in ORDER:
        points = by_loader[loader]
        xs = [n for n in counts if n in points]
        ys = [points[n] for n in xs]
        ax.plot(xs, ys, marker="o", linewidth=2.2, label=loader, color=COLORS[loader])

    subtitle = ""
    if avg_sizes:
        subtitle = f"\navg object size: {min(avg_sizes):.1f}-{max(avg_sizes):.1f} KiB; cache dropped before every cell"
    ax.set_title(title + subtitle)
    ax.set_xlabel("number of logical objects")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.set_xticks(XTICKS)
    ax.set_xticklabels(["1k", "10k", "100k", "1M"])
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(title="format")
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = run_dir / filename
    fig.savefig(out, dpi=140)
    print(f"[pure-read-plot] wrote {out}", flush=True)
    return out


def plot(run_dir: Path) -> list[Path]:
    return [
        _plot_metric(
            run_dir,
            "read_wall_seconds",
            "wall-clock time to read raw object bytes, s",
            "pure_read_wall_time.png",
            "Small-file pure read: wall time vs object count",
        ),
        _plot_metric(
            run_dir,
            "objects_per_second",
            "objects read per second",
            "pure_read_objects_per_second.png",
            "Small-file pure read: object throughput",
        ),
        _plot_metric(
            run_dir,
            "mib_per_second",
            "MiB read per second",
            "pure_read_mib_per_second.png",
            "Small-file pure read: byte throughput",
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    plot(args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
