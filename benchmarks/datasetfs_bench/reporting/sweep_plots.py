"""Plot a sweep run: line plot of throughput vs one axis (e.g. num_workers),
one line per loader, error band from seeds.

Reads `<run>/sweep_summary.csv`. Auto-detects the swept axis from `axis_*`
columns. Writes:
  - <run>/sweep_throughput.png
  - <run>/sweep_stall.png   (stall fraction vs axis — helpful for diagnosing)
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


def _read_rows(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def _find_axis_column(rows: list[dict]) -> str:
    """Pick the first axis_* column with >1 unique value."""
    if not rows:
        raise ValueError("empty sweep_summary.csv")
    axis_cols = [k for k in rows[0].keys() if k.startswith("axis_")]
    for col in axis_cols:
        vals = {r[col] for r in rows}
        if len(vals) > 1:
            return col
    if axis_cols:
        return axis_cols[0]
    raise ValueError("no axis_* columns found")


def _aggregate(rows: list[dict], axis_col: str, metric: str) -> dict[str, dict[float, tuple[float, float]]]:
    """Group rows by (loader, axis_value); compute mean ± stddev over seeds.

    Returns {loader: {axis_value: (mean, std)}}.
    Filters out warmup epochs.
    """
    by_loader: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("warmup", "").lower() == "true":
            continue
        loader = r["loader"]
        try:
            axis_val = float(r[axis_col])
            metric_val = float(r[metric])
        except (ValueError, KeyError):
            continue
        by_loader[loader][axis_val].append(metric_val)

    out: dict[str, dict[float, tuple[float, float]]] = {}
    for loader, by_axis in by_loader.items():
        out[loader] = {}
        for ax, vs in by_axis.items():
            m = mean(vs)
            s = stdev(vs) if len(vs) >= 2 else 0.0
            out[loader][ax] = (m, s)
    return out


def _plot_metric_vs_axis(
    agg: dict[str, dict[float, tuple[float, float]]],
    axis_label: str,
    metric_label: str,
    title: str,
    out_path: Path,
    log_x: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"imagefolder": "#888", "webdataset": "#3a78c0", "datasetfs": "#c0563a"}
    for loader in sorted(agg.keys()):
        xs = sorted(agg[loader].keys())
        ys = [agg[loader][x][0] for x in xs]
        errs = [agg[loader][x][1] for x in xs]
        ax.errorbar(
            xs, ys, yerr=errs, label=loader, marker="o", capsize=4,
            color=colors.get(loader, None), linewidth=1.5,
        )
    ax.set_xlabel(axis_label)
    ax.set_ylabel(metric_label)
    ax.set_title(title)
    ax.grid(linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    if log_x:
        ax.set_xscale("log", base=2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"[sweep-plots] wrote {out_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    rows = _read_rows(args.run_dir / "sweep_summary.csv")
    axis_col = _find_axis_column(rows)
    axis_label = axis_col.removeprefix("axis_")
    log_x = axis_label in ("batch_size",)  # log scale makes sense for power-of-2 axes

    throughput = _aggregate(rows, axis_col, "samples_per_second")
    _plot_metric_vs_axis(
        throughput, axis_label, "samples / sec",
        f"Throughput vs {axis_label}",
        args.run_dir / "sweep_throughput.png",
        log_x=log_x,
    )

    stall = _aggregate(rows, axis_col, "stall_fraction")
    _plot_metric_vs_axis(
        stall, axis_label, "stall fraction",
        f"DataLoader stall vs {axis_label}",
        args.run_dir / "sweep_stall.png",
        log_x=log_x,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
