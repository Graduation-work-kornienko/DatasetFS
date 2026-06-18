"""Plots for remote streaming benchmark runs."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


def _read_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def _aggregate(rows: list[dict], metric: str, *, include_warmup: bool = False) -> dict[str, tuple[float, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if not include_warmup and row.get("warmup", "").lower() == "true":
            continue
        if metric not in row or row.get(metric, "") == "":
            continue
        grouped[row.get("loader", "loader")].append(_f(row, metric))
    return {
        name: (mean(vals), stdev(vals) if len(vals) > 1 else 0.0)
        for name, vals in grouped.items()
        if vals
    }


def _scale_agg(agg: dict[str, tuple[float, float]], scale: float) -> dict[str, tuple[float, float]]:
    return {name: (value * scale, err * scale) for name, (value, err) in agg.items()}


def _plot_bar(ax, agg: dict[str, tuple[float, float]], title: str, ylabel: str, *, color: str = "#3a78c0") -> bool:
    if not agg:
        ax.set_visible(False)
        return False
    names = sorted(agg)
    means = [agg[n][0] for n in names]
    errs = [agg[n][1] for n in names]
    bars = ax.bar(names, means, yerr=errs, capsize=5, color=color, edgecolor="black", linewidth=0.5)
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=20)
    return True


def plot_remote(run_dir: Path, out_path: Path | None = None, *, cold_start: bool = False) -> Path:
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    rows = _read_rows(summary_path)
    if not rows:
        raise ValueError(f"no rows in {summary_path}")
    if cold_start:
        rows = [r for r in rows if str(r.get("epoch", "")) in ("0", "0.0")]
        if not rows:
            raise ValueError("no epoch=0 rows for cold-start plot")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plotted = False
    plotted |= _plot_bar(
        axes[0][0],
        _aggregate(rows, "steady_samples_per_second", include_warmup=cold_start)
        or _aggregate(rows, "samples_per_second", include_warmup=cold_start),
        "Remote Throughput",
        "samples/sec",
        color="#2f855a",
    )
    plotted |= _plot_bar(
        axes[0][1],
        _aggregate(rows, "time_to_first_batch", include_warmup=cold_start),
        "Time To First Batch",
        "seconds",
        color="#c0563a",
    )
    plotted |= _plot_bar(
        axes[1][0],
        _scale_agg(
            _aggregate(rows, "steady_batch_wait_fraction", include_warmup=cold_start)
            or _aggregate(rows, "stall_fraction", include_warmup=cold_start),
            100.0,
        ),
        "Loader Wait Fraction",
        "% of batch cycle",
        color="#6a4c93",
    )

    remote_io: dict[str, tuple[float, float]] = {}
    for loader in sorted({r.get("loader", "") for r in rows}):
        vals = [
            r for r in rows
            if r.get("loader") == loader and (cold_start or r.get("warmup", "").lower() != "true")
        ]
        if not vals:
            continue
        downloaded = sum(_f(r, "epoch_daemon_remote_bytes_downloaded_total_delta") for r in vals)
        if downloaded > 0:
            remote_io[loader] = (downloaded / (1024 * 1024), 0.0)
    if not remote_io:
        axes[1][1].text(0.5, 0.5, "no remote download deltas\nin selected rows", ha="center", va="center")
        axes[1][1].set_axis_off()
    else:
        plotted |= _plot_bar(axes[1][1], remote_io, "Remote Bytes Downloaded", "MiB", color="#dd6b20")

    if not plotted:
        raise ValueError(f"no known remote metrics found in {summary_path}")

    title = "Remote Streaming Benchmark"
    if cold_start:
        title += " (cold start, epoch 0)"
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = out_path or (run_dir / ("remote_streaming_cold.png" if cold_start else "remote_streaming.png"))
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[remote-plots] wrote {out}", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--cold-start", action="store_true", help="plot only epoch=0 rows, including warmup rows")
    args = parser.parse_args()
    try:
        plot_remote(args.run_dir, args.output, cold_start=args.cold_start)
    except (FileNotFoundError, ValueError) as e:
        if not args.allow_missing:
            raise
        print(f"[remote-plots] skip: {e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
