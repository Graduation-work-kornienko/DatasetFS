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


def _aggregate(rows: list[dict], metric: str) -> dict[str, tuple[float, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("warmup", "").lower() == "true":
            continue
        if metric not in row or row.get(metric, "") == "":
            continue
        grouped[row.get("loader", "loader")].append(_f(row, metric))
    return {
        name: (mean(vals), stdev(vals) if len(vals) > 1 else 0.0)
        for name, vals in grouped.items()
        if vals
    }


def _plot_bar(ax, agg: dict[str, tuple[float, float]], title: str, ylabel: str) -> bool:
    if not agg:
        ax.set_visible(False)
        return False
    names = sorted(agg)
    means = [agg[n][0] for n in names]
    errs = [agg[n][1] for n in names]
    ax.bar(names, means, yerr=errs, capsize=5, color="#3a78c0", edgecolor="black", linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=20)
    return True


def plot_remote(run_dir: Path, out_path: Path | None = None) -> Path:
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    rows = _read_rows(summary_path)
    if not rows:
        raise ValueError(f"no rows in {summary_path}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plotted = False
    plotted |= _plot_bar(
        axes[0][0],
        _aggregate(rows, "steady_samples_per_second") or _aggregate(rows, "samples_per_second"),
        "Remote Throughput",
        "samples/sec",
    )
    plotted |= _plot_bar(axes[0][1], _aggregate(rows, "time_to_first_batch"), "Time To First Batch", "seconds")
    plotted |= _plot_bar(
        axes[1][0],
        _aggregate(rows, "daemon_remote_shard_wait_latency_p95"),
        "DatasetFS Remote Wait p95",
        "seconds",
    )

    hit_miss: dict[str, tuple[float, float]] = {}
    for loader in sorted({r.get("loader", "") for r in rows}):
        vals = [r for r in rows if r.get("loader") == loader and r.get("warmup", "").lower() != "true"]
        if not vals:
            continue
        hits = sum(_f(r, "daemon_remote_cache_hits_total_delta") for r in vals)
        misses = sum(_f(r, "daemon_remote_cache_misses_total_delta") for r in vals)
        total = hits + misses
        if total > 0:
            hit_miss[loader] = (hits / total * 100.0, 0.0)
    plotted |= _plot_bar(axes[1][1], hit_miss, "DatasetFS Remote Cache Hit Rate", "%")

    if not plotted:
        raise ValueError(f"no known remote metrics found in {summary_path}")

    fig.suptitle("Remote Streaming Benchmark", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = out_path or (run_dir / "remote_streaming.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[remote-plots] wrote {out}", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    try:
        plot_remote(args.run_dir, args.output)
    except (FileNotFoundError, ValueError) as e:
        if not args.allow_missing:
            raise
        print(f"[remote-plots] skip: {e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
