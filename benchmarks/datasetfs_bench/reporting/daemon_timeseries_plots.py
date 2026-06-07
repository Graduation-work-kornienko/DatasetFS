"""Plots for daemon_timeseries.csv produced by benchmark runners."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _read_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def _series_key(row: dict) -> str:
    if row.get("loader"):
        seed = row.get("seed", "")
        return f"{row['loader']} seed={seed}" if seed else row["loader"]
    if row.get("name"):
        modality = row.get("modality", "")
        return f"{row['name']} ({modality})" if modality else row["name"]
    return "daemon"


def _group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(_series_key(row), []).append(row)
    for values in grouped.values():
        values.sort(key=lambda r: _f(r, "t"))
    return grouped


def _counter_rate(rows: list[dict], key: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    prev_t: float | None = None
    prev_v: float | None = None
    for row in rows:
        if key not in row or row.get(key, "") == "":
            continue
        t = _f(row, "t")
        v = _f(row, key)
        if prev_t is not None and prev_v is not None:
            dt = max(t - prev_t, 1e-9)
            xs.append(t)
            ys.append(max(0.0, v - prev_v) / dt)
        prev_t = t
        prev_v = v
    return xs, ys


def _plot_lines(ax, grouped: dict[str, list[dict]], metric: str, title: str, ylabel: str) -> bool:
    plotted = False
    for name, rows in grouped.items():
        points = [(row, _f(row, metric)) for row in rows if metric in row and row.get(metric, "") != ""]
        if not points:
            continue
        ax.plot([_f(row, "t") for row, _ in points], [value for _, value in points], linewidth=1.8, label=name)
        plotted = True
    ax.set_title(title)
    ax.set_xlabel("benchmark time, seconds")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return plotted


def _plot_rates(ax, grouped: dict[str, list[dict]], metric: str, title: str, ylabel: str) -> bool:
    plotted = False
    for name, rows in grouped.items():
        xs, ys = _counter_rate(rows, metric)
        if not xs:
            continue
        ax.plot(xs, ys, linewidth=1.8, label=name)
        plotted = True
    ax.set_title(title)
    ax.set_xlabel("benchmark time, seconds")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return plotted


def plot_daemon_timeseries(run_dir: Path, out_path: Path | None = None) -> Path:
    ts_path = run_dir / "daemon_timeseries.csv"
    if not ts_path.exists():
        raise FileNotFoundError(ts_path)
    rows = _read_rows(ts_path)
    if not rows:
        raise ValueError(f"no rows in {ts_path}")
    grouped = _group_rows(rows)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    plotted = [
        _plot_lines(axes[0][0], grouped, "gauge_active_pipelines", "Active Pipelines", "count"),
        _plot_rates(axes[0][1], grouped, "counter_load_requests_total", "Load Requests", "requests/sec"),
        _plot_rates(axes[1][0], grouped, "counter_shm_write_bytes_total", "SHM Write Throughput", "bytes/sec"),
        _plot_lines(axes[1][1], grouped, "hist_load_latency_p95_seconds", "Load Latency p95", "seconds"),
    ]
    if not any(plotted):
        raise ValueError(f"no known daemon metrics found in {ts_path}")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if not handles:
        for ax in axes.flat:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
    if handles:
        fig.legend(handles, labels, loc="upper center", ncols=min(3, len(labels)))
    fig.suptitle("DatasetFS Daemon Metrics Timeline", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    out = out_path or (run_dir / "daemon_timeseries.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[daemon_timeseries] wrote {out}", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    try:
        plot_daemon_timeseries(args.run_dir, args.output)
    except (FileNotFoundError, ValueError) as e:
        if not args.allow_missing:
            raise
        print(f"[daemon_timeseries] skip: {e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
