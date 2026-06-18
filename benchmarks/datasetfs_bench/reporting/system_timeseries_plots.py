"""Plots for system_timeseries.csv produced by long benchmark runners."""
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


def _group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        name = row.get("vacuum_scenario") or row.get("scenario") or "run"
        grouped.setdefault(name, []).append(row)
    for values in grouped.values():
        values.sort(key=lambda r: _f(r, "t"))
        if values:
            start = _f(values[0], "t")
            for row in values:
                row["t_rel"] = _f(row, "t") - start
    return grouped


def _plot_metric(ax, rows: list[dict], key: str, label: str, scale: float = 1.0) -> bool:
    points = [(r, _f(r, key) / scale) for r in rows if key in r and r.get(key, "") != ""]
    if not points:
        return False
    ax.plot([_f(r, "t") for r, _ in points], [v for _, v in points], linewidth=1.8, label=label)
    return True


def _plot_train_events(ax, run_dir: Path) -> None:
    path = run_dir / "train_events.csv"
    if not path.exists():
        return
    try:
        events = _read_rows(path)
    except Exception:
        return
    for event in events:
        ax.axvline(_f(event, "start_s"), color="black", alpha=0.15, linewidth=1)


def plot_system_timeseries(run_dir: Path, out_path: Path | None = None) -> Path:
    ts_path = run_dir / "system_timeseries.csv"
    if not ts_path.exists():
        raise FileNotFoundError(ts_path)
    rows = _read_rows(ts_path)
    if not rows:
        raise ValueError(f"no rows in {ts_path}")

    grouped = _group_rows(rows)
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    plotted = False

    for name, values in grouped.items():
        xs = [_f(r, "t_rel") for r in values]
        if any(r.get("cpu_percent", "") != "" for r in values):
            axes[0].plot(xs, [_f(r, "cpu_percent") for r in values], linewidth=1.5, label=name)
            plotted = True
        if any(r.get("tracked_rss_bytes", "") != "" for r in values):
            axes[1].plot(xs, [_f(r, "tracked_rss_bytes") / (1024 * 1024) for r in values], linewidth=1.5, label=name)
            plotted = True
        wx, wy = _counter_rate(values, "disk_write_bytes")
        if wx:
            start = _f(values[0], "t") if values else 0.0
            axes[2].plot([x - start for x in wx], [v / (1024 * 1024) for v in wy], linewidth=1.5, label=name)
            plotted = True

    axes[0].set_title("System CPU")
    axes[0].set_ylabel("CPU %")
    axes[1].set_title("Tracked RSS")
    axes[1].set_ylabel("MiB")
    axes[2].set_title("Disk Write Rate")
    axes[2].set_ylabel("MiB/sec")

    if not plotted:
        raise ValueError(f"no known system metrics found in {ts_path}")

    for ax in axes:
        ax.set_xlabel("benchmark time, seconds")
        ax.grid(True, alpha=0.3)
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(ncols=2, fontsize=8)
    fig.suptitle("System Resource Timeline by Scenario", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = out_path or (run_dir / "system_timeseries.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[system-timeseries] wrote {out}", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    try:
        plot_system_timeseries(args.run_dir, args.output)
    except (FileNotFoundError, ValueError) as e:
        if not args.allow_missing:
            raise
        print(f"[system-timeseries] skip: {e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
