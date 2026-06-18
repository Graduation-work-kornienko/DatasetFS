"""Plots for the concurrent mutation benchmark."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
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


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _series(rows: list[dict], metric: str) -> dict[str, list[tuple[float, float]]]:
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["mode"], _f(row, "mutation_rate_s"))].append(_f(row, metric))
    out: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (mode, rate), vals in grouped.items():
        out[mode].append((rate, _mean(vals)))
    return {mode: sorted(points) for mode, points in out.items()}


def _plot_metric(ax, rows: list[dict], metric: str, title: str, ylabel: str) -> None:
    for mode, points in _series(rows, metric).items():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", linewidth=2, label=mode)
    ax.set_title(title)
    ax.set_xlabel("mutations/sec")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def plot_mutation(run_dir: Path, out_path: Path | None = None) -> Path:
    rows = _read_rows(run_dir / "summary.csv")
    if not rows:
        raise ValueError(f"no rows in {run_dir / 'summary.csv'}")

    if "vacuum_scenario" in rows[0]:
        return plot_vacuum_matrix(run_dir, out_path)
    if rows[0].get("scenario") == "image_endurance":
        return plot_image_endurance(run_dir, out_path)
    if rows[0].get("scenario") == "format_mutation":
        return plot_format_mutation(run_dir, out_path)

    # Convert byte metrics to MiB to keep axes readable.
    for row in rows:
        row["tracked_rss_max_mib"] = _f(row, "tracked_rss_max_bytes") / (1024 * 1024)
        row["disk_read_mib"] = _f(row, "disk_read_bytes") / (1024 * 1024)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    _plot_metric(axes[0][0], rows, "samples_per_s", "Training Throughput", "samples/sec")
    _plot_metric(axes[0][1], rows, "cpu_pct_mean", "CPU Usage", "system CPU %")
    _plot_metric(axes[1][0], rows, "tracked_rss_max_mib", "Tracked RSS", "MiB")
    _plot_metric(axes[1][1], rows, "consistency_violations", "Consistency Violations", "count")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncols=min(5, len(labels)))
    fig.suptitle("DatasetFS Training Under Concurrent FUSE Mutations", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = out_path or (run_dir / "mutation_benchmark.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_format_mutation(run_dir: Path, out_path: Path | None = None) -> Path:
    rows = _read_rows(run_dir / "summary.csv")
    selected_changed: set[float] | None = None
    if "mutation_audio_speech_commands" in run_dir.name:
        selected_changed = {500, 1000, 2500, 6000, 10000}

    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in rows:
        changed = _f(row, "changed_files")
        if selected_changed is not None and changed not in selected_changed:
            continue
        key = (row.get("format", "format"), changed)
        grouped[key].append(_f(row, "mean_operation_ms"))

    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (fmt, changed), values in grouped.items():
        series[fmt].append((changed, _mean(values)))

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = {
        "datasetfs": "DatasetFS",
        "datasetfs_tx": "DatasetFS batch",
        "imagefolder": "ImageFolder",
        "webdataset": "WebDataset",
    }
    order = ["imagefolder", "datasetfs_tx", "datasetfs", "webdataset"]
    for fmt in [f for f in order if f in series] + sorted(f for f in series if f not in order):
        points = series[fmt]
        points = sorted(points)
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", linewidth=2, label=labels.get(fmt, fmt))
    ax.set_xlabel("Число изменяемых объектов")
    ax.set_ylabel("Среднее время изменения, мс/объект")
    if "mutation_audio_speech_commands" in run_dir.name:
        ax.set_title("Стоимость изменения объектов")
        ax.set_xlim(left=0)
        ax.set_xticks(sorted(selected_changed or []))
    else:
        ax.set_title("Стоимость изменения объектов без обучения")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = out_path or (run_dir / "mutation_format_compare.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_vacuum_matrix(run_dir: Path, out_path: Path | None = None) -> Path:
    rows = _read_rows(run_dir / "summary.csv")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row.get("vacuum_scenario", "scenario")].append(row)

    preferred = ["binary_wal_no_vacuum", "binary_wal_with_vacuum", "json_wal_no_vacuum", "json_wal_with_vacuum"]
    scenarios = [s for s in preferred if s in grouped] + sorted(s for s in grouped if s not in preferred)

    def vals(metric: str, scale: float = 1.0) -> list[float]:
        return [_mean([_f(r, metric) / scale for r in grouped[s] if r.get(metric) != ""]) for s in scenarios]

    throughput = vals("steady_samples_per_second")
    throughput_by_scenario = dict(zip(scenarios, throughput))

    def baseline_for(scenario: str) -> float:
        if scenario.startswith("json_") and "json_wal_no_vacuum" in throughput_by_scenario:
            return throughput_by_scenario["json_wal_no_vacuum"]
        if "binary_wal_no_vacuum" in throughput_by_scenario:
            return throughput_by_scenario["binary_wal_no_vacuum"]
        return max(throughput)

    throughput_delta = [(v / baseline_for(s) - 1.0) * 100.0 if baseline_for(s) else 0.0 for s, v in zip(scenarios, throughput)]
    disk_write = vals("disk_write_bytes", 1024 * 1024)
    latency = vals("mutation_latency_mean_ms")

    y = list(range(len(scenarios)))
    fig, axes = plt.subplots(1, 3, figsize=(14, max(4.5, 0.55 * len(scenarios) + 2.0)), sharey=True)
    panels = [
        (axes[0], throughput_delta, "Throughput overhead", "change vs same-WAL/no vacuum, %", "#3a78c0"),
        (axes[1], disk_write, "Disk writes", "MiB", "#2f855a"),
        (axes[2], latency, "Mutation latency", "mean ms/op", "#c0563a"),
    ]
    for ax, values, title, xlabel, color in panels:
        bars = ax.barh(y, values, color=color, edgecolor="black", linewidth=0.5)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(True, axis="x", alpha=0.3)
        if title == "Throughput overhead":
            ax.axvline(0.0, color="black", linewidth=0.8)
        for bar, value in zip(bars, values):
            ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.1f}", va="center", fontsize=8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(scenarios)
    fig.suptitle("Vacuum/Compaction: Cost Summary", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = out_path or (run_dir / "vacuum_compaction.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _plot_events(ax, events: list[dict]) -> None:
    for ev in events:
        start = _f(ev, "start_s")
        ax.axvline(start, color="black", alpha=0.18, linewidth=1)


def plot_image_endurance(run_dir: Path, out_path: Path | None = None) -> Path:
    rows = _read_rows(run_dir / "summary.csv")
    events_path = run_dir / "train_events.csv"
    ts_path = run_dir / "system_timeseries.csv"
    events = _read_rows(events_path) if events_path.exists() else []
    ts = _read_rows(ts_path) if ts_path.exists() else []

    runs = [_f(r, "train_run") for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0][0].plot(runs, [_f(r, "steady_samples_per_second") for r in rows], marker="o")
    axes[0][0].set_title("Steady Training Throughput")
    axes[0][0].set_ylabel("samples/sec")
    axes[0][1].plot(runs, [_f(r, "time_to_first_batch") for r in rows], marker="o", color="#c0563a")
    axes[0][1].set_title("Time To First Batch")
    axes[0][1].set_ylabel("seconds")
    axes[1][0].plot(runs, [_f(r, "stall_fraction") * 100.0 for r in rows], marker="o", color="#6a4c93")
    axes[1][0].set_title("Loader Stall Fraction")
    axes[1][0].set_ylabel("%")
    axes[1][1].plot(runs, [_f(r, "live_samples_after") for r in rows], marker="o", label="live samples")
    axes[1][1].bar(runs, [_f(r, "mutations_succeeded") for r in rows], alpha=0.35, label="mutations")
    axes[1][1].set_title("Dataset Changes Per Training")
    axes[1][1].legend()
    for ax in axes.flat:
        ax.set_xlabel("training run")
        ax.grid(True, alpha=0.3)
    fig.suptitle("DatasetFS Image Training Under Repeated Fixed FUSE Mutations", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = out_path or (run_dir / "mutation_endurance.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)

    if ts:
        fig2, ax1 = plt.subplots(figsize=(13, 5))
        xs = [_f(r, "t") for r in ts]
        cpu = [_f(r, "cpu_percent") for r in ts]
        rss = [_f(r, "tracked_rss_bytes") / (1024 * 1024) for r in ts]
        ax1.plot(xs, cpu, color="#3a78c0", label="CPU %")
        ax1.set_xlabel("benchmark time, seconds")
        ax1.set_ylabel("CPU %", color="#3a78c0")
        ax2 = ax1.twinx()
        ax2.plot(xs, rss, color="#c0563a", label="tracked RSS MiB")
        ax2.set_ylabel("tracked RSS MiB", color="#c0563a")
        _plot_events(ax1, events)
        ax1.grid(True, alpha=0.25)
        ax1.set_title("Resource Timeline; Vertical Lines = Training Start")
        fig2.tight_layout()
        fig2.savefig(run_dir / "mutation_endurance_timeline.png", dpi=160)
        plt.close(fig2)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = plot_mutation(args.run_dir, args.output)
    print(out)


if __name__ == "__main__":
    main()
