#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diploma" / "figures"
SNAPSHOT_RUN_CANDIDATES = [
    ROOT / "runs" / "mutation_snapshot_compare",
    ROOT / "runs" / "mutation_imagewoof_snapshot_compare",
    ROOT / "runs" / "mutation_imagewoof_20260609T194249",
]


def read_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def mean_by(rows, key, value, where=lambda row: True):
    vals = defaultdict(list)
    for row in rows:
        if where(row):
            vals[row[key]].append(float(row[value]))
    return {k: sum(v) / len(v) for k, v in vals.items() if v}


def save_bar(path, labels, values, title, ylabel, color="#4C78A8", ylim_pad=1.15, value_fmt="{:.1f}"):
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=180)
    bars = ax.bar(labels, values, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(values) * ylim_pad)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            value_fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.autofmt_xdate(rotation=18, ha="right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_grouped_bar(path, groups, series, title, ylabel, ylim_pad=1.15, value_fmt="{:.0f}"):
    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=180)
    x = np.arange(len(groups))
    width = min(0.8 / max(len(series), 1), 0.18)
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * width
    colors = ["#5B8FF9", "#61DDAA", "#65789B", "#F6BD16", "#7262FD", "#F6903D"]

    max_value = 0
    for idx, (name, values) in enumerate(series.items()):
        max_value = max(max_value, max(values))
        bars = ax.bar(x + offsets[idx], values, width, label=name, color=colors[idx % len(colors)])
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                value_fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max_value * ylim_pad)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def image_format_plot():
    rows = read_rows(ROOT / "runs" / "fmt_images_20260609T174653" / "imagenette" / "summary.csv")
    means = mean_by(rows, "loader", "steady_samples_per_second", lambda r: r["warmup"] == "False")
    decode_rows = read_rows(ROOT / "runs" / "decode_compare_simplecnn_20260609T193152" / "sweep_summary.csv")
    decode_means = mean_by(decode_rows, "axis_dfs_decode_mode", "steady_samples_per_second", lambda r: r["warmup"] == "False")

    labels = ["ImageFolder", "WebDataset", "LMDB", "HDF5", "HF Dataset", "DatasetFS", "DatasetFS-rgb"]
    values = [
        means["imagefolder"],
        means["webdataset"],
        means["lmdb"],
        means["hdf5"],
        means["huggingface"],
        means["datasetfs"],
        decode_means["rgb_uint8"],
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=180)
    y = np.arange(len(labels))
    colors = ["#5B8FF9" if label != "DatasetFS" else "#D62728" for label in labels]
    colors[-1] = "#2CA02C"
    bars = ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title("Пропускная способность загрузки изображений")
    ax.set_xlabel("Образцов в секунду")
    ax.set_xlim(min(values) - 15, max(values) + 15)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(value + 1.2, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "image_imagenette_throughput_ru.png")
    plt.close(fig)


def audio_format_plot():
    rows = read_rows(ROOT / "runs" / "fmt_audio_20260609T184047" / "summary.csv")
    means = mean_by(rows, "loader", "steady_samples_per_second", lambda r: r["warmup"] == "False")
    labels = ["ImageFolder", "WebDataset", "LMDB", "HDF5", "TFRecord", "DatasetFS"]
    values = [means["imagefolder"], means["webdataset"], means["lmdb"], means["hdf5"], means["tfrecord"], means["datasetfs"]]
    save_bar(
        OUT / "audio_formats_ru.png",
        labels,
        values,
        "Аудиоданные: пропускная способность форматов",
        "Образцов в секунду, steady-state",
        color="#4C78A8",
    )


def small_files_plot():
    rows = read_rows(ROOT / "runs" / "small_files_augmented_combined_20260613T165203" / "summary_combined.csv")
    order = ["imagefolder", "webdataset", "datasetfs"]
    colors = {
        "imagefolder": "#6b7280",
        "webdataset": "#2563eb",
        "datasetfs": "#dc2626",
    }

    def points(loader, predicate):
        values = {}
        for row in rows:
            n = int(float(row["sample_count"]))
            if row["loader"] == loader and predicate(n):
                values[n] = float(row["objects_per_second"])
        xs = sorted(values)
        return xs, [values[x] for x in xs]

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for loader in order:
        xs, ys = points(loader, lambda n: n <= 1_000_000)
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2.2, color=colors[loader], label=loader)

    xs, ys = points("datasetfs", lambda n: n > 1_000_000)
    if xs:
        ax.plot(
            xs,
            ys,
            marker="D",
            linestyle="--",
            linewidth=2.0,
            color=colors["datasetfs"],
            label="datasetfs extra",
        )

    ax.set_title("Чтение большого числа малых объектов")
    ax.set_xlabel("Число логических объектов")
    ax.set_ylabel("Объектов в секунду")
    ax.set_xscale("log")
    ax.set_yscale("log")
    xticks = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000]
    ax.set_xticks(xticks)
    ax.set_xticklabels(["1k", "2k", "5k", "10k", "20k", "50k", "100k", "200k", "500k", "1M", "1.5M", "2M", "2.5M", "3M"], rotation=35, ha="right")
    ax.grid(True, which="both", linestyle=":", alpha=0.45)
    ax.legend(title="Формат")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "small_files_objects_ru.png", dpi=160)
    plt.close(fig)


def mutation_plot():
    rows = read_rows(ROOT / "runs" / "mutation_format_compare_20260609T185921" / "summary.csv")
    means = mean_by(rows, "format", "mean_operation_ms", lambda r: r["changed_files"] == "1")
    labels = ["Файловая система", "DatasetFS", "WebDataset"]
    values = [means["imagefolder"], means["datasetfs"], means["webdataset"]]
    save_bar(
        OUT / "mutation_one_file_ru.png",
        labels,
        values,
        "Стоимость изменения одного объекта",
        "Миллисекунд на операцию",
        color="#F58518",
    )


def vacuum_plot():
    rows = read_rows(ROOT / "runs" / "vacuum_compaction_20260613T173115" / "summary.csv")
    wanted = ["binary_wal_no_vacuum", "binary_wal_with_vacuum", "json_wal_no_vacuum", "json_wal_with_vacuum"]
    label_map = {
        "binary_wal_no_vacuum": "Binary WAL, без vacuum",
        "binary_wal_with_vacuum": "Binary WAL, vacuum",
        "json_wal_no_vacuum": "JSON WAL, без vacuum",
        "json_wal_with_vacuum": "JSON WAL, vacuum",
    }
    means = mean_by(rows, "vacuum_scenario", "steady_samples_per_second", lambda r: r["vacuum_scenario"] in set(wanted))
    available = [s for s in wanted if s in means]
    labels = [label_map[s] for s in available]
    values = [means[s] for s in available]

    def scenario_baseline(scenario: str) -> float:
        if scenario.startswith("json_") and "json_wal_no_vacuum" in means:
            return means["json_wal_no_vacuum"]
        return means["binary_wal_no_vacuum"]

    deltas = [(means[s] / scenario_baseline(s) - 1.0) * 100.0 for s in available]
    fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=180)
    y = np.arange(len(labels))
    colors = ["#6B7280" if "no_vacuum" in s else "#9467BD" for s in available]
    bars = ax.barh(y, deltas, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title("Фоновое обслуживание: изменение throughput")
    ax.set_xlabel("Изменение относительно режима без vacuum с тем же WAL, %")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_xlim(min(deltas) - 0.4, max(deltas) + 0.6)
    for bar, value in zip(bars, deltas):
        x = value - 0.05 if value < 0 else value + 0.05
        ha = "right" if value < 0 else "left"
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", ha=ha, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "vacuum_throughput_ru.png")
    plt.close(fig)

    write_means = mean_by(rows, "vacuum_scenario", "disk_write_bytes", lambda r: r["vacuum_scenario"] in set(available))
    writes = [write_means[s] / (1024 * 1024) for s in available]

    def write_baseline(scenario: str) -> float:
        if scenario.startswith("json_") and "json_wal_no_vacuum" in write_means:
            return write_means["json_wal_no_vacuum"] / (1024 * 1024)
        return write_means["binary_wal_no_vacuum"] / (1024 * 1024)

    write_deltas = [w - write_baseline(s) for s, w in zip(available, writes)]
    fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=180)
    bars = ax.barh(y, write_deltas, color=["#6B7280" if "no_vacuum" in s else "#F58518" for s in available])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title("Фоновое обслуживание: дополнительная дисковая запись")
    ax.set_xlabel("Изменение относительно режима без vacuum с тем же WAL, MiB")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    pad = max(abs(v) for v in write_deltas) * 0.18
    ax.set_xlim(min(write_deltas) - pad, max(write_deltas) + pad)
    for bar, value, total in zip(bars, write_deltas, writes):
        x = value + (12 if value >= 0 else -12)
        ha = "left" if value >= 0 else "right"
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{value:+.0f} MiB\n({total:.0f})", va="center", ha=ha, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "vacuum_disk_write_ru.png")
    plt.close(fig)


def snapshot_isolation_plot():
    run_dir = first_existing(SNAPSHOT_RUN_CANDIDATES)
    rows = read_rows(run_dir / "summary.csv")
    rows = sorted(rows, key=lambda r: (r.get("vacuum_scenario", ""), int(float(r["train_run"]))))
    ts_path = run_dir / "system_timeseries.csv"
    system_rows = read_rows(ts_path) if ts_path.exists() else []

    labels = {
        "binary_wal_no_vacuum": "без фонового vacuum",
        "binary_wal_with_vacuum": "с фоновым vacuum",
    }
    scenarios = [s for s in labels if any(r.get("vacuum_scenario") == s for r in rows)]
    if not scenarios:
        scenarios = [rows[0].get("vacuum_scenario", "single_run")]
        labels[scenarios[0]] = f"WAL={rows[0].get('wal_format', '')}, vacuum={rows[0].get('auto_vacuum', '')}"

    def f(row, key):
        try:
            return float(row.get(key, "") or 0.0)
        except ValueError:
            return 0.0

    fig = plt.figure(figsize=(10.5, 7.4), dpi=180)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0])
    ax_thr = fig.add_subplot(gs[0, :])
    ax_cpu = fig.add_subplot(gs[1, 0])
    ax_io = fig.add_subplot(gs[1, 1])
    colors = ["#5B8FF9", "#F58518", "#2CA02C"]
    max_run = 0
    total_failed = 0
    total_succeeded = 0
    total_requested = 0
    cpu_groups = ["system CPU", "daemon CPU"]
    io_groups = ["disk read", "disk write"]
    cpu_series = []
    io_series = []

    for idx, scenario in enumerate(scenarios):
        srows = [r for r in rows if r.get("vacuum_scenario", scenario) == scenario]
        srows = sorted(srows, key=lambda r: int(float(r["train_run"])))
        runs = np.array([int(float(r["train_run"])) + 1 for r in srows])
        max_run = max(max_run, int(max(runs)))
        throughput = [float(r["steady_samples_per_second"]) for r in srows]
        succeeded = [f(r, "mutations_succeeded") for r in srows]
        failed = [f(r, "mutations_failed") for r in srows]
        requested = [f(r, "mutations_requested") for r in srows]
        total_failed += int(sum(failed))
        total_succeeded += int(sum(succeeded))
        total_requested += int(sum(requested))

        label = labels.get(scenario, scenario)
        color = colors[idx % len(colors)]
        ax_thr.plot(runs, throughput, marker="o", linewidth=2.0, color=color, label=f"{label}, avg={sum(throughput) / len(throughput):.1f}")

        sts = [r for r in system_rows if r.get("vacuum_scenario") == scenario]
        if sts:
            cpu_series.append((label, color, [sum(f(r, "cpu_percent") for r in sts) / len(sts), sum(f(r, "daemon_cpu_percent") for r in sts) / len(sts)]))
            reads = [f(r, "disk_read_bytes") for r in sts]
            writes = [f(r, "disk_write_bytes") for r in sts]
            io_series.append((label, color, [(reads[-1] - reads[0]) / (1024 ** 3), (writes[-1] - writes[0]) / (1024 ** 3)]))

    ax_thr.set_title("Изменения во время обучения: snapshot isolation и фоновый vacuum")
    ax_thr.set_ylabel("Образцов/с, steady-state")
    ax_thr.set_xlabel("Номер training run")
    ax_thr.set_xticks(range(1, max_run + 1))
    ax_thr.grid(True, alpha=0.25)
    ax_thr.legend(fontsize=8)
    ax_thr.text(
        0.01,
        0.06,
        f"мутации: {total_succeeded}/{total_requested} успешных, ошибок: {total_failed}",
        transform=ax_thr.transAxes,
        fontsize=9,
    )

    x_cpu = np.arange(len(cpu_groups))
    x_io = np.arange(len(io_groups))
    width = 0.34 if len(scenarios) > 1 else 0.55
    offsets = (np.arange(max(len(cpu_series), 1)) - (max(len(cpu_series), 1) - 1) / 2) * width
    for idx, (label, color, values) in enumerate(cpu_series):
        ax_cpu.bar(x_cpu + offsets[idx], values, width=width, color=color, label=label)
        for x, value in zip(x_cpu + offsets[idx], values):
            ax_cpu.text(x, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)
    ax_cpu.set_title("Средняя CPU-нагрузка")
    ax_cpu.set_ylabel("CPU, %")
    ax_cpu.set_xticks(x_cpu)
    ax_cpu.set_xticklabels(cpu_groups)
    ax_cpu.grid(axis="y", alpha=0.25)

    for idx, (label, color, values) in enumerate(io_series):
        ax_io.bar(x_io + offsets[idx], values, width=width, color=color, label=label)
        for x, value in zip(x_io + offsets[idx], values):
            ax_io.text(x, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    ax_io.set_title("Дисковый I/O за весь прогон")
    ax_io.set_ylabel("GiB")
    ax_io.set_xticks(x_io)
    ax_io.set_xticklabels(io_groups)
    ax_io.grid(axis="y", alpha=0.25)
    if io_series:
        ax_io.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "mutation_training_snapshot_ru.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })
    image_format_plot()
    small_files_plot()
    mutation_plot()
    vacuum_plot()
    snapshot_isolation_plot()


if __name__ == "__main__":
    main()
