"""Plots for real_universal.py benchmark runs."""
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


def _ok_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("status", "ok") == "ok"]


def plot_real_universal(run_dir: Path, out_path: Path | None = None) -> Path:
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    rows = _ok_rows(_read_rows(summary_path))
    if not rows:
        raise ValueError(f"no successful rows in {summary_path}")

    names = [r.get("name", f"dataset-{i}") for i, r in enumerate(rows)]
    modalities = [r.get("modality", "") for r in rows]
    labels = [f"{n}\n{m}" if m else n for n, m in zip(names, modalities)]

    throughput = [_f(r, "samples_per_s") for r in rows]
    cpu = [_f(r, "sys_cpu_pct_mean") for r in rows]
    daemon_rss = [_f(r, "sys_daemon_rss_max_bytes") / (1024 * 1024) for r in rows]
    total_gib = [_f(r, "dataset_total_bytes") / (1024 ** 3) for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(max(10, 1.4 * len(rows)), 8))
    x = list(range(len(rows)))

    axes[0][0].bar(x, throughput, color="#3a78c0", edgecolor="black", linewidth=0.5)
    axes[0][0].set_title("Training Throughput")
    axes[0][0].set_ylabel("samples/sec")

    axes[0][1].bar(x, cpu, color="#c0563a", edgecolor="black", linewidth=0.5)
    axes[0][1].set_title("System CPU")
    axes[0][1].set_ylabel("CPU %")

    axes[1][0].bar(x, daemon_rss, color="#6a4c93", edgecolor="black", linewidth=0.5)
    axes[1][0].set_title("Daemon RSS")
    axes[1][0].set_ylabel("MiB")

    axes[1][1].bar(x, total_gib, color="#5f8f3f", edgecolor="black", linewidth=0.5)
    axes[1][1].set_title("Dataset Logical Size")
    axes[1][1].set_ylabel("GiB")

    for ax in axes.flat:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle("DatasetFS Real-Dataset Universality Probe", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = out_path or (run_dir / "real_universal.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[real-universal-plots] wrote {out}", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    try:
        plot_real_universal(args.run_dir, args.output)
    except (FileNotFoundError, ValueError) as e:
        if not args.allow_missing:
            raise
        print(f"[real-universal-plots] skip: {e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
