"""Plot small-file scaling benchmark results."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


ORDER = ["imagefolder", "webdataset", "datasetfs"]
XTICKS = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
COLORS = {
    "imagefolder": "#6b7280",
    "webdataset": "#2563eb",
    "datasetfs": "#dc2626",
}


def _rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, "") or default)
    except ValueError:
        return default


def plot(run_dir: Path, metric: str = "steady_batch_wait_total_s") -> Path:
    rows = _rows(run_dir / "summary.csv")
    values: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    avg_sizes: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("warmup", "").lower() == "true":
            continue
        n = int(float(row["sample_count"]))
        loader = row["loader"]
        if loader not in ORDER:
            continue
        values[loader][n].append(_f(row, metric))
        avg_sizes[n].append(_f(row, "avg_file_bytes"))

    counts = sorted({n for by_n in values.values() for n in by_n})
    if not counts:
        raise ValueError("no non-warmup rows with sample_count in summary.csv")

    fig, ax = plt.subplots(figsize=(10, 6))
    for loader in ORDER:
        if loader not in values:
            continue
        means = [mean(values[loader].get(n, [0.0])) for n in counts]
        errs = [stdev(values[loader][n]) if len(values[loader].get(n, [])) > 1 else 0.0 for n in counts]
        ax.errorbar(
            counts, means, yerr=errs, marker="o", linewidth=2.2, capsize=4,
            label=loader, color=COLORS[loader],
        )

    avg_kib = [mean(avg_sizes[n]) / 1024 for n in counts]
    subtitle = f"avg object size: {min(avg_kib):.1f}-{max(avg_kib):.1f} KiB"
    ax.set_xlabel("number of small files / objects")
    ax.set_ylabel("steady data-loading wait time per epoch, s")
    ax.set_title("Small-file scaling: loader wait vs object count\n" + subtitle)
    ax.set_xscale("log")
    ax.set_xticks(XTICKS)
    ax.set_xticklabels(["1k", "10k", "100k", "1M", "10M", "100M"])
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(title="format")
    ax.set_axisbelow(True)
    fig.tight_layout()

    out = run_dir / "small_files_loading_time.png"
    fig.savefig(out, dpi=140)
    print(f"[small-files-plot] wrote {out}", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--metric", default="steady_batch_wait_total_s")
    args = parser.parse_args()
    plot(args.run_dir, args.metric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
