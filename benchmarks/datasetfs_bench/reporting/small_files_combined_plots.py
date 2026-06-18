"""Combine baseline small-file pure-read rows with DatasetFS-only extra points."""
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


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _points(rows: list[dict[str, str]], loader: str, metric: str) -> tuple[list[int], list[float]]:
    values: dict[int, float] = {}
    for row in rows:
        if row.get("loader") != loader:
            continue
        n = int(float(row["sample_count"]))
        values[n] = float(row[metric])
    xs = sorted(values)
    return xs, [values[x] for x in xs]


def plot(baseline: Path, extra: Path, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    baseline_rows = _read_rows(baseline / "summary.csv")
    extra_rows = _read_rows(extra / "summary.csv")
    combined_rows = baseline_rows + extra_rows
    _write_rows(output / "summary_combined.csv", combined_rows)

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for loader in ORDER:
        xs, ys = _points(baseline_rows, loader, "objects_per_second")
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2.2, color=COLORS[loader], label=loader)

    xs, ys = _points(extra_rows, "datasetfs", "objects_per_second")
    if xs:
        ax.plot(
            xs,
            ys,
            marker="D",
            linestyle="--",
            linewidth=2.0,
            color=COLORS["datasetfs"],
            label="datasetfs extra",
        )

    ax.set_title("Small-file pure read: object throughput\naugmented WAV objects, no page-cache purge")
    ax.set_xlabel("number of logical objects")
    ax.set_ylabel("objects read per second")
    ax.set_xscale("log")
    ax.set_yscale("log")
    xticks = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000]
    ax.set_xticks(xticks)
    ax.set_xticklabels(["1k", "2k", "5k", "10k", "20k", "50k", "100k", "200k", "500k", "1M", "1.5M", "2M", "2.5M", "3M"], rotation=35, ha="right")
    ax.grid(True, which="both", linestyle=":", alpha=0.45)
    ax.legend(title="format")
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = output / "pure_read_objects_per_second_combined.png"
    fig.savefig(out, dpi=160)
    print(f"[small-files-combined] wrote {out}", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--extra", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.baseline, args.extra, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
