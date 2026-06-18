"""Generic wait-vs-compute breakdown plots from benchmark summary.csv."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


def _read_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, "") or default)
    except ValueError:
        return default


def _label(row: dict) -> str:
    name = row.get("loader") or row.get("name") or row.get("mode") or row.get("scenario") or "run"
    if row.get("format"):
        name = f"{name}/{row['format']}"
    return name


def _stage_values(row: dict) -> dict[str, float] | None:
    if row.get("steady_batch_wait_fraction"):
        wait = _f(row, "steady_batch_wait_fraction")
        return {"wait": wait, "compute": max(0.0, 1.0 - wait)}
    if not (row.get("batch_wait_fraction") or row.get("stall_fraction")):
        return None
    wait = _f(row, "batch_wait_fraction", _f(row, "stall_fraction"))
    fwd_bwd = _f(row, "forward_backward_fraction")
    optimizer = _f(row, "optimizer_fraction")
    if fwd_bwd == 0.0 and optimizer == 0.0:
        compute = max(0.0, 1.0 - wait)
        return {"wait": wait, "compute": compute}
    other = max(0.0, 1.0 - wait - fwd_bwd - optimizer)
    return {"wait": wait, "forward_backward": fwd_bwd, "optimizer": optimizer, "other_compute": other}


def _aggregate(rows: list[dict]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("warmup", "").lower() == "true":
            continue
        stages = _stage_values(row)
        if stages is None:
            continue
        label = _label(row)
        for key, value in stages.items():
            grouped[label][key].append(value)
    return {
        label: {key: mean(values) if values else 0.0 for key, values in stages.items()}
        for label, stages in grouped.items()
    }


def plot_wait_compute(run_dir: Path, out_path: Path | None = None) -> Path:
    summary = run_dir / "summary.csv"
    if not summary.exists():
        raise FileNotFoundError(summary)
    rows = _read_rows(summary)
    agg = _aggregate(rows)
    if not agg:
        raise ValueError(f"no wait/compute timing columns in {summary}")

    labels = sorted(agg, key=lambda label: agg[label].get("wait", 0.0))

    height = max(4.5, min(10.0, 0.45 * len(labels) + 2.0))
    fig, ax = plt.subplots(figsize=(10, height))
    ys = list(range(len(labels)))
    vals = [agg[label].get("wait", 0.0) * 100.0 for label in labels]
    bars = ax.barh(ys, vals, color="#3a78c0", edgecolor="black", linewidth=0.35)
    for bar, value in zip(bars, vals):
        ax.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f" {value:.2f}%",
            va="center",
            ha="left",
            fontsize=8,
        )

    ax.set_yticks(ys, labels)
    ax.set_xlabel("data wait share of steady batch cycle, %")
    ax.set_title("Loader Wait Fraction (zoomed; compute is the remaining share)")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.set_xlim(0.0, max(vals) * 1.18 if vals else 1.0)
    fig.tight_layout()
    out = out_path or (run_dir / "wait_compute_breakdown.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[wait-compute] wrote {out}", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    try:
        plot_wait_compute(args.run_dir, args.output)
    except (FileNotFoundError, ValueError) as e:
        if not args.allow_missing:
            raise
        print(f"[wait-compute] skip: {e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
