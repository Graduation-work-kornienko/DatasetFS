"""Plots for synthetic pipeline memory benchmark."""
from __future__ import annotations

import argparse
import csv
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


def plot_pipeline_memory(run_dir: Path, out_path: Path | None = None) -> Path:
    path = run_dir / "memory_timeseries.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = _read_rows(path)
    if not rows:
        raise ValueError(f"no rows in {path}")

    by_mode: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("scenario") == "pipeline_memory":
            by_mode[row.get("mode", "mode")].append(row)
    if not by_mode:
        raise ValueError(f"no pipeline_memory rows in {path}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for mode, vals in sorted(by_mode.items()):
        vals = sorted(vals, key=lambda r: _f(r, "cycle"))
        xs = [_f(r, "cycle") for r in vals]
        axes[0].plot(xs, [_f(r, "daemon_rss_mib") for r in vals], marker="o", markersize=3, linewidth=1.8, label=mode)
        axes[1].plot(xs, [_f(r, "drain_elapsed_s") * 1000.0 for r in vals], marker="o", markersize=3, linewidth=1.8, label=mode)

    axes[0].set_title("Daemon RSS Across Pipeline Restarts")
    axes[0].set_xlabel("cycle")
    axes[0].set_ylabel("daemon RSS, MiB")
    axes[1].set_title("Session Drain Latency")
    axes[1].set_xlabel("cycle")
    axes[1].set_ylabel("drain elapsed, ms")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Pipeline Memory Stability", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = out_path or (run_dir / "pipeline_memory.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[pipeline-memory-plots] wrote {out}", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    try:
        plot_pipeline_memory(args.run_dir, args.output)
    except (FileNotFoundError, ValueError) as e:
        if not args.allow_missing:
            raise
        print(f"[pipeline-memory-plots] skip: {e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
