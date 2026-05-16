"""Generate the headline bar chart from a single_run output dir.

Reads `<run>/summary.csv` and writes:
  - <run>/throughput_bar.png      (mean ± stddev across seeds, steady-state only)
  - <run>/latency_table.md        (latency percentiles per loader)
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


def _read_rows(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def _aggregate_throughput(rows: list[dict]) -> dict[str, tuple[float, float, int]]:
    """For each loader, compute mean ± stddev of samples_per_second across
    seeds, using ONLY non-warmup epochs. Returns {loader: (mean, stddev, n_seeds)}."""
    per_loader: dict[str, list[float]] = defaultdict(list)
    per_loader_seeds: dict[str, set[int]] = defaultdict(set)

    for r in rows:
        if r.get("warmup", "").lower() == "true":
            continue
        loader = r["loader"]
        sps = float(r["samples_per_second"])
        per_loader[loader].append(sps)
        per_loader_seeds[loader].add(int(r["seed"]))

    out = {}
    for loader, vals in per_loader.items():
        if len(vals) >= 2:
            out[loader] = (mean(vals), stdev(vals), len(per_loader_seeds[loader]))
        else:
            out[loader] = (vals[0] if vals else 0.0, 0.0, len(per_loader_seeds[loader]))
    return out


def _aggregate_latency(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Mean of per-epoch latency percentiles (fetch_p50/p95/p99) across non-warmup."""
    per_loader: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("warmup", "").lower() == "true":
            continue
        loader = r["loader"]
        for k in ("fetch_p50", "fetch_p95", "fetch_p99", "stall_fraction", "time_to_first_batch"):
            try:
                per_loader[loader][k].append(float(r[k]))
            except (ValueError, KeyError):
                pass
    out: dict[str, dict[str, float]] = {}
    for loader, by_metric in per_loader.items():
        out[loader] = {k: (mean(vs) if vs else float("nan")) for k, vs in by_metric.items()}
    return out


def plot_throughput_bar(run_dir: Path) -> Path:
    rows = _read_rows(run_dir / "summary.csv")
    agg = _aggregate_throughput(rows)
    if not agg:
        raise ValueError("no non-warmup rows in summary.csv")

    # Preserve config order if config.yaml present; else alphabetical.
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            order = yaml.safe_load(f).get("loaders", sorted(agg.keys()))
    else:
        order = sorted(agg.keys())
    order = [l for l in order if l in agg]

    means = [agg[l][0] for l in order]
    errs = [agg[l][1] for l in order]
    n_seeds = [agg[l][2] for l in order]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        order, means, yerr=errs, capsize=8,
        color=["#888", "#3a78c0", "#c0563a"][:len(order)],
        edgecolor="black", linewidth=0.6,
    )
    for bar, m, n in zip(bars, means, n_seeds):
        ax.text(
            bar.get_x() + bar.get_width() / 2, m,
            f"{m:.0f}\nn={n}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylabel("samples / sec")
    ax.set_title("DatasetFS MVP throughput — Imagenette, ResNet-18\n(mean ± stddev across seeds, steady-state)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()

    out_path = run_dir / "throughput_bar.png"
    fig.savefig(out_path, dpi=120)
    print(f"[plots] wrote {out_path}", flush=True)
    return out_path


def write_latency_table(run_dir: Path) -> Path:
    rows = _read_rows(run_dir / "summary.csv")
    lat = _aggregate_latency(rows)
    if not lat:
        raise ValueError("no non-warmup rows")

    lines = [
        "| loader | TTFB (s) | fetch p50 (s) | fetch p95 (s) | fetch p99 (s) | stall % |",
        "|---|---|---|---|---|---|",
    ]
    for loader, m in sorted(lat.items()):
        lines.append(
            f"| {loader} "
            f"| {m.get('time_to_first_batch', float('nan')):.3f} "
            f"| {m.get('fetch_p50', float('nan')):.4f} "
            f"| {m.get('fetch_p95', float('nan')):.4f} "
            f"| {m.get('fetch_p99', float('nan')):.4f} "
            f"| {m.get('stall_fraction', float('nan'))*100:.1f} |"
        )
    out_path = run_dir / "latency_table.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[plots] wrote {out_path}", flush=True)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    plot_throughput_bar(args.run_dir)
    write_latency_table(args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
