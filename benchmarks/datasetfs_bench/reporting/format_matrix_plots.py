"""Format-matrix bar chart (thesis graph G1): throughput per storage format.

Takes ONE or MORE single_run output dirs. One dir → a single bar per format.
Multiple dirs → grouped bars (one group per dataset), so the same formats can be
compared across datasets in a single figure (e.g. imagenette vs imagewoof).

    python -m benchmarks.datasetfs_bench.reporting.format_matrix_plots \
        runs/fmt_images_<stamp>/imagenette runs/fmt_images_<stamp>/imagewoof

Writes `format_matrix.png` into the FIRST run dir's parent (or the dir itself
for a single run).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .plots import _aggregate_throughput, _read_rows, loader_display_names


def _dataset_name(run_dir: Path) -> str:
    cfg = run_dir / "config.yaml"
    if cfg.exists():
        import yaml
        with open(cfg) as f:
            return yaml.safe_load(f).get("dataset", {}).get("name", run_dir.name)
    return run_dir.name


def _format_order(run_dir: Path, present: set[str]) -> list[str]:
    cfg = run_dir / "config.yaml"
    if cfg.exists():
        import yaml
        with open(cfg) as f:
            order = loader_display_names(yaml.safe_load(f).get("loaders", []))
        ordered = [l for l in order if l in present]
        ordered += [l for l in sorted(present) if l not in ordered]
        return ordered
    return sorted(present)


def plot_format_matrix(run_dirs: list[Path], out_path: Path | None = None) -> Path:
    # per_dataset[name] = {format: (mean, std, n)}
    per_dataset: dict[str, dict] = {}
    all_formats: set[str] = set()
    for d in run_dirs:
        agg = _aggregate_throughput(_read_rows(d / "summary.csv"))
        if not agg:
            continue
        name = _dataset_name(d)
        per_dataset[name] = agg
        all_formats.update(agg.keys())
    if not per_dataset:
        raise ValueError("no non-warmup rows in any provided run dir")

    formats = _format_order(run_dirs[0], all_formats)
    datasets = list(per_dataset.keys())

    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(formats)), 5))
    n_groups = len(datasets)
    width = 0.8 / n_groups
    x = range(len(formats))

    for gi, ds_name in enumerate(datasets):
        agg = per_dataset[ds_name]
        means = [agg.get(f, (0.0, 0.0, 0))[0] for f in formats]
        errs = [agg.get(f, (0.0, 0.0, 0))[1] for f in formats]
        offs = [xi + (gi - (n_groups - 1) / 2) * width for xi in x]
        bars = ax.bar(offs, means, width=width, yerr=errs, capsize=4,
                      label=ds_name, edgecolor="black", linewidth=0.5)
        if n_groups == 1:
            for bar, m, f in zip(bars, means, formats):
                n = agg.get(f, (0, 0, 0))[2]
                ax.text(bar.get_x() + bar.get_width() / 2, m,
                        f"{m:.0f}\nn={n}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(formats, rotation=20, ha="right")
    ax.set_ylabel("samples / sec")
    ax.set_title("Throughput by storage format (loader-bound, SimpleCNN)\n"
                 "mean ± stddev across seeds, steady-state")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    if n_groups > 1:
        ax.legend(title="dataset")
    fig.tight_layout()

    if out_path is None:
        out_path = (run_dirs[0].parent if len(run_dirs) > 1 else run_dirs[0]) / "format_matrix.png"
    fig.savefig(out_path, dpi=120)
    print(f"[format_matrix] wrote {out_path}", flush=True)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    plot_format_matrix(args.run_dirs, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
