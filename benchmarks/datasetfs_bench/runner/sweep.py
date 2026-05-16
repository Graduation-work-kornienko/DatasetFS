"""Sweep runner — iterate single_run over a Cartesian product of axes.

Config format extends the single-run YAML with a `matrix:` block specifying
which fields to vary:

    matrix:
      num_workers: [0, 1, 2, 4, 8]

    # ... everything else fixed (same shape as mvp.yaml)
    loaders: [imagefolder, webdataset, datasetfs]
    seeds: [0, 1, 2]
    batch_size: 64
    ...

The sweep visits 5 × 3 × 3 = 45 cells (axes × loaders × seeds), each writing
its own per-cell `summary.csv` into a subdirectory. A combined
`sweep_summary.csv` at the top-level adds `axis_*` columns for each
matrix dimension — that's the file plot code reads.

Usage:
    python -m benchmarks.datasetfs_bench.runner.sweep \
        --config benchmarks/datasetfs_bench/configs/workers_sweep.yaml \
        --output runs/workers_$(date +%Y%m%dT%H%M%S)
"""
from __future__ import annotations

import argparse
import csv
import itertools
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

# Make `benchmarks.datasetfs_bench` importable when run as a module.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.datasetfs_bench.runner.single_run import run_config


def _write_combined(out_dir: Path, rows: list[dict]) -> None:
    csv_path = out_dir / "sweep_summary.csv"
    if not rows:
        return
    # Union of keys across all rows. Different axes contribute different fields.
    fieldnames: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[sweep] wrote {csv_path}", flush=True)


def _cell_subdir_name(axes: list[str], combo: tuple) -> str:
    """Filesystem-safe name encoding the axis values for this cell."""
    parts = []
    for axis, val in zip(axes, combo):
        v = str(val).replace("/", "_").replace(" ", "_")
        parts.append(f"{axis}={v}")
    return "_".join(parts) or "default"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    matrix = base_cfg.pop("matrix", None)
    if not matrix:
        raise ValueError(
            f"{args.config}: sweep config must contain a `matrix:` block"
        )

    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_dir / "config.yaml")

    axes = list(matrix.keys())
    value_lists = [matrix[a] for a in axes]
    combos = list(itertools.product(*value_lists))

    print(
        f"[sweep] {len(combos)} cells over axes={axes}; "
        f"writing into {out_dir}",
        flush=True,
    )

    all_rows: list[dict] = []
    for i, combo in enumerate(combos):
        # Build the effective per-cell config by overriding the matrix axes.
        cfg = dict(base_cfg)
        for axis, val in zip(axes, combo):
            cfg[axis] = val

        sub = out_dir / _cell_subdir_name(axes, combo)
        print(
            f"\n[sweep] === cell {i+1}/{len(combos)} {dict(zip(axes, combo))} ===",
            flush=True,
        )
        rows = run_config(cfg, sub)
        # Tag every row with the axis values so the combined CSV is plottable.
        for r in rows:
            for axis, val in zip(axes, combo):
                r[f"axis_{axis}"] = val
        all_rows.extend(rows)
        _write_combined(out_dir, all_rows)  # incremental persistence

    print(f"\n[sweep] DONE. {len(combos)} cells in {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
