"""End-to-end runner for one MVP benchmark config.

For each (loader, seed) cell:
  - set seeds
  - construct loader (start daemon for DFS)
  - run `epochs` training epochs, capturing EpochStats per epoch
  - tear down

Persists:
  - <out>/config.yaml         (copy of the input config)
  - <out>/host_info.json      (machine fingerprint)
  - <out>/summary.csv         (one row per epoch, all loaders/seeds)
  - <out>/daemon-*.log        (daemon stdout/stderr per DFS run)

Usage:
    python -m benchmarks.datasetfs_bench.runner.single_run \
        --config benchmarks/datasetfs_bench/configs/mvp.yaml \
        --output runs/mvp_$(date +%Y%m%dT%H%M%S)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml

# Make `clients.python` and `benchmarks.datasetfs_bench` importable.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.datasetfs_bench.loaders.datasetfs import DatasetFSLoader
from benchmarks.datasetfs_bench.loaders.imagefolder import ImageFolderLoader
from benchmarks.datasetfs_bench.loaders.webdataset_loader import WebDatasetLoader
from benchmarks.datasetfs_bench.metrics.training import EpochStats
from benchmarks.datasetfs_bench.models.registry import build_model
from benchmarks.datasetfs_bench.runner import host_info
from benchmarks.datasetfs_bench.runner.daemon_ctl import DaemonManager
from benchmarks.datasetfs_bench.train.loop import train_one_epoch


LOADER_CLASSES = {
    "datasetfs": DatasetFSLoader,
    "webdataset": WebDatasetLoader,
    "imagefolder": ImageFolderLoader,
}


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _label_to_idx(imagefolder_root: Path) -> dict[str, int]:
    classes = sorted(
        p.name for p in imagefolder_root.iterdir() if p.is_dir()
    )
    return {c: i for i, c in enumerate(classes)}


def _build_loader(loader_name: str, cfg: dict, label_to_idx: dict, seed: int):
    """Construct a loader with the right spec for its format."""
    ds = cfg["dataset"]
    common = {
        "batch_size": cfg["batch_size"],
        "num_workers": cfg["num_workers"],
        "image_size": cfg["image_size"],
        "label_to_idx": label_to_idx,
        "seed": seed,
    }
    cls = LOADER_CLASSES[loader_name]
    if loader_name == "imagefolder":
        return cls({**common, "root": ds["imagefolder"]})
    if loader_name == "webdataset":
        return cls({**common, "root": ds["webdataset"]})
    if loader_name == "datasetfs":
        return cls({**common, "root": ds["datasetfs"]})
    raise ValueError(loader_name)


def _run_one_cell(
    loader_name: str,
    cfg: dict,
    label_to_idx: dict,
    seed: int,
    out_dir: Path,
    daemon: DaemonManager | None,
) -> list[EpochStats]:
    """Run one (loader, seed) cell, returning per-epoch stats."""
    print(f"\n=== {loader_name} seed={seed} ===", flush=True)
    _set_all_seeds(seed)

    if loader_name == "datasetfs":
        assert daemon is not None, "DatasetFS run requires a daemon"
        # Fresh session per seed — avoids any state leak from previous run.
        daemon.restart()

    loader = _build_loader(loader_name, cfg, label_to_idx, seed)
    loader.setup()

    model = build_model(cfg["model"], num_classes=len(label_to_idx))
    optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    stats: list[EpochStats] = []
    try:
        for epoch in range(cfg["epochs"]):
            dl = loader.make_loader()
            it = iter(dl)
            ep_stats = train_one_epoch(
                model, it, optim, loss_fn,
                epoch_idx=epoch,
                max_batches=cfg.get("max_batches_per_epoch"),
            )
            print(
                f"  epoch={epoch} samples/sec={ep_stats.samples_per_second:.1f} "
                f"TTFB={ep_stats.time_to_first_batch:.2f}s "
                f"stall={ep_stats.stall_fraction:.2%}",
                flush=True,
            )
            stats.append(ep_stats)
            # Drop the DataLoader workers between epochs to avoid pipe state
            # bleeding across epochs.
            del it, dl
    finally:
        loader.teardown()

    return stats


def _write_summary(out_dir: Path, rows: list[dict]) -> None:
    csv_path = out_dir / "summary.csv"
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[runner] wrote {csv_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot config + host info into the run dir.
    shutil.copy(args.config, out_dir / "config.yaml")
    host_info.write(out_dir / "host_info.json")

    label_to_idx = _label_to_idx(Path(cfg["dataset"]["imagefolder"]))
    print(f"[runner] {len(label_to_idx)} classes", flush=True)

    # Spin up the daemon once if DatasetFS is in the loader list. We reuse
    # the process across seeds; each seed calls daemon.restart() for a fresh
    # session (handled inside _run_one_cell).
    daemon: DaemonManager | None = None
    if "datasetfs" in cfg["loaders"]:
        daemon = DaemonManager(
            binary=REPO_ROOT / cfg["daemon_binary"],
            root_path=Path(cfg["dataset"]["datasetfs"]).resolve(),
            cwd=REPO_ROOT,
            log_path=out_dir / "daemon.log",
        )

    summary_rows: list[dict] = []
    try:
        for loader_name in cfg["loaders"]:
            for seed in cfg["seeds"]:
                stats = _run_one_cell(
                    loader_name, cfg, label_to_idx, seed, out_dir, daemon,
                )
                for ep in stats:
                    row: dict[str, Any] = {
                        "loader": loader_name,
                        "seed": seed,
                        "warmup": ep.epoch < cfg.get("warmup_epochs", 0),
                    }
                    row.update(ep.summary())
                    summary_rows.append(row)
                _write_summary(out_dir, summary_rows)  # incremental persist
    finally:
        if daemon is not None:
            daemon.stop()

    print(f"\n[runner] DONE. Output: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
