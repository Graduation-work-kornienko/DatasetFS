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
import os
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

from benchmarks.datasetfs_bench.loaders.base import FormatUnavailable
from benchmarks.datasetfs_bench.loaders.datasetfs import DatasetFSLoader
from benchmarks.datasetfs_bench.loaders.ffcv_loader import FFCVLoader
from benchmarks.datasetfs_bench.loaders.hdf5_loader import HDF5Loader
from benchmarks.datasetfs_bench.loaders.hf_loader import HuggingFaceLoader
from benchmarks.datasetfs_bench.loaders.imagefolder import ImageFolderLoader
from benchmarks.datasetfs_bench.loaders.lmdb_loader import LMDBLoader
from benchmarks.datasetfs_bench.loaders.synthetic import SyntheticLoader
from benchmarks.datasetfs_bench.loaders.tfrecord_loader import TFRecordLoader
from benchmarks.datasetfs_bench.loaders.webdataset_loader import WebDatasetLoader
from benchmarks.datasetfs_bench.metrics import daemon as daemon_metrics
from benchmarks.datasetfs_bench.metrics.system import SystemSampler
from benchmarks.datasetfs_bench.metrics.training import EpochStats
from benchmarks.datasetfs_bench.models.registry import build_model
from benchmarks.datasetfs_bench.runner import cache_control, host_info
from benchmarks.datasetfs_bench.runner.daemon_ctl import DaemonManager
from benchmarks.datasetfs_bench.train.loop import train_one_epoch


LOADER_CLASSES = {
    "datasetfs": DatasetFSLoader,
    "webdataset": WebDatasetLoader,
    "imagefolder": ImageFolderLoader,
    # Format matrix (G1): same files, different storage engines.
    "lmdb": LMDBLoader,
    "hdf5": HDF5Loader,
    "tfrecord": TFRecordLoader,
    "huggingface": HuggingFaceLoader,
    "ffcv": FFCVLoader,
    "synthetic": SyntheticLoader,
}


def _load_dotenv_var(name: str) -> None:
    if os.environ.get(name):
        return
    dotenv = REPO_ROOT / ".env"
    if not dotenv.is_file():
        return
    for line in dotenv.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == name:
            os.environ[name] = value.strip().strip('"').strip("'")
            return

# Formats whose prepared data lives at cfg["dataset"][<name>] (falling back to
# the conventional data/formats/<ds>/<name> path) and whose loader just needs
# that root. Keeps _build_loader from growing one branch per format.
_ROOT_FORMATS = ("lmdb", "hdf5", "tfrecord", "huggingface", "ffcv")


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _loader_cells(cfg: dict) -> list[tuple[str, str, dict]]:
    """Normalize `cfg['loaders']` into (display_name, format, overrides) cells.

    Each entry is either a plain string (`display_name == format`, no overrides)
    or a dict with a required `format` plus an optional `name` (the bar label,
    defaults to `format`) and any per-cell config overrides merged over the
    top-level cfg. This lets ONE matrix carry several configurations of the same
    format as distinct bars — e.g. both `datasetfs` (raw, Python-side decode, an
    apples-to-apples storage comparison) and `datasetfs-rgb` (server-side
    rgb_uint8 + parallel decode, DFS's unique edge) in the G1 graph:

        loaders:
          - datasetfs
          - {format: datasetfs, name: datasetfs-rgb, dfs_decode_mode: rgb_uint8}
    """
    cells: list[tuple[str, str, dict]] = []
    for entry in cfg["loaders"]:
        if isinstance(entry, str):
            cells.append((entry, entry, {}))
        elif isinstance(entry, dict):
            if "format" not in entry:
                raise ValueError(f"loaders entry missing 'format': {entry!r}")
            fmt = entry["format"]
            name = entry.get("name", fmt)
            overrides = {k: v for k, v in entry.items() if k not in ("name", "format")}
            cells.append((name, fmt, overrides))
        else:
            raise ValueError(f"bad loaders entry (want str or dict): {entry!r}")
    return cells


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
    # Modality (image|audio) routes every format's decode path; default image.
    if "modality" in cfg:
        common["modality"] = cfg["modality"]
    cls = LOADER_CLASSES[loader_name]
    if loader_name == "imagefolder":
        return cls({**common, "root": ds["imagefolder"]})
    if loader_name == "webdataset":
        spec = {**common, "root": ds["webdataset"]}
        if "webdataset_remote" in ds:
            spec.update(ds["webdataset_remote"])
        if "wds_cache_dir" in cfg:
            spec["wds_cache_dir"] = cfg["wds_cache_dir"]
        if "wds_cache_size" in cfg:
            spec["wds_cache_size"] = cfg["wds_cache_size"]
        return cls(spec)
    if loader_name == "datasetfs":
        spec = {**common, "root": ds["datasetfs"]}
        # Modality selects the client decode path: "image" (default, PIL or
        # server-side rgb_uint8) or "audio" (soundfile → mel spectrogram, the
        # generic raw transport — opt 03). Audio rejects rgb_uint8.
        if "modality" in cfg:
            spec["modality"] = cfg["modality"]
        # Optional server-side decode (Phase 3 architectural optimization):
        # daemon does JPEG decode + resize, Python skips PIL. Default is "raw".
        if "dfs_decode_mode" in cfg:
            spec["decode_mode"] = cfg["dfs_decode_mode"]
        # Optional decode parallelism (opt 02): daemon decode worker goroutines
        # per pipeline. 0/absent = auto (NumCPU/num_workers).
        if "dfs_decode_parallelism" in cfg:
            spec["decode_parallelism"] = cfg["dfs_decode_parallelism"]
        if "dfs_timeout_seconds" in cfg:
            spec["timeout_seconds"] = cfg["dfs_timeout_seconds"]
        return cls(spec)
    if loader_name == "synthetic":
        spec = {**common}
        if "synthetic_samples" in cfg:
            spec["synthetic_samples"] = cfg["synthetic_samples"]
        return cls(spec)
    if loader_name in _ROOT_FORMATS:
        root = ds.get(loader_name, f"data/formats/{ds['name']}/{loader_name}")
        return cls({**common, "root": root})
    raise ValueError(loader_name)


def _run_one_cell(
    display_name: str,
    loader_format: str,
    cfg: dict,
    label_to_idx: dict,
    seed: int,
    out_dir: Path,
    daemon: DaemonManager | None,
) -> tuple[list[tuple[EpochStats, dict]], dict, dict, list[dict[str, Any]]]:
    """Run one (loader, seed) cell, returning (epoch_stats, system_summary, daemon_summary).

    `display_name` is the bar label (may differ from `loader_format` when a
    matrix carries several configs of one format — see `_loader_cells`); `cfg`
    is already the per-cell effective config (top-level merged with overrides)."""
    print(f"\n=== {display_name} seed={seed} ===", flush=True)
    _set_all_seeds(seed)

    if loader_format == "datasetfs":
        assert daemon is not None, "DatasetFS run requires a daemon"
        # Fresh session per seed — avoids any state leak from previous run.
        daemon.restart()

    loader = _build_loader(loader_format, cfg, label_to_idx, seed)
    loader.setup()

    model = build_model(cfg["model"], num_classes=len(label_to_idx))
    optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    # Track Python tree + daemon (if DFS) for per-process RSS.
    track_pids = [os.getpid()]
    if loader_format == "datasetfs" and daemon is not None and daemon.pid is not None:
        track_pids.append(daemon.pid)

    track_labels = {"python": os.getpid()}
    if loader_format == "datasetfs" and daemon is not None and daemon.pid is not None:
        track_labels["daemon"] = daemon.pid
    sampler = SystemSampler(interval_s=0.2, track_pids=track_pids, track_labels=track_labels)
    daemon_sampler = None
    if loader_format == "datasetfs":
        daemon_sampler = daemon_metrics.DaemonSampler(
            interval_s=float(cfg.get("daemon_metrics_interval_s", 0.5)),
            context={"loader": display_name, "seed": seed},
        )
    daemon_before = daemon_metrics.snapshot() if loader_format == "datasetfs" else {}

    sampler.start()
    if daemon_sampler is not None:
        daemon_sampler.start()
    stats: list[tuple[EpochStats, dict]] = []
    try:
        for epoch in range(cfg["epochs"]):
            dl = loader.make_loader()
            epoch_daemon_before = daemon_metrics.snapshot() if loader_format == "datasetfs" else {}
            ep_stats = train_one_epoch(
                model, dl, optim, loss_fn,
                epoch_idx=epoch,
                max_batches=cfg.get("max_batches_per_epoch"),
                warmup_batches=cfg.get("warmup_batches", 0),
            )
            epoch_daemon_after = daemon_metrics.snapshot() if loader_format == "datasetfs" else {}
            epoch_daemon_summary = daemon_metrics.cell_summary(epoch_daemon_before, epoch_daemon_after)
            print(
                f"  epoch={epoch} sps={ep_stats.samples_per_second:.1f} "
                f"steady_sps={ep_stats.steady_samples_per_second:.1f} "
                f"TTFB={ep_stats.time_to_first_batch:.2f}s "
                f"steady_wait={ep_stats.steady_batch_wait_fraction:.2%} "
                f"full_wait={ep_stats.stall_fraction:.2%}",
                flush=True,
            )
            stats.append((ep_stats, epoch_daemon_summary))
            # Drop the DataLoader workers between epochs to avoid pipe state
            # bleeding across epochs.
            del dl
    finally:
        if daemon_sampler is not None:
            daemon_sampler.stop()
        sampler.stop()
        loader.teardown()

    daemon_after = daemon_metrics.snapshot() if loader_format == "datasetfs" else {}
    daemon_samples = daemon_sampler.samples if daemon_sampler is not None else []
    return stats, sampler.summary(), daemon_metrics.cell_summary(daemon_before, daemon_after), daemon_samples


def _write_summary(out_dir: Path, rows: list[dict]) -> None:
    csv_path = out_dir / "summary.csv"
    if not rows:
        return
    # Union of all keys — different loaders contribute different fields
    # (daemon_* only on DatasetFS rows). Missing values render as "".
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
    print(f"\n[runner] wrote {csv_path}", flush=True)


def _resolve_cache_dir(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _clear_cache_dir(path: str | Path | None, label: str) -> None:
    p = _resolve_cache_dir(path)
    if p is None:
        return
    if p in (Path("/"), Path.home(), REPO_ROOT):
        raise ValueError(f"refusing to clear unsafe {label} cache path: {p}")
    if p.exists():
        print(f"[runner] clear {label} cache: {p}", flush=True)
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def run_config(cfg: dict, out_dir: Path) -> list[dict]:
    """Run all (loader, seed) cells for one config; return per-epoch summary rows.

    Side effects: writes `summary.csv`, `host_info.json`, optional `daemon.log`
    into `out_dir`. Used both by single_run.main and by sweep.py.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    host_info.write(out_dir / "host_info.json")

    if str((cfg.get("datasetfs_remote") or {}).get("root_url", "")).startswith("ydisk://"):
        _load_dotenv_var("YADISK_TOKEN")

    label_to_idx = _label_to_idx(Path(cfg["dataset"]["imagefolder"]))
    print(f"[runner] {len(label_to_idx)} classes", flush=True)

    cells = _loader_cells(cfg)

    # Spin up the daemon once if any cell is a DatasetFS format (possibly under
    # several display names, e.g. raw + rgb_uint8). We reuse the process across
    # seeds; each seed calls daemon.restart() for a fresh session (handled
    # inside _run_one_cell), and the decode-mode override is sent per cell.
    daemon: DaemonManager | None = None
    if any(fmt == "datasetfs" for _, fmt, _ in cells):
        remote = cfg.get("datasetfs_remote") or {}
        root_path = remote.get("root_url") or Path(cfg["dataset"]["datasetfs"]).resolve()
        daemon = DaemonManager(
            binary=REPO_ROOT / cfg["daemon_binary"],
            root_path=root_path,
            cwd=REPO_ROOT,
            log_path=out_dir / "daemon.log",
            cache_dir=remote.get("cache_dir"),
            remote_manifest_url=remote.get("manifest_url"),
            prefetch_concurrency=remote.get("prefetch_concurrency"),
            remote_throttle=remote.get("remote_throttle"),
        )

    drop_caches = cfg.get("drop_page_cache_between_cells", False)
    if drop_caches and not cache_control.can_drop_caches():
        print(
            "[runner] WARN: drop_page_cache_between_cells=true but `sudo -n` "
            "is not configured for the cache-drop command. Cells will run "
            "with whatever cache state the previous run left behind. "
            "Configure passwordless sudo for `purge` (macOS) or "
            "`drop_caches` (Linux) for honest cold-cache numbers.",
            flush=True,
        )
        drop_caches = False

    summary_rows: list[dict] = []
    daemon_timeseries_rows: list[dict[str, Any]] = []
    try:
        for display_name, loader_format, overrides in cells:
            cfg_eff = {**cfg, **overrides}
            for seed in cfg["seeds"]:
                if cfg_eff.get("reset_remote_caches_before_seed"):
                    if loader_format == "datasetfs":
                        _clear_cache_dir((cfg_eff.get("datasetfs_remote") or {}).get("cache_dir"), display_name)
                    if loader_format == "webdataset":
                        _clear_cache_dir(cfg_eff.get("wds_cache_dir"), display_name)
                # Drop OS page cache before each cell for fair I/O measurement.
                cache_state = "uncontrolled"
                if drop_caches:
                    if cache_control.drop_page_cache():
                        cache_state = "cold"
                    else:
                        cache_state = "uncontrolled"

                try:
                    ep_stats, sys_summary, dmn_summary, dmn_timeseries = _run_one_cell(
                        display_name, loader_format, cfg_eff, label_to_idx, seed, out_dir, daemon,
                    )
                except FormatUnavailable as e:
                    # Sparse matrix cell (e.g. FFCV on macOS, or data not
                    # prepared). Log and skip the rest of this loader's seeds.
                    print(f"[runner] SKIP {display_name}: {e}", flush=True)
                    break
                for ep, ep_dmn_summary in ep_stats:
                    row: dict[str, Any] = {
                        "loader": display_name,
                        "seed": seed,
                        "warmup": ep.epoch < cfg.get("warmup_epochs", 0),
                        "cache_state": cache_state,
                    }
                    row.update(ep.summary())
                    # System + daemon metrics are per-CELL not per-epoch, but we
                    # attach them to every epoch row to keep the schema flat.
                    # Plot code can dedupe by (loader, seed) when needed.
                    row.update({f"sys_{k}": v for k, v in sys_summary.items()})
                    row.update({f"epoch_{k}": v for k, v in ep_dmn_summary.items()})
                    row.update(dmn_summary)
                    summary_rows.append(row)
                _write_summary(out_dir, summary_rows)  # incremental persist
                if dmn_timeseries:
                    daemon_timeseries_rows.extend(dmn_timeseries)
                    daemon_metrics.write_rows_union(out_dir / "daemon_timeseries.csv", daemon_timeseries_rows)
    finally:
        if daemon is not None:
            daemon.stop()

    print(f"\n[runner] cell complete. Output: {out_dir}", flush=True)
    return summary_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_dir / "config.yaml")
    run_config(cfg, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
