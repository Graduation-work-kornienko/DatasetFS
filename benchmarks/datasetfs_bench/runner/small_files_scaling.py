"""Benchmark loader wait time vs number of small files.

Builds nested ImageFolder-style subsets from one source and materializes each
subset in three formats: plain files (ImageFolder), WebDataset, and DatasetFS.
Using one source keeps object-size distribution comparable across x-axis points.

For the thesis-scale run, first create the 10x source with:

    python -m scripts.datasets.replicate_speech_commands --replicas 10

Then run this module with the default sample counts (1k/10k/100k/1M).
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.datasetfs_bench.runner.single_run import run_config
from scripts.datasets.datasetfs_writer import DatasetFSWriter


SUFFIXES = {
    "image": {".jpg", ".jpeg", ".jpe", ".png"},
    "audio": {".wav"},
}


def _list_balanced_samples(root: Path, modality: str) -> list[tuple[Path, str]]:
    suffixes = SUFFIXES[modality]
    by_class: dict[str, list[Path]] = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = [
            p for p in sorted(class_dir.iterdir())
            if p.is_file() and not p.name.startswith("._") and p.suffix.lower() in suffixes
        ]
        if files:
            by_class[class_dir.name] = files
    if not by_class:
        raise ValueError(f"no image samples under {root}")

    out: list[tuple[Path, str]] = []
    max_len = max(len(v) for v in by_class.values())
    for i in range(max_len):
        for label in sorted(by_class):
            files = by_class[label]
            if i < len(files):
                out.append((files[i], label))
    return out


def _prepare_imagefolder(samples: list[tuple[Path, str]], out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for src, label in samples:
        class_dir = out / label
        class_dir.mkdir(exist_ok=True)
        os.symlink(src.resolve(), class_dir / src.name)
    (out / ".done").touch()


def _prepare_webdataset(samples: list[tuple[Path, str]], out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    from webdataset import ShardWriter

    with ShardWriter(str(out / "shard-%06d.tar"), maxsize=500 * 1024 * 1024) as sink:
        for idx, (src, label) in enumerate(samples):
            data = src.read_bytes()
            ext = src.suffix.lower().lstrip(".") or "jpg"
            if ext in ("jpeg", "jpe"):
                ext = "jpg"
            sink.write({
                "__key__": f"{idx:08d}",
                ext: data,
                "cls": label.encode("utf-8"),
            })
    (out / ".done").touch()


def _prepare_datasetfs(samples: list[tuple[Path, str]], out: Path) -> None:
    # The daemon loads one shard into one shared-memory slot (110 MiB). Keep
    # prepared shards below that limit so large small-file subsets remain valid.
    with DatasetFSWriter(out, shard_target_bytes=96 * 1024 * 1024) as writer:
        for src, label in samples:
            writer.add(f"{label}/{src.name}", src.read_bytes(), {"label": label})


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
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


def _config(sample_count: int, root: Path, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dataset": {
            "name": f"small_files_{sample_count}",
            "imagefolder": str(root / "imagefolder"),
            "webdataset": str(root / "webdataset"),
            "datasetfs": str(root / "datasetfs"),
        },
        "model": args.model,
        "modality": args.modality,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "image_size": args.image_size,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "warmup_batches": args.warmup_batches,
        "seeds": args.seeds,
        "loaders": ["imagefolder", "webdataset", "datasetfs"],
        "drop_page_cache_between_cells": False,
        "daemon_binary": "bin/datasetfs",
    }


def run(args: argparse.Namespace) -> Path:
    source = args.source_imagefolder
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    samples_all = _list_balanced_samples(source, args.modality)
    max_count = max(args.sample_counts)
    if max_count > len(samples_all):
        raise ValueError(f"requested {max_count} samples, but source has only {len(samples_all)}")

    all_rows: list[dict[str, Any]] = []
    for sample_count in args.sample_counts:
        subset = samples_all[:sample_count]
        total_bytes = sum(src.stat().st_size for src, _ in subset)
        avg_bytes = total_bytes / sample_count
        subset_root = output / "prepared" / f"n_{sample_count:06d}"
        print(
            f"[small-files] n={sample_count} total={total_bytes / (1024**2):.1f} MiB "
            f"avg={avg_bytes / 1024:.1f} KiB",
            flush=True,
        )
        _prepare_imagefolder(subset, subset_root / "imagefolder")
        _prepare_webdataset(subset, subset_root / "webdataset")
        _prepare_datasetfs(subset, subset_root / "datasetfs")

        run_dir = output / f"n_{sample_count:06d}"
        cfg = _config(sample_count, subset_root, args)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        rows = run_config(cfg, run_dir)
        for row in rows:
            row = dict(row)
            row["sample_count"] = sample_count
            row["total_file_bytes"] = total_bytes
            row["avg_file_bytes"] = avg_bytes
            all_rows.append(row)
        _write_csv(output / "summary.csv", all_rows)

    meta = {
        "source_imagefolder": str(source),
        "modality": args.modality,
        "sample_counts": args.sample_counts,
        "metric": "steady_batch_wait_total_s",
        "formats": ["imagefolder", "webdataset", "datasetfs"],
    }
    (output / "config.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    print(f"[small-files] wrote {output}", flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-imagefolder",
        type=Path,
        default=Path("data/formats/speech_commands_replicated_10x/imagefolder"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-counts", type=int, nargs="+", default=[1000, 10000, 100000, 1000000])
    parser.add_argument("--modality", choices=sorted(SUFFIXES), default="audio")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--model", default="simplecnn_audio")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
