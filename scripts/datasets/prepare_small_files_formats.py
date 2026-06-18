"""Prepare ImageFolder/WebDataset/DatasetFS subsets for small-file benchmarks."""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from webdataset import ShardWriter

from scripts.datasets.datasetfs_writer import DatasetFSWriter


VALID_SUFFIXES = {".wav"}


def _list_balanced_samples(root: Path) -> list[tuple[Path, str]]:
    by_class: dict[str, list[Path]] = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = [
            p for p in sorted(class_dir.iterdir())
            if p.is_file() and not p.name.startswith("._") and p.suffix.lower() in VALID_SUFFIXES
        ]
        if files:
            by_class[class_dir.name] = files
    if not by_class:
        raise SystemExit(f"no .wav samples under {root}")

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
    with ShardWriter(str(out / "shard-%06d.tar"), maxsize=512 * 1024 * 1024) as sink:
        for idx, (src, label) in enumerate(samples):
            sink.write({
                "__key__": f"{idx:08d}",
                "wav": src.read_bytes(),
                "cls": label.encode("utf-8"),
            })
    (out / ".done").touch()


def _prepare_datasetfs(samples: list[tuple[Path, str]], out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    with DatasetFSWriter(out, shard_target_bytes=96 * 1024 * 1024) as writer:
        for src, label in samples:
            writer.add(f"{label}/{src.name}", src.read_bytes(), {"label": label})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-imagefolder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["imagefolder", "webdataset", "datasetfs"],
        default=["imagefolder", "webdataset", "datasetfs"],
    )
    args = parser.parse_args()

    samples = _list_balanced_samples(args.source_imagefolder)
    if args.sample_count > len(samples):
        raise SystemExit(f"requested {args.sample_count}, source has {len(samples)}")
    subset = samples[:args.sample_count]
    args.output.mkdir(parents=True, exist_ok=True)

    if "imagefolder" in args.formats:
        print(f"[prepare-small] imagefolder n={args.sample_count}", flush=True)
        _prepare_imagefolder(subset, args.output / "imagefolder")
    if "webdataset" in args.formats:
        print(f"[prepare-small] webdataset n={args.sample_count}", flush=True)
        _prepare_webdataset(subset, args.output / "webdataset")
    if "datasetfs" in args.formats:
        print(f"[prepare-small] datasetfs n={args.sample_count}", flush=True)
        _prepare_datasetfs(subset, args.output / "datasetfs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
