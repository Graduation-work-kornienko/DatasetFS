"""Create an object-count scaled Speech Commands ImageFolder tree.

The output keeps the ImageFolder layout (`class/file.wav`) but gives every
replica a unique logical filename. This is intended for small-file scaling
benchmarks where the controlled variable is object count while average object
size and class distribution remain the same.

By default files are hardlinked, so the plain-file baseline gets real directory
entries and unique paths without multiplying disk payload. Use `--link-mode copy`
when a benchmark explicitly needs physically duplicated bytes.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path


VALID_SUFFIXES = {".wav"}


def _samples(source: Path) -> list[tuple[Path, str]]:
    rows: list[tuple[Path, str]] = []
    for class_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and not path.name.startswith("._") and path.suffix.lower() in VALID_SUFFIXES:
                rows.append((path, class_dir.name))
    if not rows:
        raise SystemExit(f"no .wav samples found under {source}")
    return rows


def _link(src: Path, dst: Path, mode: str) -> None:
    if mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(mode)


def replicate(source: Path, output: Path, replicas: int, link_mode: str, overwrite: bool) -> None:
    source = source.resolve()
    if not source.exists():
        raise SystemExit(f"source does not exist: {source}")
    if output.exists():
        if not overwrite:
            raise SystemExit(f"output already exists: {output}; pass --overwrite to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    base = _samples(source)
    total_files = len(base) * replicas
    total_payload = sum(path.stat().st_size for path, _ in base) * replicas
    print(
        f"[replicate] source={len(base)} files replicas={replicas} "
        f"target={total_files} files logical_payload={total_payload / (1024**3):.2f} GiB "
        f"mode={link_mode}",
        flush=True,
    )

    manifest_path = output / "replication_manifest.csv"
    done = 0
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["logical_path", "source_path", "label", "replica", "size_bytes"])
        writer.writeheader()
        for replica in range(replicas):
            for src, label in base:
                rel_name = f"rep{replica:04d}__{src.name}"
                dst_dir = output / label
                dst_dir.mkdir(exist_ok=True)
                dst = dst_dir / rel_name
                _link(src, dst, link_mode)
                size = src.stat().st_size
                writer.writerow({
                    "logical_path": str(dst.relative_to(output)),
                    "source_path": str(src),
                    "label": label,
                    "replica": replica,
                    "size_bytes": size,
                })
                done += 1
                if done % 100_000 == 0:
                    print(f"[replicate] linked {done}/{total_files}", flush=True)

    (output / ".done").touch()
    (output / "README.txt").write_text(
        "Replicated Speech Commands ImageFolder tree for object-count scaling.\n"
        f"source={source}\n"
        f"replicas={replicas}\n"
        f"link_mode={link_mode}\n"
        f"files={total_files}\n"
        f"logical_payload_bytes={total_payload}\n",
        encoding="utf-8",
    )
    print(f"[replicate] done -> {output}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/formats/speech_commands/imagefolder"))
    parser.add_argument("--output", type=Path, default=Path("data/formats/speech_commands_replicated_10x/imagefolder"))
    parser.add_argument("--replicas", type=int, default=10)
    parser.add_argument("--link-mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.replicas < 1:
        raise SystemExit("--replicas must be >= 1")
    replicate(args.source, args.output, args.replicas, args.link_mode, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
