"""Restore archived benchmark raw datasets from Yandex.Disk.

The archive convention is one tar per extracted raw directory:

    datasetfs-bench/raw-archives/imagenette2.tar
    datasetfs-bench/raw-archives/imagewoof2.tar
    datasetfs-bench/raw-archives/speech_commands_v2.tar

After restore, the normal ``scripts.datasets.prepare_formats`` command can
rebuild any required storage format locally without re-downloading from the
original dataset host.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"

DATASETS = {
    "imagenette": ("imagenette2", ".imagenette.done"),
    "imagewoof": ("imagewoof2", ".imagewoof.done"),
    "speech_commands": ("speech_commands_v2", ".speech_commands.done"),
}


def run(cmd: list[str]) -> None:
    print("[run] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def restore_one(name: str, ydisk_root: str, force: bool) -> None:
    raw_dir_name, marker_name = DATASETS[name]
    raw_root = DATA_ROOT / "raw"
    raw_dir = raw_root / raw_dir_name
    marker = raw_root / marker_name
    if raw_dir.exists() and marker.exists() and not force:
        print(f"[skip] {name}: already restored at {raw_dir}", flush=True)
        return
    if raw_dir.exists() and force:
        print(f"[rm] {raw_dir}", flush=True)
        shutil.rmtree(raw_dir)

    archive_dir = DATA_ROOT / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    tar_path = archive_dir / f"{raw_dir_name}.tar"
    remote = f"{ydisk_root}/raw-archives/{raw_dir_name}.tar"

    run([sys.executable, "-m", "scripts.storage.ydisk", "pull", remote, str(tar_path)])
    raw_root.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {tar_path} -> {raw_root}", flush=True)
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(raw_root, filter="data")
    marker.touch()
    tar_path.unlink(missing_ok=True)
    print(f"[done] {name}: restored {raw_dir}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--ydisk-root", default="datasetfs-bench")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for name in args.names:
        restore_one(name, args.ydisk_root, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
