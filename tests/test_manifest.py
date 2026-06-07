"""Manifest <-> on-disk consistency tests.

If the converter crashed mid-write or a shard was deleted, the manifest could
list shards whose files don't exist (or are wrong size). Catch that statically.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from scripts.datasets.datasetfs_writer import read_parquet_manifest


pytestmark = pytest.mark.timeout(60)


def _load_manifest(datasetfs_root: Path) -> dict:
    return read_parquet_manifest(datasetfs_root)


def _check_dataset(datasetfs_root: Path) -> None:
    """Manifest invariants:
      - every shard_id in shards_meta has a corresponding file on disk
      - every file in `files` references a known shard_id
      - delta shard (id=-1) is allowed to be empty / smaller
    """
    manifest = _load_manifest(datasetfs_root)
    shards_meta = manifest.get("shards_meta", {})
    files = manifest.get("files", {})
    assert shards_meta, f"manifest has no shards_meta at {datasetfs_root}"
    assert files, f"manifest has no files at {datasetfs_root}"

    missing_files = []
    suspicious_size = []
    for shard_id_str, info in shards_meta.items():
        sid = int(shard_id_str)
        shard_path = datasetfs_root / f"shard_{sid}"
        if not shard_path.exists():
            missing_files.append(shard_id_str)
            continue
        on_disk = shard_path.stat().st_size
        total = info.get("total_size", 0)
        if sid == -1:
            # Delta shard: not exercised in this conversion path
            continue
        # On-disk size includes tar EOF marker (~1024 zero bytes appended by
        # tar.Writer.Close), so it can exceed total_size slightly.
        if total == 0:
            suspicious_size.append((shard_id_str, "manifest total_size=0"))
        elif on_disk < total:
            suspicious_size.append(
                (shard_id_str, f"file {on_disk}B < manifest {total}B")
            )

    assert not missing_files, f"manifest references missing shards: {missing_files}"
    assert not suspicious_size, f"shard size inconsistencies: {suspicious_size}"

    # Every file's c_id must reference a known shard
    known_ids = {int(k) for k in shards_meta.keys()}
    orphaned = []
    for name, meta in files.items():
        cid = meta.get("c_id")
        if cid not in known_ids:
            orphaned.append((name, cid))
    assert not orphaned, (
        f"{len(orphaned)} files reference unknown shards; first 5: {orphaned[:5]}"
    )


def test_manifest_imagenette(imagenette_prepared):
    _check_dataset(imagenette_prepared["datasetfs"])


def test_manifest_imagewoof(imagewoof_prepared):
    _check_dataset(imagewoof_prepared["datasetfs"])
