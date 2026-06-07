from __future__ import annotations

import tarfile
from pathlib import Path

from scripts.datasets.datasetfs_writer import DatasetFSWriter, read_parquet_manifest


def test_datasetfs_writer_manifest_offsets_and_shard_rollover(tmp_path: Path):
    out = tmp_path / "datasetfs"
    payloads = {
        "a.bin": b"a" * 100,
        "b.bin": b"b" * 100,
        "c.bin": b"c" * 100,
    }
    with DatasetFSWriter(out, shard_target_bytes=700) as writer:
        for name, payload in payloads.items():
            writer.add(name, payload, {"label": name[0]})

    manifest = read_parquet_manifest(out)
    assert (out / ".done").exists()
    assert (out / "metadata.parquet").exists()
    assert not (out / "metadata.jsonl").exists()
    assert set(manifest["files"]) == set(payloads)
    assert len(manifest["shards_meta"]) >= 2

    for name, meta in manifest["files"].items():
        shard_path = out / f"shard_{meta['c_id']}"
        with open(shard_path, "rb") as f:
            f.seek(meta["offset"])
            assert f.read(meta["size"]) == payloads[name]
        assert meta["meta"]["label"] == name[0]

    with tarfile.open(out / "shard_0", "r") as tf:
        assert tf.getmembers(), "first shard should be a valid tar"


def test_datasetfs_writer_rejects_duplicate_logical_paths(tmp_path: Path):
    with DatasetFSWriter(tmp_path / "datasetfs") as writer:
        writer.add("x.bin", b"one", {})
        try:
            writer.add("x.bin", b"two", {})
        except ValueError as e:
            assert "duplicate" in str(e)
        else:
            raise AssertionError("duplicate logical path was accepted")
