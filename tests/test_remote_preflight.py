from __future__ import annotations

from benchmarks.datasetfs_bench.runner.remote_preflight import (
    datasetfs_shard_urls_from_manifest,
    datasetfs_urls,
    validate_config,
    webdataset_urls,
)
from scripts.datasets.datasetfs_writer import DatasetFSWriter


def _cfg():
    return {
        "dataset": {
            "webdataset_remote": {
                "http_base": "http://example.test/wds/",
                "num_shards": 2,
                "shard_pattern": "shard-{i:06d}.tar",
                "wds_http_mode": "curl",
                "wds_curl_limit_rate": "20m",
            }
        },
        "datasetfs_remote": {
            "root_url": "http://example.test/dfs",
            "remote_throttle": 20_971_520,
        },
        "loaders": [
            {"format": "webdataset", "name": "webdataset-remote"},
            {"format": "datasetfs", "name": "datasetfs-remote"},
        ],
    }


def test_remote_preflight_resolves_urls():
    cfg = _cfg()

    assert datasetfs_urls(cfg) == ["http://example.test/dfs/metadata.parquet"]
    assert webdataset_urls(cfg) == [
        "http://example.test/wds/shard-000000.tar",
        "http://example.test/wds/shard-000001.tar",
    ]


def test_remote_preflight_validates_required_settings():
    assert validate_config(_cfg()) == []

    bad = _cfg()
    bad["dataset"]["webdataset_remote"]["wds_http_mode"] = "native"
    results = validate_config(bad)

    assert any(r.name == "dataset.webdataset_remote.wds_http_mode" and not r.ok for r in results)


def test_remote_preflight_reads_datasetfs_shards_from_parquet_manifest(tmp_path):
    root = tmp_path / "datasetfs"
    with DatasetFSWriter(root, shard_target_bytes=700) as writer:
        writer.add("a.bin", b"a" * 100, {})
        writer.add("b.bin", b"b" * 100, {})
        writer.add("c.bin", b"c" * 100, {})

    assert datasetfs_shard_urls_from_manifest("http://example.test/dfs", root / "metadata.parquet") == [
        "http://example.test/dfs/shard_0",
        "http://example.test/dfs/shard_1",
        "http://example.test/dfs/shard_2",
    ]
