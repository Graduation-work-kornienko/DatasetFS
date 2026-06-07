from __future__ import annotations

from benchmarks.datasetfs_bench.runner.remote_preflight import datasetfs_urls, validate_config, webdataset_urls


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

    assert datasetfs_urls(cfg) == ["http://example.test/dfs/metadata.jsonl"]
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
