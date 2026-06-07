"""Preflight checks for remote_streaming.yaml without downloading datasets."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

import requests
import yaml


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _join_url(base: str, suffix: str) -> str:
    return urljoin(base.rstrip("/") + "/", suffix.lstrip("/"))


def datasetfs_urls(cfg: dict) -> list[str]:
    remote = cfg.get("datasetfs_remote") or {}
    root = remote.get("root_url")
    if not root:
        return []
    return [_join_url(str(root), "metadata.jsonl")]


def webdataset_urls(cfg: dict) -> list[str]:
    ds = cfg.get("dataset") or {}
    remote = ds.get("webdataset_remote") or {}
    urls = remote.get("shard_urls")
    if urls:
        return [str(u) for u in urls]
    base = remote.get("http_base")
    if not base:
        return []
    n = int(remote.get("num_shards", 0) or 0)
    pattern = remote.get("shard_pattern", "shard-{i:06d}.tar")
    return [_join_url(str(base), pattern.format(i=i)) for i in range(n)]


def _probe_url(url: str, timeout: float) -> CheckResult:
    headers = {"Range": "bytes=0-0"}
    try:
        r = requests.get(url, headers=headers, stream=True, timeout=timeout)
        try:
            if r.status_code in (200, 206):
                return CheckResult(url, True, f"HTTP {r.status_code}")
            return CheckResult(url, False, f"HTTP {r.status_code}")
        finally:
            r.close()
    except Exception as e:
        return CheckResult(url, False, repr(e))


def validate_config(cfg: dict) -> list[CheckResult]:
    results: list[CheckResult] = []
    loaders = cfg.get("loaders") or []
    loader_names = [x.get("format") if isinstance(x, dict) else x for x in loaders]
    if "datasetfs" not in loader_names:
        results.append(CheckResult("loader:datasetfs", False, "missing datasetfs loader"))
    if "webdataset" not in loader_names:
        results.append(CheckResult("loader:webdataset", False, "missing webdataset loader"))

    remote = cfg.get("datasetfs_remote") or {}
    ds = cfg.get("dataset") or {}
    wds_remote = ds.get("webdataset_remote") or {}
    dfs_throttle = remote.get("remote_throttle")
    wds_limit = wds_remote.get("wds_curl_limit_rate")
    if not dfs_throttle:
        results.append(CheckResult("datasetfs_remote.remote_throttle", False, "not set"))
    if not wds_limit:
        results.append(CheckResult("dataset.webdataset_remote.wds_curl_limit_rate", False, "not set"))
    if wds_remote.get("wds_http_mode") != "curl":
        results.append(CheckResult("dataset.webdataset_remote.wds_http_mode", False, "must be curl for --limit-rate"))
    if not datasetfs_urls(cfg):
        results.append(CheckResult("datasetfs_remote.root_url", False, "not set"))
    if not webdataset_urls(cfg):
        results.append(CheckResult("dataset.webdataset_remote", False, "no shard URLs resolved"))
    return results


def run_preflight(config_path: Path, timeout: float = 5.0, check_urls: bool = True) -> list[CheckResult]:
    cfg = yaml.safe_load(config_path.read_text())
    results = validate_config(cfg)
    if check_urls:
        for url in datasetfs_urls(cfg):
            results.append(_probe_url(url, timeout))
        for url in webdataset_urls(cfg):
            results.append(_probe_url(url, timeout))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("benchmarks/datasetfs_bench/configs/remote_streaming.yaml"))
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--no-url-check", action="store_true")
    args = parser.parse_args()

    results = run_preflight(args.config, timeout=args.timeout, check_urls=not args.no_url_check)
    failed = False
    for result in results:
        status = "OK" if result.ok else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}", flush=True)
        failed = failed or not result.ok
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
