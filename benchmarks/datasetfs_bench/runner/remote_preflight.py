"""Preflight checks for remote_streaming.yaml without downloading datasets."""
from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

import requests
import yaml

from scripts.datasets.datasetfs_writer import read_parquet_manifest


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _join_url(base: str, suffix: str) -> str:
    if base.startswith("ydisk://"):
        return base.rstrip("/") + "/" + suffix.lstrip("/")
    return urljoin(base.rstrip("/") + "/", suffix.lstrip("/"))


def datasetfs_urls(cfg: dict) -> list[str]:
    remote = cfg.get("datasetfs_remote") or {}
    if remote.get("manifest_url"):
        return [str(remote["manifest_url"])]
    root = remote.get("root_url")
    if not root:
        return []
    return [_join_url(str(root), "metadata.parquet")]


def datasetfs_legacy_urls(cfg: dict) -> list[str]:
    remote = cfg.get("datasetfs_remote") or {}
    root = remote.get("root_url")
    if not root:
        return []
    return [_join_url(str(root), "metadata.jsonl")]


def datasetfs_shard_urls_from_manifest(root_url: str, manifest_path: Path, max_shards: int = 3) -> list[str]:
    manifest = read_parquet_manifest(manifest_path.parent)
    shard_ids = sorted(int(k) for k in manifest.get("shards_meta", {}) if int(k) >= 0)
    return [_join_url(root_url, f"shard_{i}") for i in shard_ids[:max_shards]]


def datasetfs_total_bytes_from_manifest(manifest_path: Path) -> int:
    manifest = read_parquet_manifest(manifest_path.parent)
    return sum(int(s.get("total_size", 0) or 0) for s in manifest.get("shards_meta", {}).values())


def webdataset_urls(cfg: dict) -> list[str]:
    ds = cfg.get("dataset") or {}
    remote = ds.get("webdataset_remote") or {}
    urls = remote.get("shard_urls")
    if urls:
        return [str(u) for u in urls]
    base = remote.get("http_base")
    if not base:
        base = remote.get("ydisk_remote_root")
    if not base:
        return []
    base = str(base)
    if not base.startswith(("http://", "https://", "ydisk://")) and remote.get("ydisk_remote_root"):
        base = "ydisk://" + base.lstrip("/")
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


def _content_length(url: str, timeout: float) -> tuple[CheckResult, int | None]:
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        try:
            if r.status_code not in (200, 206):
                return CheckResult(url, False, f"HEAD HTTP {r.status_code}"), None
            raw = r.headers.get("Content-Length")
            if raw is None:
                return CheckResult(url, False, "HEAD missing Content-Length"), None
            return CheckResult(url, True, f"Content-Length={raw}"), int(raw)
        finally:
            r.close()
    except Exception as e:
        return CheckResult(url, False, repr(e)), None


def _download_manifest(url: str, timeout: float) -> tuple[CheckResult, Path | None, tempfile.TemporaryDirectory | None]:
    td = tempfile.TemporaryDirectory(prefix="datasetfs-remote-manifest-")
    path = Path(td.name) / "metadata.parquet"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            td.cleanup()
            return CheckResult(url, False, f"HTTP {r.status_code}"), None, None
        path.write_bytes(r.content)
        read_parquet_manifest(path.parent)
        return CheckResult(url, True, f"HTTP 200 parquet_bytes={len(r.content)}"), path, td
    except Exception as e:
        td.cleanup()
        return CheckResult(url, False, repr(e)), None, None


def _probe_absent(url: str, timeout: float) -> CheckResult:
    try:
        r = requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=timeout)
        try:
            if r.status_code == 404:
                return CheckResult(url, True, "absent")
            return CheckResult(url, False, f"legacy manifest present: HTTP {r.status_code}")
        finally:
            r.close()
    except Exception:
        return CheckResult(url, True, "absent/unreachable")


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
    max_total = int((cfg.get("remote_limits") or {}).get("max_total_bytes", 0) or 0)
    if max_total <= 0:
        results.append(CheckResult("remote_limits.max_total_bytes", False, "not set"))
    return results


def run_preflight(config_path: Path, timeout: float = 5.0, check_urls: bool = True) -> list[CheckResult]:
    cfg = yaml.safe_load(config_path.read_text())
    results = validate_config(cfg)
    max_total = int((cfg.get("remote_limits") or {}).get("max_total_bytes", 0) or 0)
    if check_urls:
        remote = cfg.get("datasetfs_remote") or {}
        root_url = str(remote.get("root_url") or "")
        for url in datasetfs_urls(cfg):
            result, manifest_path, td = _download_manifest(url, timeout)
            results.append(result)
            try:
                if manifest_path is not None:
                    total = datasetfs_total_bytes_from_manifest(manifest_path)
                    ok = max_total <= 0 or total <= max_total
                    results.append(CheckResult(
                        "datasetfs_remote.total_bytes",
                        ok,
                        f"{total} bytes (limit {max_total})",
                    ))
                    for shard_url in datasetfs_shard_urls_from_manifest(root_url, manifest_path):
                        results.append(_probe_url(shard_url, timeout))
            finally:
                if td is not None:
                    td.cleanup()
        for url in datasetfs_legacy_urls(cfg):
            results.append(_probe_absent(url, timeout))
        wds_total = 0
        wds_known = True
        for url in webdataset_urls(cfg):
            results.append(_probe_url(url, timeout))
            size_result, size = _content_length(url, timeout)
            results.append(size_result)
            if size is None:
                wds_known = False
            else:
                wds_total += size
        if webdataset_urls(cfg):
            ok = wds_known and (max_total <= 0 or wds_total <= max_total)
            detail = f"{wds_total} bytes (limit {max_total})" if wds_known else "unknown: missing Content-Length"
            results.append(CheckResult("webdataset_remote.total_bytes", ok, detail))
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
