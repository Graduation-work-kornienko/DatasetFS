"""Stage a bounded PubLayNet remote-streaming subset from Yandex.Disk.

Downloads only enough DatasetFS/WebDataset shards to stay under a byte cap, then
rewrites the DatasetFS manifest so the daemon sees a valid subset rather than a
full dataset with missing shard files. The staged directory is meant to be served
over HTTP by the benchmark target and deleted afterwards.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path, PurePosixPath

from scripts.datasets.datasetfs_writer import read_parquet_manifest, write_parquet_manifest
from scripts.storage import ydisk


DEFAULT_REMOTE_ROOT = "datasetfs-bench/formats/publaynet"


def _fmt(n: int) -> str:
    return ydisk._fmt_bytes(n)  # type: ignore[attr-defined]


def _pull_file(client, remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    ydisk._pull_file(client, remote, local)  # type: ignore[attr-defined]


def _stage_datasetfs(client, remote_root: str, out: Path, max_bytes: int) -> int:
    remote = ydisk._norm_remote(f"{remote_root}/datasetfs")  # type: ignore[attr-defined]
    if not client.exists(remote):
        raise SystemExit(f"remote DatasetFS path does not exist: {remote}")

    out.mkdir(parents=True, exist_ok=True)
    full_manifest = out / "metadata.full.parquet"
    _pull_file(client, f"{remote}/metadata.parquet", full_manifest)
    tmp_manifest_dir = out / "_manifest_full"
    tmp_manifest_dir.mkdir(exist_ok=True)
    shutil.copy2(full_manifest, tmp_manifest_dir / "metadata.parquet")
    manifest = read_parquet_manifest(tmp_manifest_dir)
    shutil.rmtree(tmp_manifest_dir)

    selected: list[int] = []
    total = 0
    for shard_id, meta in sorted((int(k), v) for k, v in manifest["shards_meta"].items() if int(k) >= 0):
        size = int(meta.get("total_size", 0) or 0)
        if selected and total + size > max_bytes:
            break
        selected.append(shard_id)
        total += size
        if total >= max_bytes:
            break
    if not selected:
        raise SystemExit("no DatasetFS shards selected; max bytes too small")

    selected_set = set(selected)
    subset = {
        "version": manifest.get("version", "1.0"),
        "shards_meta": {str(i): manifest["shards_meta"][str(i)] for i in selected},
        "files": {
            path: info
            for path, info in manifest["files"].items()
            if int(info.get("c_id", -999)) in selected_set and not info.get("deleted")
        },
    }
    for shard_id in selected:
        _pull_file(client, f"{remote}/shard_{shard_id}", out / f"shard_{shard_id}")
    full_manifest.unlink(missing_ok=True)
    write_parquet_manifest(out, subset)
    (out / ".done").touch()
    print(
        f"[stage] datasetfs shards={len(selected)} objects={len(subset['files'])} bytes={_fmt(total)} -> {out}",
        flush=True,
    )
    return total


def _remote_files(client, remote: str) -> list[str]:
    files = list(ydisk._walk_remote(client, remote))  # type: ignore[attr-defined]
    return sorted(files, key=lambda p: PurePosixPath(p).name)


def _stage_webdataset(client, remote_root: str, out: Path, max_bytes: int) -> int:
    remote = ydisk._norm_remote(f"{remote_root}/webdataset")  # type: ignore[attr-defined]
    if not client.exists(remote):
        raise SystemExit(f"remote WebDataset path does not exist: {remote}")
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    count = 0
    for rfile in _remote_files(client, remote):
        name = PurePosixPath(rfile).name
        if not (name.startswith("shard-") and name.endswith(".tar")):
            continue
        size = ydisk._remote_file_size(client, rfile) or 0  # type: ignore[attr-defined]
        if count and total + size > max_bytes:
            break
        _pull_file(client, rfile, out / name)
        total += size
        count += 1
        if total >= max_bytes:
            break
    if count == 0:
        raise SystemExit("no WebDataset shards selected; max bytes too small")
    (out / ".done").touch()
    print(f"[stage] webdataset shards={count} bytes={_fmt(total)} -> {out}", flush=True)
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--output", type=Path, default=Path("runs/remote_stage/publaynet"))
    parser.add_argument("--max-bytes-per-format", type=int, default=6 * 1024 * 1024 * 1024)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists: {args.output}; pass --overwrite")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    client = ydisk.get_client()
    dfs_bytes = _stage_datasetfs(client, args.remote_root, args.output / "datasetfs", args.max_bytes_per_format)
    wds_bytes = _stage_webdataset(client, args.remote_root, args.output / "webdataset", args.max_bytes_per_format)

    # single_run only needs class directories to build label_to_idx. PubLayNet's
    # prepared DatasetFS/WebDataset rows use label="layout".
    (args.output / "imagefolder" / "layout").mkdir(parents=True, exist_ok=True)
    (args.output / "README.txt").write_text(
        "Bounded PubLayNet remote-streaming stage from Yandex.Disk.\n"
        f"remote_root={args.remote_root}\n"
        f"max_bytes_per_format={args.max_bytes_per_format}\n"
        f"datasetfs_bytes={dfs_bytes}\n"
        f"webdataset_bytes={wds_bytes}\n",
        encoding="utf-8",
    )
    print(f"[stage] ready {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
