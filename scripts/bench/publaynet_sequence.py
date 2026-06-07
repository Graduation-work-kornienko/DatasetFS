"""Sequential PubLayNet >RAM benchmark driver.

Runs one storage format at a time so the 40GB PubLayNet working set fits on a
small local disk:

1. optionally archive old local datasets to Yandex.Disk;
2. prepare one PubLayNet format;
3. run the remote benchmark first (when the format supports HTTP streaming);
4. run the local benchmark;
5. archive results + prepared artifacts;
6. delete local PubLayNet data before moving to the next format.

The remote origin is a local HTTP server over the prepared format directory. The
benchmark still measures remote/streaming code paths, and bandwidth is controlled
explicitly: DatasetFS uses daemon ``--remote-throttle``; WebDataset uses
``curl --limit-rate``. ImageFolder has no HTTP streaming backend, so its remote
cell is recorded as N/A and only the local cell is run.
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import shutil
import socketserver
import subprocess
import sys
import tarfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
PUB_RAW = DATA_ROOT / "raw" / "publaynet"
PUB_FORMATS = DATA_ROOT / "formats" / "publaynet"
ARCHIVES = DATA_ROOT / "archives"

FORMATS = ("imagefolder", "webdataset", "datasetfs", "datasetfs-rgb")


def run(cmd: list[str], *, cwd: Path = REPO_ROOT) -> None:
    print("[run] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def ydisk(args: argparse.Namespace, op: str, src: Path | str, dst: Path | str) -> bool:
    if args.no_ydisk:
        print(f"[ydisk skip] {op} {src} {dst}", flush=True)
        return False
    if not os.environ.get("YADISK_TOKEN") and not (REPO_ROOT / ".env").exists():
        raise RuntimeError("YADISK_TOKEN is not set and .env is absent; cannot archive to Yandex.Disk")
    run([sys.executable, "-m", "scripts.storage.ydisk", op, str(src), str(dst)])
    return True


def archive_if_exists(args: argparse.Namespace, local: Path, remote: str, delete: bool) -> bool:
    if not local.exists():
        print(f"[archive skip] {local} absent", flush=True)
        return False
    uploaded = ydisk(args, "push", local, remote)
    if delete and uploaded:
        print(f"[delete] {local}", flush=True)
        if local.is_dir() and not local.is_symlink():
            shutil.rmtree(local)
        else:
            local.unlink()
    elif delete and not uploaded:
        print(f"[delete skip] {local} was not archived", flush=True)
    return uploaded


def archive_dir_as_tar(args: argparse.Namespace, local: Path, remote_tar: str, delete: bool) -> None:
    """Archive a directory as one tar before Yandex.Disk upload.

    Recursive `ydisk push` over datasets with tens of thousands of files is very
    slow; a single tar is much faster and easier to restore.
    """
    if not local.exists():
        print(f"[archive skip] {local} absent", flush=True)
        return
    ARCHIVES.mkdir(parents=True, exist_ok=True)
    tar_path = ARCHIVES / f"{local.name}.tar"
    print(f"[tar] {local} -> {tar_path}", flush=True)
    with tarfile.open(tar_path, "w") as tf:
        tf.add(local, arcname=local.name)
    try:
        uploaded = archive_if_exists(args, tar_path, remote_tar, delete=True)
        if delete and uploaded:
            print(f"[delete] {local}", flush=True)
            shutil.rmtree(local)
        elif delete and not uploaded:
            print(f"[delete skip] {local} was not archived", flush=True)
    finally:
        tar_path.unlink(missing_ok=True)


def archive_old_datasets(args: argparse.Namespace) -> None:
    for dirname in ("imagenette2", "imagewoof2", "speech_commands_v2"):
        archive_dir_as_tar(
            args,
            DATA_ROOT / "raw" / dirname,
            f"{args.ydisk_root}/raw-archives/{dirname}.tar",
            args.delete_old,
        )
    for ds in ("imagenette", "imagewoof", "speech_commands"):
        archive_if_exists(args, DATA_ROOT / "formats" / ds, f"{args.ydisk_root}/formats/{ds}", args.delete_old)


def prepare_format(fmt: str, n_shards: int) -> None:
    prep_fmt = "datasetfs" if fmt == "datasetfs-rgb" else fmt
    run([
        sys.executable, "-m", "scripts.datasets.prepare_formats", "publaynet",
        "--n-shards", str(n_shards), "--formats", prep_fmt,
    ])


def format_dir(fmt: str) -> Path:
    return PUB_FORMATS / ("datasetfs" if fmt == "datasetfs-rgb" else fmt)


def shard_count(fmt: str) -> int:
    root = format_dir(fmt)
    if fmt == "webdataset":
        return len(list(root.glob("shard-*.tar")))
    return len([p for p in root.glob("shard_*") if p.name != "shard_-1"])


@contextmanager
def http_origin(root: Path, port: int) -> Iterator[str]:
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

    class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    srv = Server(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


def base_config(fmt: str, loader: str) -> dict:
    cfg = {
        "dataset": {
            "name": "publaynet",
            "imagefolder": str(PUB_FORMATS / "imagefolder"),
            "webdataset": str(PUB_FORMATS / "webdataset"),
            "datasetfs": str(PUB_FORMATS / "datasetfs"),
        },
        "model": "simplecnn",
        "batch_size": 64,
        "num_workers": 4,
        "image_size": 96,
        "epochs": 2,
        "warmup_epochs": 1,
        "max_batches_per_epoch": 160,
        "warmup_batches": 10,
        "seeds": [0, 1, 2],
        "loaders": [loader],
        "drop_page_cache_between_cells": False,
        "daemon_binary": "bin/datasetfs",
    }
    if fmt == "datasetfs-rgb":
        cfg["loaders"] = [{
            "format": "datasetfs",
            "name": "datasetfs-rgb",
            "dfs_decode_mode": "rgb_uint8",
            "dfs_decode_parallelism": 0,
        }]
    return cfg


def write_config(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2))


def run_single(cfg: dict, out: Path) -> None:
    cfg_path = out / "config.generated.json"
    write_config(cfg_path, cfg)
    run([
        sys.executable, "-m", "benchmarks.datasetfs_bench.runner.single_run",
        "--config", str(cfg_path), "--output", str(out),
    ])


def run_remote(args: argparse.Namespace, fmt: str, out: Path, port: int) -> None:
    if fmt == "imagefolder":
        out.mkdir(parents=True, exist_ok=True)
        (out / "REMOTE_NOT_SUPPORTED.txt").write_text(
            "ImageFolder has no HTTP streaming backend in this harness; local-only.\n"
        )
        return

    loader = "webdataset" if fmt == "webdataset" else "datasetfs"
    with http_origin(format_dir(fmt), port) as base_url:
        cfg = base_config(fmt, loader)
        if fmt == "webdataset":
            per_stream = args.remote_throttle // max(1, cfg["num_workers"])
            cfg["dataset"]["webdataset_remote"] = {
                "http_base": base_url,
                "num_shards": shard_count(fmt),
                "shard_pattern": "shard-{i:06d}.tar",
                "wds_http_mode": "curl",
                "wds_curl_limit_rate": per_stream if per_stream > 0 else None,
            }
        else:
            cfg["datasetfs_remote"] = {
                "root_url": base_url,
                "cache_dir": str(out / "datasetfs_cache"),
                "prefetch_concurrency": args.prefetch_concurrency,
                "remote_throttle": args.remote_throttle,
            }
        run_single(cfg, out)


def run_local(fmt: str, out: Path) -> None:
    loader = "webdataset" if fmt == "webdataset" else "datasetfs" if fmt.startswith("datasetfs") else "imagefolder"
    run_single(base_config(fmt, loader), out)


def cleanup_publaynet_local(fmt: str, keep_local: bool) -> None:
    if keep_local:
        return
    targets = [format_dir(fmt)]
    # imagefolder is only symlinks into raw; raw is the actual large artifact.
    if fmt == "imagefolder":
        targets.append(PUB_RAW)
    # For self-contained formats, raw can be removed after the local run.
    if fmt in ("webdataset", "datasetfs", "datasetfs-rgb"):
        targets.append(PUB_RAW)
    for target in targets:
        if target.exists():
            print(f"[delete] {target}", flush=True)
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-shards", type=int, default=85)
    parser.add_argument("--formats", nargs="*", choices=FORMATS, default=list(FORMATS))
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "runs" / "publaynet_sequence")
    parser.add_argument("--ydisk-root", default="datasetfs-bench")
    parser.add_argument("--no-ydisk", action="store_true", help="do not archive/pull anything; useful for dry infrastructure tests")
    parser.add_argument("--archive-old", action="store_true", help="archive old imagenette/imagewoof/speech_commands datasets before PubLayNet")
    parser.add_argument("--delete-old", action="store_true", help="delete old datasets after successful Yandex.Disk archive")
    parser.add_argument("--keep-local", action="store_true", help="do not delete local PubLayNet artifacts after each format")
    parser.add_argument("--remote-throttle", type=int, default=50_000_000,
                        help="aggregate remote bandwidth cap in bytes/sec; DFS uses this directly, WebDataset divides it across DataLoader workers")
    parser.add_argument("--prefetch-concurrency", type=int, default=4)
    parser.add_argument("--port-base", type=int, default=8800)
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%dT%H%M%S")
    root_out = args.output_root / stamp
    root_out.mkdir(parents=True, exist_ok=True)
    (root_out / "sequence_args.json").write_text(json.dumps(vars(args), indent=2, default=str))

    if args.archive_old:
        archive_old_datasets(args)

    run(["make", "build"])

    for idx, fmt in enumerate(args.formats):
        fmt_out = root_out / fmt
        print(f"\n=== PubLayNet format {fmt} ===", flush=True)
        prepare_format(fmt, args.n_shards)

        remote_out = fmt_out / "remote"
        local_out = fmt_out / "local"
        run_remote(args, fmt, remote_out, args.port_base + idx)
        run_local(fmt, local_out)

        archive_if_exists(args, fmt_out, f"{args.ydisk_root}/runs/publaynet_sequence/{stamp}/{fmt}", False)
        if fmt == "imagefolder":
            archive_if_exists(args, PUB_RAW, f"{args.ydisk_root}/raw/publaynet", False)
        else:
            archive_if_exists(args, format_dir(fmt), f"{args.ydisk_root}/formats/publaynet/{format_dir(fmt).name}", False)

        cleanup_publaynet_local(fmt, args.keep_local)

    print(f"[done] results under {root_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
