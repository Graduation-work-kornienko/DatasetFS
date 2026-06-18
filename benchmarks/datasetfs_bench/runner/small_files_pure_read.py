"""Pure-read small-file benchmark.

This benchmark intentionally removes model compute, audio decode, and transforms.
It measures raw object iteration/read cost for ImageFolder-style files,
WebDataset tar shards, and DatasetFS shards.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

from webdataset import ShardWriter

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.datasetfs_bench.runner.daemon_ctl import DaemonManager
from clients.python import DatasetFS


VALID_SUFFIXES = {".wav"}


def _list_balanced_samples(root: Path) -> list[tuple[Path, str]]:
    by_class: dict[str, list[Path]] = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = [
            p for p in sorted(class_dir.iterdir())
            if p.is_file() and not p.name.startswith("._") and p.suffix.lower() in VALID_SUFFIXES
        ]
        if files:
            by_class[class_dir.name] = files
    if not by_class:
        raise ValueError(f"no .wav samples under {root}")

    out: list[tuple[Path, str]] = []
    max_len = max(len(v) for v in by_class.values())
    for i in range(max_len):
        for label in sorted(by_class):
            files = by_class[label]
            if i < len(files):
                out.append((files[i], label))
    return out


def _drop_cache() -> tuple[bool, str]:
    try:
        subprocess.run(["sudo", "-n", "purge"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        time.sleep(2.0)
        return True, "sudo -n purge"
    except Exception as exc:
        return False, str(exc)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _prepare_webdataset(samples: list[tuple[Path, str]], out: Path) -> None:
    done = out / ".done"
    if done.exists():
        return
    if out.exists():
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True)
    with ShardWriter(str(out / "shard-%06d.tar"), maxsize=512 * 1024 * 1024) as sink:
        for idx, (src, label) in enumerate(samples):
            sink.write({
                "__key__": f"{idx:08d}",
                "wav": src.read_bytes(),
                "cls": label.encode("utf-8"),
            })
    done.touch()


def _read_imagefolder(samples: list[tuple[Path, str]], n: int) -> tuple[int, int, float]:
    total = 0
    first = 0.0
    t0 = time.perf_counter()
    for idx, (path, _label) in enumerate(samples[:n]):
        with path.open("rb") as f:
            data = f.read()
        if idx == 0:
            first = time.perf_counter() - t0
        total += len(data)
    return n, total, first


def _read_webdataset(root: Path, n: int) -> tuple[int, int, float]:
    count = 0
    total = 0
    first = 0.0
    t0 = time.perf_counter()
    for shard in sorted(root.glob("*.tar")):
        with tarfile.open(shard, "r:") as tf:
            for member in tf:
                if not member.isfile() or not member.name.endswith(".wav"):
                    continue
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                data = fh.read()
                if count == 0:
                    first = time.perf_counter() - t0
                total += len(data)
                count += 1
                if count >= n:
                    return count, total, first
    return count, total, first


def _identity_decode(raw: bytes) -> int:
    return len(raw)


def _identity_transform(value: int) -> int:
    return value


def _read_datasetfs(root: Path, n: int, out_dir: Path, timeout_seconds: float) -> tuple[int, int, float]:
    daemon = DaemonManager(
        binary=REPO_ROOT / "bin/datasetfs",
        root_path=root.resolve(),
        cwd=REPO_ROOT,
        log_path=out_dir / "daemon.log",
    )
    count = 0
    total = 0
    first = 0.0
    t0 = time.perf_counter()
    daemon.start()
    try:
        ds = DatasetFS(
            num_workers=0,
            decode_fn=_identity_decode,
            transform=_identity_transform,
            timeout_seconds=timeout_seconds,
        )
        for item in ds:
            if count == 0:
                first = time.perf_counter() - t0
            total += int(item["image"])
            count += 1
            if count >= n:
                break
    finally:
        daemon.stop()
    return count, total, first


def _row(loader: str, n: int, count: int, total_bytes: int, first: float, wall: float,
         cache_ok: bool, cache_detail: str) -> dict[str, Any]:
    mib = total_bytes / (1024 * 1024)
    return {
        "loader": loader,
        "sample_count": n,
        "objects_read": count,
        "total_file_bytes": total_bytes,
        "avg_file_bytes": total_bytes / count if count else 0,
        "read_wall_seconds": wall,
        "time_to_first_object": first,
        "objects_per_second": count / wall if wall else 0,
        "mib_per_second": mib / wall if wall else 0,
        "cache_drop_ok": cache_ok,
        "cache_drop_detail": cache_detail,
    }


def run(args: argparse.Namespace) -> Path:
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    needs_source_samples = "imagefolder" in args.loaders or ("webdataset" in args.loaders and not args.webdataset_root)
    samples = _list_balanced_samples(args.source_imagefolder) if needs_source_samples else []
    max_count = max(args.sample_counts)
    if needs_source_samples and max_count > len(samples):
        raise ValueError(f"requested {max_count} samples, source has only {len(samples)}")

    webdataset_root = args.webdataset_root or output / "prepared" / "webdataset_1m"
    if "webdataset" in args.loaders:
        if args.webdataset_root:
            print(f"[pure-read] reuse webdataset {webdataset_root}", flush=True)
        else:
            print(f"[pure-read] prepare webdataset max_count={max_count}", flush=True)
            _prepare_webdataset(samples[:max_count], webdataset_root)

    rows: list[dict[str, Any]] = []
    for n in args.sample_counts:
        datasetfs_root = args.datasetfs_template.format(n=n)
        readers = {
            "imagefolder": lambda out: _read_imagefolder(samples, n),
            "webdataset": lambda out: _read_webdataset(webdataset_root, n),
            "datasetfs": lambda out: _read_datasetfs(Path(datasetfs_root), n, out, args.timeout_seconds),
        }
        cells = [(loader, readers[loader]) for loader in args.loaders]
        for loader, fn in cells:
            cell_dir = output / f"n_{n:07d}" / loader
            cell_dir.mkdir(parents=True, exist_ok=True)
            cache_ok, cache_detail = _drop_cache() if args.drop_cache else (False, "disabled")
            print(f"[pure-read] n={n} loader={loader} cache_drop={cache_ok}", flush=True)
            t0 = time.perf_counter()
            count, total, first = fn(cell_dir)
            wall = time.perf_counter() - t0
            rows.append(_row(loader, n, count, total, first, wall, cache_ok, cache_detail))
            _write_csv(output / "summary.csv", rows)
            print(
                f"[pure-read] n={n} {loader} objects={count} wall={wall:.2f}s "
                f"ops={count / wall if wall else 0:.1f}/s mib={total / (1024**2) / wall if wall else 0:.1f}/s",
                flush=True,
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-imagefolder",
        type=Path,
        default=Path("data/formats/speech_commands_replicated_10x/imagefolder"),
    )
    parser.add_argument("--sample-counts", type=int, nargs="+", default=[1000, 10000, 100000, 1000000])
    parser.add_argument(
        "--datasetfs-template",
        default="runs/small_files_20260612T132744/prepared/n_{n:06d}/datasetfs",
        help="Python format string with {n}; reuses already prepared DatasetFS subsets by default.",
    )
    parser.add_argument(
        "--webdataset-root",
        type=Path,
        help="Reuse an existing WebDataset directory instead of materializing it in the output run.",
    )
    parser.add_argument(
        "--loaders",
        nargs="+",
        choices=["imagefolder", "webdataset", "datasetfs"],
        default=["imagefolder", "webdataset", "datasetfs"],
    )
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--drop-cache", action="store_true", dest="drop_cache", default=True)
    parser.add_argument("--no-drop-cache", action="store_false", dest="drop_cache")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
