"""Concurrent training + FUSE mutation benchmark (G3/G13).

This is the benchmark version of tests/test_mutation_consistency.py: it drives
the real FUSE mutation path while a DatasetFS epoch is being drained, records
snapshot-consistency violations, throughput, mutation latency, and system usage.
"""
from __future__ import annotations

import argparse
import csv
import functools
import io
import os
import random
import shutil
import signal
import struct
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
import torch
import torch.nn as nn

from benchmarks.datasetfs_bench.loaders.datasetfs import DatasetFSLoader
from benchmarks.datasetfs_bench.metrics import daemon as daemon_metrics
from benchmarks.datasetfs_bench.metrics.system import SystemSampler
from benchmarks.datasetfs_bench.models.registry import build_model
from benchmarks.datasetfs_bench.train.loop import train_one_epoch
from clients.python import DatasetFS
from scripts.datasets.datasetfs_writer import read_parquet_manifest, write_parquet_manifest


DAEMON_URL = "http://localhost:51409"


def _decode_bytes(x) -> bytes:
    return bytes(x)


def _identity(x):
    return x


def _payload(seq: int, size: int) -> bytes:
    prefix = f"DATASETFS-MUT-{seq:08d}|".encode()
    return prefix + bytes([seq % 251]) * max(0, size - len(prefix))


def _write_on_mount(mount: Path, name: str, data: bytes) -> None:
    with open(mount / name, "wb") as f:
        f.write(data)


def _force_unmount(mount: Path) -> None:
    try:
        if not os.path.ismount(mount):
            return
    except OSError:
        pass
    for cmd in (["umount", str(mount)], ["diskutil", "unmount", "force", str(mount)]):
        try:
            subprocess.run(cmd, capture_output=True, timeout=15)
        except Exception:
            pass
        try:
            if not os.path.ismount(mount):
                return
        except OSError:
            return


def _wait_for_mount(mount: Path, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if os.path.ismount(mount):
                return
        except OSError:
            pass
        time.sleep(0.1)
    raise RuntimeError(f"FUSE mount did not appear: {mount}")


def _wait_for_healthz(timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    last: Exception | None = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{DAEMON_URL}/healthz", timeout=1)
            if r.status_code == 200:
                return
        except Exception as e:
            last = e
        time.sleep(0.1)
    raise RuntimeError(f"daemon /healthz did not respond: {last}")


class MountedDaemon:
    def __init__(self, binary: Path, root: Path, mount: Path, log_dir: Path, *,
                 no_wal: bool = False, wal_format: str = "binary",
                 auto_vacuum: bool = False, vacuum_interval: str | None = None,
                 vacuum_threshold: float | None = None, vacuum_throttle: int | None = None):
        self.binary = binary
        self.root = root
        self.mount = mount
        self.log_dir = log_dir
        self.no_wal = no_wal
        self.wal_format = wal_format
        self.auto_vacuum = auto_vacuum
        self.vacuum_interval = vacuum_interval
        self.vacuum_threshold = vacuum_threshold
        self.vacuum_throttle = vacuum_throttle
        self.proc: subprocess.Popen | None = None
        self.log_file = None

    def start(self) -> None:
        _cleanup_tmp_files()
        _force_unmount(self.mount)
        self.mount.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"daemon-mutation-{int(time.time() * 1000)}.log"
        self.log_file = open(self.log_path, "w")
        argv = [str(self.binary), "daemon", "--mount", str(self.mount), "--root", str(self.root)]
        if self.no_wal:
            argv.append("--no-wal")
        else:
            argv += ["--wal-format", self.wal_format]
        if self.auto_vacuum:
            argv.append("--auto-vacuum")
            if self.vacuum_interval:
                argv += ["--vacuum-interval", self.vacuum_interval]
            if self.vacuum_threshold is not None:
                argv += ["--vacuum-threshold", str(self.vacuum_threshold)]
            if self.vacuum_throttle is not None:
                argv += ["--vacuum-throttle", str(self.vacuum_throttle)]
        self.proc = subprocess.Popen(
            argv,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
        _wait_for_healthz()
        _wait_for_mount(self.mount)

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                else:
                    self.proc.terminate()
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                else:
                    self.proc.kill()
                self.proc.wait(timeout=5)
        self.proc = None
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
        _force_unmount(self.mount)
        _cleanup_tmp_files()

    @property
    def pid(self) -> int | None:
        if self.proc is not None and self.proc.poll() is None:
            return self.proc.pid
        return None


def _cleanup_tmp_files() -> None:
    for path in ["/tmp/mlfs_data.bin", "/tmp/mlfs_refs.bin"]:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    for fifo in Path("/tmp").glob("datasetfs_pipe_*"):
        try:
            fifo.unlink()
        except FileNotFoundError:
            pass


@dataclass
class MutatorStats:
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    latency_sum_s: float = 0.0
    latency_max_s: float = 0.0


@dataclass(frozen=True)
class PlannedMutation:
    name: str
    data: bytes


@dataclass(frozen=True)
class ImagePayload:
    label: str
    data: bytes


class Mutator(threading.Thread):
    def __init__(self, mount: Path, mode: str, rate: float, payload_size: int,
                 live: set[str], live_lock: threading.Lock, stop_event: threading.Event, seed: int):
        super().__init__(daemon=True)
        self.mount = mount
        self.mode = mode
        self.rate = rate
        self.payload_size = payload_size
        self.live = live
        self.live_lock = live_lock
        self.stop_event = stop_event
        self.rng = random.Random(seed)
        self.stats = MutatorStats()
        self.seq = 10_000_000 + seed

    def run(self) -> None:
        if self.rate <= 0 or self.mode == "none":
            return
        next_at = time.perf_counter()
        while not self.stop_event.is_set():
            now = time.perf_counter()
            if now < next_at:
                self.stop_event.wait(next_at - now)
                continue
            self._mutate_once()
            next_at += 1.0 / self.rate

    def _choose_op(self) -> str:
        if self.mode == "mixed":
            return self.rng.choice(["add", "delete", "replace"])
        return self.mode

    def _mutate_once(self) -> None:
        op = self._choose_op()
        start = time.perf_counter()
        self.stats.attempted += 1
        try:
            if op == "add":
                name = f"m{self.seq:08d}"
                self.seq += 1
                _write_on_mount(self.mount, name, _payload(self.seq, self.payload_size))
                with self.live_lock:
                    self.live.add(name)
            elif op == "delete":
                with self.live_lock:
                    choices = list(self.live)
                if choices:
                    name = self.rng.choice(choices)
                    os.remove(self.mount / name)
                    with self.live_lock:
                        self.live.discard(name)
            elif op == "replace":
                with self.live_lock:
                    choices = list(self.live)
                if choices:
                    name = self.rng.choice(choices)
                    self.seq += 1
                    os.remove(self.mount / name)
                    _write_on_mount(self.mount, name, _payload(self.seq, self.payload_size))
            else:
                raise ValueError(f"unknown mutation op {op!r}")
            self.stats.succeeded += 1
        except Exception:
            self.stats.failed += 1
        finally:
            elapsed = time.perf_counter() - start
            self.stats.latency_sum_s += elapsed
            self.stats.latency_max_s = max(self.stats.latency_max_s, elapsed)


class FixedDeleteMutator(threading.Thread):
    """Delete a fixed number of existing labeled samples through FUSE."""

    def __init__(self, mount: Path, candidates: list[str], count: int, rate: float, seed: int):
        super().__init__(daemon=True)
        self.mount = mount
        self.candidates = list(candidates)
        self.count = count
        self.rate = rate
        self.rng = random.Random(seed)
        self.stats = MutatorStats()
        self.deleted: list[str] = []

    def run(self) -> None:
        interval = (1.0 / self.rate) if self.rate > 0 else 0.0
        self.rng.shuffle(self.candidates)
        for name in self.candidates[:self.count]:
            start = time.perf_counter()
            self.stats.attempted += 1
            try:
                os.remove(self.mount / name)
                self.deleted.append(name)
                self.stats.succeeded += 1
            except Exception:
                self.stats.failed += 1
            finally:
                elapsed = time.perf_counter() - start
                self.stats.latency_sum_s += elapsed
                self.stats.latency_max_s = max(self.stats.latency_max_s, elapsed)
            if interval > 0:
                time.sleep(interval)


class FixedCountImageMutator(threading.Thread):
    """Apply a fixed number of add/delete/replace operations with valid image bytes."""

    def __init__(self, mount: Path, mode: str, count: int, rate: float,
                 live: set[str], live_lock: threading.Lock,
                 payloads: list[ImagePayload], seed: int):
        super().__init__(daemon=True)
        self.mount = mount
        self.mode = mode
        self.count = count
        self.rate = rate
        self.live = live
        self.live_lock = live_lock
        self.payloads = payloads
        self.rng = random.Random(seed)
        self.stats = MutatorStats()
        self.seq = 20_000_000 + seed

    def run(self) -> None:
        interval = (1.0 / self.rate) if self.rate > 0 else 0.0
        for _ in range(self.count):
            self._mutate_once()
            if interval > 0:
                time.sleep(interval)

    def _choose_op(self) -> str:
        if self.mode == "mixed":
            return self.rng.choice(["add", "delete", "replace"])
        return self.mode

    def _choose_existing(self) -> str | None:
        with self.live_lock:
            choices = list(self.live)
        return self.rng.choice(choices) if choices else None

    def _mutate_once(self) -> None:
        op = self._choose_op()
        payload = self.rng.choice(self.payloads)
        start = time.perf_counter()
        self.stats.attempted += 1
        try:
            if op == "add":
                name = f"m{self.seq:08d}__{payload.label}__added.jpg"
                self.seq += 1
                _write_on_mount(self.mount, name, payload.data)
                with self.live_lock:
                    self.live.add(name)
            elif op == "delete":
                name = self._choose_existing()
                if name is not None:
                    os.remove(self.mount / name)
                    with self.live_lock:
                        self.live.discard(name)
            elif op == "replace":
                name = self._choose_existing()
                if name is not None:
                    os.remove(self.mount / name)
                    _write_on_mount(self.mount, name, payload.data)
            else:
                raise ValueError(f"unknown mutation op {op!r}")
            self.stats.succeeded += 1
        except Exception:
            self.stats.failed += 1
        finally:
            elapsed = time.perf_counter() - start
            self.stats.latency_sum_s += elapsed
            self.stats.latency_max_s = max(self.stats.latency_max_s, elapsed)


def _create_empty_datasetfs_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    write_parquet_manifest(root, {"version": "1.0", "shards_meta": {}, "files": {}})


def _copy_file_for_benchmark(src: str, dst: str) -> str:
    clonefile = getattr(os, "clonefile", None)
    if clonefile is not None:
        try:
            clonefile(src, dst)
            return dst
        except OSError:
            pass
    return shutil.copy2(src, dst)


def _copytree_for_benchmark(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, copy_function=_copy_file_for_benchmark)


def _vacuum_datasetfs_root(args, root: Path) -> None:
    if not args.vacuum_after_run:
        return
    subprocess.run(
        [str(args.binary), "vacuum", "--root", str(root), "--max-shard-size", str(args.shard_target_mb * 1024 * 1024)],
        check=True,
    )


def _cleanup_work_root(args, root: Path) -> None:
    if args.cleanup_work_dir:
        shutil.rmtree(root, ignore_errors=True)


def _prepared_format_names(base_imagefolder: Path, base_webdataset: Path) -> list[str]:
    web_index = base_webdataset / "webdataset_index.csv"
    if web_index.exists():
        return [row["name"] for row in _read_rows(web_index)]
    return sorted(p.name for p in base_imagefolder.iterdir() if p.is_file())


def _label_to_idx(imagefolder_root: Path) -> dict[str, int]:
    classes = sorted(p.name for p in imagefolder_root.iterdir() if p.is_dir())
    if not classes:
        raise ValueError(f"no class directories under {imagefolder_root}")
    return {c: i for i, c in enumerate(classes)}


def _label_from_flat_name(name: str) -> str:
    parts = name.split("__", 2)
    if len(parts) < 3:
        raise ValueError(f"flat image name does not contain a label: {name}")
    return parts[1]


def _load_image_payloads(mount: Path, names: list[str], limit: int, seed: int) -> list[ImagePayload]:
    rng = random.Random(seed)
    candidates = list(names)
    rng.shuffle(candidates)
    payloads: list[ImagePayload] = []
    for name in candidates:
        try:
            payloads.append(ImagePayload(label=_label_from_flat_name(name), data=(mount / name).read_bytes()))
        except Exception:
            continue
        if len(payloads) >= limit:
            break
    if not payloads:
        raise ValueError("could not load any image payloads for add/replace mutations")
    return payloads


FORMAT_MUTATION_SUFFIXES = {".jpg", ".jpeg", ".png", ".wav"}


def _image_files(imagefolder_root: Path, max_files: int | None, seed: int) -> list[tuple[Path, str, str]]:
    files_no_name: list[tuple[Path, str]] = []
    for class_dir in sorted(p for p in imagefolder_root.iterdir() if p.is_dir()):
        for p in sorted(class_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in FORMAT_MUTATION_SUFFIXES:
                files_no_name.append((p, class_dir.name))
    rng = random.Random(seed)
    rng.shuffle(files_no_name)
    if max_files is not None and max_files > 0:
        files_no_name = files_no_name[:max_files]
    if not files_no_name:
        raise ValueError(f"no image/audio files found under {imagefolder_root}")
    files = [
        (p, label, f"{idx:08d}__{label}__{p.name}")
        for idx, (p, label) in enumerate(files_no_name)
    ]
    return files


def prepare_flat_image_datasetfs(
    imagefolder_root: Path,
    out_root: Path,
    *,
    max_files: int | None,
    shard_target_bytes: int,
    seed: int,
) -> list[str]:
    """Build a flat, labeled DatasetFS root from an ImageFolder tree.

    The FUSE implementation is currently flat-name only, so this benchmark uses
    logical paths like ``class__file.jpg`` while preserving labels in metadata.
    """
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    files = _image_files(imagefolder_root, max_files=max_files, seed=seed)
    manifest = {"version": "1.0", "shards_meta": {}, "files": {}}
    live_names: list[str] = []

    shard_id = 0
    shard_entries = []
    shard_bytes = 0

    def flush(entries: list[tuple[Path, str, str, int]]) -> None:
        nonlocal shard_id
        if not entries:
            return
        tar_path = out_root / f"shard_{shard_id}"
        current = 0
        objects = []
        with tarfile.open(tar_path, "w") as tf:
            for src, label, flat_name, size in entries:
                info = tarfile.TarInfo(name=flat_name)
                info.size = size
                info.mode = 0o600
                with open(src, "rb") as f:
                    tf.addfile(info, f)
                offset = current + 512
                padding = (512 - (size % 512)) % 512
                current += 512 + size + padding
                meta = {
                    "c_id": shard_id,
                    "offset": offset,
                    "size": size,
                    "deleted": False,
                    "path": flat_name,
                    "meta": {"label": label},
                }
                manifest["files"][flat_name] = meta
                objects.append(meta)
                live_names.append(flat_name)
        manifest["shards_meta"][str(shard_id)] = {
            "type": "base",
            "total_size": current,
        }
        shard_id += 1

    for src, label, flat_name in files:
        size = src.stat().st_size
        if shard_entries and shard_bytes + size > shard_target_bytes:
            flush(shard_entries)
            shard_entries = []
            shard_bytes = 0
        shard_entries.append((src, label, flat_name, size))
        shard_bytes += size
    flush(shard_entries)

    write_parquet_manifest(out_root, manifest)
    return live_names


def prepare_flat_imagefolder(
    imagefolder_root: Path,
    out_root: Path,
    *,
    max_files: int | None,
    seed: int,
) -> list[str]:
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    files = _image_files(imagefolder_root, max_files=max_files, seed=seed)
    names: list[str] = []
    for src, _label, flat_name in files:
        _copy_file_for_benchmark(str(src), str(out_root / flat_name))
        names.append(flat_name)
    return names


def prepare_flat_webdataset(
    imagefolder_root: Path,
    out_root: Path,
    *,
    max_files: int | None,
    shard_target_bytes: int,
    seed: int,
) -> list[str]:
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    files = _image_files(imagefolder_root, max_files=max_files, seed=seed)
    names: list[str] = []
    index_rows: list[dict] = []
    shard_id = 0
    shard_entries: list[tuple[Path, str]] = []
    shard_bytes = 0

    def flush(entries: list[tuple[Path, str]]) -> None:
        nonlocal shard_id
        if not entries:
            return
        shard_name = f"shard_{shard_id:06d}.tar"
        with tarfile.open(out_root / shard_name, "w") as tf:
            for src, flat_name in entries:
                info = tarfile.TarInfo(name=flat_name)
                info.size = src.stat().st_size
                info.mode = 0o600
                with open(src, "rb") as f:
                    tf.addfile(info, f)
                index_rows.append({"name": flat_name, "shard": shard_name})
                names.append(flat_name)
        shard_id += 1

    for src, _label, flat_name in files:
        size = src.stat().st_size
        if shard_entries and shard_bytes + size > shard_target_bytes:
            flush(shard_entries)
            shard_entries = []
            shard_bytes = 0
        shard_entries.append((src, flat_name))
        shard_bytes += size
    flush(shard_entries)
    _write_rows_union(out_root / "webdataset_index.csv", index_rows)
    return names


def _drain_epoch(timeout_s: float, start_after_first: threading.Thread | None = None,
                 per_sample_delay_s: float = 0.0) -> tuple[list[str], dict[str, bytes], set[int], float]:
    start = time.perf_counter()
    ds = DatasetFS(num_workers=0, decode_fn=_decode_bytes, transform=_identity, timeout_seconds=timeout_s)
    paths: list[str] = []
    contents: dict[str, bytes] = {}
    generations: set[int] = set()
    started = False
    for sample in ds:
        path = sample["path"]
        paths.append(path)
        contents[path] = bytes(sample["image"])
        generations.add(sample["dfs_generation"])
        if start_after_first is not None and not started:
            start_after_first.start()
            started = True
        if per_sample_delay_s > 0:
            time.sleep(per_sample_delay_s)
    return paths, contents, generations, time.perf_counter() - start


def _write_rows(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_rows_union(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _run_case(args, mode: str, rate: float, repeat: int, out_dir: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="datasetfs-mut-") as td:
        tmp = Path(td)
        root = tmp / "flat_ds"
        mount = tmp / "mnt"
        _create_empty_datasetfs_root(root)
        daemon = MountedDaemon(args.binary, root, mount, out_dir / "logs")
        daemon.start()
        try:
            payloads: dict[str, bytes] = {}
            live = set()
            for i in range(args.files):
                name = f"f{i:08d}"
                payload = _payload(i, args.payload_size)
                _write_on_mount(mount, name, payload)
                payloads[name] = payload
                live.add(name)

            baseline_paths, baseline_contents, baseline_gens, _ = _drain_epoch(args.timeout)
            expected_snapshot = set(baseline_paths)
            expected_gen = next(iter(baseline_gens)) if baseline_gens else -1

            live_lock = threading.Lock()
            stop_event = threading.Event()
            mutator = Mutator(
                mount, mode, rate, args.payload_size, live, live_lock, stop_event,
                seed=args.seed + repeat,
            )
            sampler = SystemSampler(
                interval_s=args.sample_interval,
                track_pids=[os.getpid()] + ([daemon.pid] if daemon.pid else []),
                track_labels={"python": os.getpid(), **({"daemon": daemon.pid} if daemon.pid else {})},
            )
            sampler.start()
            paths, contents, generations, elapsed_s = _drain_epoch(
                args.timeout,
                start_after_first=mutator,
                per_sample_delay_s=args.per_sample_delay,
            )
            stop_event.set()
            mutator.join(timeout=5)
            sampler.stop()

            seen = set(paths)
            duplicate_count = len(paths) - len(seen)
            missing_count = len(expected_snapshot - seen)
            leaked_count = len(seen - expected_snapshot)
            torn_generation = 0 if len(generations) == 1 else 1
            wrong_generation = 0 if generations == {expected_gen} else 1
            wrong_content = 0
            for name in expected_snapshot & seen:
                if baseline_contents.get(name) != contents.get(name):
                    wrong_content += 1

            stats = mutator.stats
            row = {
                "mode": mode,
                "mutation_rate_s": rate,
                "repeat": repeat,
                "files_initial": args.files,
                "payload_size": args.payload_size,
                "epoch_samples": len(paths),
                "elapsed_s": elapsed_s,
                "samples_per_s": len(paths) / elapsed_s if elapsed_s > 0 else 0.0,
                "generations": ";".join(str(g) for g in sorted(generations)),
                "expected_generation": expected_gen,
                "duplicate_count": duplicate_count,
                "missing_count": missing_count,
                "leaked_count": leaked_count,
                "wrong_content_count": wrong_content,
                "torn_generation": torn_generation,
                "wrong_generation": wrong_generation,
                "consistency_violations": duplicate_count + missing_count + leaked_count + wrong_content + torn_generation + wrong_generation,
                "mutations_attempted": stats.attempted,
                "mutations_succeeded": stats.succeeded,
                "mutations_failed": stats.failed,
                "mutation_latency_mean_ms": (stats.latency_sum_s / stats.attempted * 1000.0) if stats.attempted else 0.0,
                "mutation_latency_max_ms": stats.latency_max_s * 1000.0,
                "daemon_log": str(daemon.log_path),
            }
            row.update(sampler.summary())
            return row
        finally:
            daemon.stop()


def _load_live_names(flat_root: Path) -> list[str]:
    meta = read_parquet_manifest(flat_root)
    return [name for name, m in meta["files"].items() if not m.get("deleted")]


def _run_image_endurance_case(args, *, flat_root: Path, output: Path, scenario_name: str,
                              no_wal: bool = False, wal_format: str = "binary",
                              auto_vacuum: bool = False, seed: int | None = None,
                              repeat: int | None = None, order_index: int | None = None) -> tuple[list[dict], list[dict]]:
    output.mkdir(parents=True, exist_ok=True)
    live_names = _load_live_names(flat_root)
    mount = output / "mnt"
    daemon = MountedDaemon(
        args.binary,
        flat_root,
        mount,
        output / "logs",
        no_wal=no_wal,
        wal_format=wal_format,
        auto_vacuum=auto_vacuum,
        vacuum_interval=args.vacuum_interval,
        vacuum_threshold=args.vacuum_threshold,
        vacuum_throttle=args.vacuum_throttle,
    )
    label_to_idx = _label_to_idx(args.imagefolder_root)
    rows: list[dict] = []
    events: list[dict] = []
    remaining = list(live_names)
    live = set(live_names)
    live_lock = threading.Lock()
    case_seed = args.seed if seed is None else seed
    rng = random.Random(case_seed)

    daemon.start()
    sampler = SystemSampler(
        interval_s=args.sample_interval,
        track_pids=[os.getpid()] + ([daemon.pid] if daemon.pid else []),
        track_labels={"python": os.getpid(), **({"daemon": daemon.pid} if daemon.pid else {})},
        disk_path=flat_root,
    )
    bench_t0 = time.perf_counter()
    sampler.start()
    try:
        loader = DatasetFSLoader({
            "root": str(flat_root),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "image_size": args.image_size,
            "label_to_idx": label_to_idx,
            "seed": case_seed,
            "modality": "image",
        })
        loader.setup()
        try:
            for run_idx in range(args.training_runs):
                if len(remaining) <= args.mutations_per_run:
                    print("[mutation-endurance] stopping: not enough remaining samples", flush=True)
                    break
                live_before = len(remaining)
                rng.shuffle(remaining)
                mutation_candidates = remaining[:args.mutations_per_run]
                if args.endurance_mutation_mode == "fixed_delete":
                    mutator = FixedDeleteMutator(
                        mount,
                        mutation_candidates,
                        count=args.mutations_per_run,
                        rate=args.mutation_rate,
                        seed=case_seed + run_idx,
                    )
                else:
                    payloads = _load_image_payloads(mount, remaining, args.mutation_payload_pool, case_seed + run_idx)
                    mutator = FixedCountImageMutator(
                        mount=mount,
                        mode=args.endurance_mutation_mode,
                        count=args.mutations_per_run,
                        rate=args.mutation_rate,
                        live=live,
                        live_lock=live_lock,
                        payloads=payloads,
                        seed=case_seed + run_idx,
                    )

                model = build_model(args.model, num_classes=len(label_to_idx))
                optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
                loss_fn = nn.CrossEntropyLoss()
                dl = loader.make_loader()
                start_s = time.perf_counter() - bench_t0
                daemon_before = daemon_metrics.snapshot()
                print(
                    f"[mutation-endurance] train={run_idx} start={start_s:.2f}s "
                    f"remaining={len(remaining)} mutate={args.mutations_per_run}",
                    flush=True,
                )
                stats = train_one_epoch(
                    model,
                    dl,
                    optim,
                    loss_fn,
                    epoch_idx=run_idx,
                    max_batches=args.max_batches_per_train,
                    warmup_batches=args.warmup_batches,
                    after_first_batch=mutator.start,
                    batch_delay_s=args.train_batch_delay,
                )
                mutator.join(timeout=max(10.0, args.mutations_per_run / max(args.mutation_rate, 1.0) + 5.0))
                daemon_after = daemon_metrics.snapshot()
                end_s = time.perf_counter() - bench_t0
                del dl

                deleted = set(getattr(mutator, "deleted", []))
                if args.endurance_mutation_mode == "fixed_delete":
                    with live_lock:
                        live.difference_update(deleted)
                with live_lock:
                    remaining = list(live)
                mstats = mutator.stats
                row = {
                    "scenario": "image_endurance",
                    "vacuum_scenario": scenario_name,
                    "wal_enabled": not no_wal,
                    "wal_format": "none" if no_wal else wal_format,
                    "auto_vacuum": auto_vacuum,
                    "repeat": repeat,
                    "order_index": order_index,
                    "seed": case_seed,
                    "train_run": run_idx,
                    "train_start_s": start_s,
                    "train_end_s": end_s,
                    "endurance_mutation_mode": args.endurance_mutation_mode,
                    "live_samples_before": live_before,
                    "live_samples_after": len(remaining),
                    "mutations_requested": args.mutations_per_run,
                    "mutations_attempted": mstats.attempted,
                    "mutations_succeeded": mstats.succeeded,
                    "mutations_failed": mstats.failed,
                    "mutation_latency_mean_ms": (mstats.latency_sum_s / mstats.attempted * 1000.0) if mstats.attempted else 0.0,
                    "mutation_latency_max_ms": mstats.latency_max_s * 1000.0,
                }
                row.update(stats.summary())
                row.update({f"epoch_{k}": v for k, v in daemon_metrics.cell_summary(daemon_before, daemon_after).items()})
                rows.append(row)
                events.append({"train_run": run_idx, "start_s": start_s, "end_s": end_s})
                _write_rows_union(output / "summary.csv", rows)
                _write_rows_union(output / "train_events.csv", events)
        finally:
            loader.teardown()
    finally:
        sampler.stop()
        daemon.stop()

    system_summary = sampler.summary()
    for row in rows:
        row.update(system_summary)

    for sample in sampler.samples:
        sample["vacuum_scenario"] = scenario_name
        sample["wal_format"] = "none" if no_wal else wal_format
        sample["auto_vacuum"] = auto_vacuum
        sample["repeat"] = repeat
        sample["order_index"] = order_index
        sample["seed"] = case_seed
    _write_rows_union(output / "summary.csv", rows)
    _write_rows_union(output / "system_timeseries.csv", sampler.samples)
    print(f"[mutation-endurance] wrote {output / 'summary.csv'}", flush=True)
    return rows, sampler.samples


def _prepare_or_load_flat_root(args, flat_root: Path) -> None:
    if args.prepare_flat or not flat_root.exists():
        print(f"[mutation-endurance] preparing flat DatasetFS at {flat_root}", flush=True)
        prepare_flat_image_datasetfs(
            args.imagefolder_root,
            flat_root,
            max_files=args.max_flat_files,
            shard_target_bytes=args.shard_target_mb * 1024 * 1024,
            seed=args.seed,
        )


def _planned_replacements(names: list[str], count: int, payload_size: int, seed: int) -> list[PlannedMutation]:
    if count <= 0:
        return []
    if count > len(names):
        raise ValueError(f"changed_files={count} exceeds available files={len(names)}")
    rng = random.Random(seed)
    chosen = rng.sample(names, count)
    return [PlannedMutation(name=name, data=_payload(seed + i, payload_size)) for i, name in enumerate(chosen)]


def _replace_regular_files(root: Path, mutations: list[PlannedMutation]) -> None:
    for mutation in mutations:
        path = root / mutation.name
        path.unlink(missing_ok=True)
        with open(path, "wb") as f:
            f.write(mutation.data)


def _read_webdataset_index(root: Path) -> dict[str, str]:
    rows = _read_rows(root / "webdataset_index.csv")
    return {row["name"]: row["shard"] for row in rows}


def _replace_webdataset_files(root: Path, mutations: list[PlannedMutation]) -> None:
    index = _read_webdataset_index(root)
    replacements = {m.name: m.data for m in mutations}
    by_shard: dict[str, set[str]] = {}
    for name in replacements:
        shard = index.get(name)
        if shard is None:
            raise FileNotFoundError(f"{name} not found in webdataset index")
        by_shard.setdefault(shard, set()).add(name)

    for shard_name, names in by_shard.items():
        shard_path = root / shard_name
        tmp_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
        seen: set[str] = set()
        with tarfile.open(shard_path, "r") as src_tf, tarfile.open(tmp_path, "w") as dst_tf:
            for member in src_tf.getmembers():
                data = replacements[member.name] if member.name in names else src_tf.extractfile(member).read()
                if member.name in names:
                    seen.add(member.name)
                info = tarfile.TarInfo(name=member.name)
                info.size = len(data)
                info.mode = member.mode
                dst_tf.addfile(info, io.BytesIO(data))
        missing = names - seen
        if missing:
            tmp_path.unlink(missing_ok=True)
            raise FileNotFoundError(f"missing members in {shard_name}: {sorted(missing)[:3]}")
        os.replace(tmp_path, shard_path)


def _write_uvarint(buf: bytearray, value: int) -> None:
    while value >= 0x80:
        buf.append((value & 0x7F) | 0x80)
        value >>= 7
    buf.append(value)


def _write_tx_string(buf: bytearray, value: str) -> None:
    data = value.encode("utf-8")
    _write_uvarint(buf, len(data))
    buf.extend(data)


def _write_dfstx(path: Path, ops: list[tuple[int, str, str]]) -> None:
    buf = bytearray()
    buf.extend(b"DFTX")
    buf.extend(struct.pack("<HHI", 1, 0, len(ops)))
    for op, logical_path, aux in ops:
        buf.append(op)
        _write_tx_string(buf, logical_path)
        _write_tx_string(buf, aux)
    path.write_bytes(buf)


def _replace_datasetfs_tx_files(mount: Path, mutations: list[PlannedMutation], repeat: int, changed_files: int) -> None:
    tx_id = f"bench_{changed_files}_{repeat}_{time.time_ns()}"
    tx_dir = mount / ".datasetfs" / "tx" / tx_id
    put_dir = tx_dir / "put"
    tx_dir.mkdir()
    ops: list[tuple[int, str, str]] = []
    for idx, mutation in enumerate(mutations):
        staged_name = f"payload_{idx:06d}"
        rel_staged = f"put/{staged_name}"
        (put_dir / staged_name).write_bytes(mutation.data)
        ops.append((1, mutation.name, rel_staged))
    _write_dfstx(tx_dir / "ops.dfstx", ops)
    os.rename(tx_dir, mount / ".datasetfs" / "commit" / tx_id)


def _format_mutation_row(fmt: str, changed_files: int, repeat: int, elapsed_s: float, failed: int, bytes_written: int) -> dict:
    succeeded = changed_files - failed
    return {
        "scenario": "format_mutation",
        "format": fmt,
        "operation": "replace",
        "changed_files": changed_files,
        "repeat": repeat,
        "elapsed_s": elapsed_s,
        "mean_operation_ms": (elapsed_s / changed_files * 1000.0) if changed_files else 0.0,
        "operations_succeeded": succeeded,
        "operations_failed": failed,
        "bytes_written": bytes_written,
    }


def _run_datasetfs_format_mutation(args, base_root: Path, names: list[str], mutations: list[PlannedMutation], repeat: int, changed_files: int, out_dir: Path) -> dict:
    root = out_dir / f"datasetfs_{changed_files}_{repeat}"
    if root.exists():
        shutil.rmtree(root)
    _copytree_for_benchmark(base_root, root)
    (root / "wal.log").unlink(missing_ok=True)
    mount = out_dir / f"mnt_{changed_files}_{repeat}"
    daemon = MountedDaemon(args.binary, root, mount, out_dir / "logs", wal_format="binary")
    daemon.start()
    try:
        start = time.perf_counter()
        failed = 0
        try:
            _replace_regular_files(mount, mutations)
        except Exception:
            failed = changed_files
        elapsed = time.perf_counter() - start
        row = _format_mutation_row("datasetfs", changed_files, repeat, elapsed, failed, sum(len(m.data) for m in mutations))
        row["available_files"] = len(names)
        return row
    finally:
        daemon.stop()
        _vacuum_datasetfs_root(args, root)
        _cleanup_work_root(args, root)


def _run_datasetfs_tx_format_mutation(args, base_root: Path, names: list[str], mutations: list[PlannedMutation], repeat: int, changed_files: int, out_dir: Path) -> dict:
    root = out_dir / f"datasetfs_tx_{changed_files}_{repeat}"
    if root.exists():
        shutil.rmtree(root)
    _copytree_for_benchmark(base_root, root)
    (root / "wal.log").unlink(missing_ok=True)
    mount = out_dir / f"mnt_tx_{changed_files}_{repeat}"
    daemon = MountedDaemon(args.binary, root, mount, out_dir / "logs", wal_format="binary")
    daemon.start()
    try:
        start = time.perf_counter()
        failed = 0
        try:
            _replace_datasetfs_tx_files(mount, mutations, repeat, changed_files)
        except Exception:
            failed = changed_files
        elapsed = time.perf_counter() - start
        row = _format_mutation_row("datasetfs_tx", changed_files, repeat, elapsed, failed, sum(len(m.data) for m in mutations))
        row["available_files"] = len(names)
        return row
    finally:
        daemon.stop()
        _vacuum_datasetfs_root(args, root)
        _cleanup_work_root(args, root)


def _run_directory_format_mutation(args, fmt: str, base_root: Path, names: list[str], mutations: list[PlannedMutation], repeat: int, changed_files: int, out_dir: Path) -> dict:
    root = out_dir / f"{fmt}_{changed_files}_{repeat}"
    if root.exists():
        shutil.rmtree(root)
    _copytree_for_benchmark(base_root, root)
    start = time.perf_counter()
    failed = 0
    try:
        if fmt == "imagefolder":
            _replace_regular_files(root, mutations)
        elif fmt == "webdataset":
            _replace_webdataset_files(root, mutations)
        else:
            raise ValueError(f"unknown format {fmt!r}")
    except Exception:
        failed = changed_files
    elapsed = time.perf_counter() - start
    row = _format_mutation_row(fmt, changed_files, repeat, elapsed, failed, sum(len(m.data) for m in mutations))
    row["available_files"] = len(names)
    _cleanup_work_root(args, root)
    return row


def _run_format_mutation(args) -> None:
    if args.imagefolder_root is None:
        raise SystemExit("--imagefolder-root is required for --scenario format_mutation")
    args.output.mkdir(parents=True, exist_ok=True)
    base_dir = args.prepared_dir or (args.output / "prepared")
    base_datasetfs = base_dir / "datasetfs"
    base_imagefolder = base_dir / "imagefolder"
    base_webdataset = base_dir / "webdataset"
    prepared_exists = base_datasetfs.exists() and base_imagefolder.exists() and base_webdataset.exists()
    if prepared_exists and not args.rebuild_prepared:
        print(f"[format-mutation] reusing prepared artifacts under {base_dir}", flush=True)
        names = _prepared_format_names(base_imagefolder, base_webdataset)
    else:
        print(f"[format-mutation] preparing scratch artifacts under {base_dir}", flush=True)
        names = prepare_flat_image_datasetfs(
            args.imagefolder_root,
            base_datasetfs,
            max_files=args.max_flat_files,
            shard_target_bytes=args.shard_target_mb * 1024 * 1024,
            seed=args.seed,
        )
        image_names = prepare_flat_imagefolder(args.imagefolder_root, base_imagefolder, max_files=args.max_flat_files, seed=args.seed)
        web_names = prepare_flat_webdataset(
            args.imagefolder_root,
            base_webdataset,
            max_files=args.max_flat_files,
            shard_target_bytes=args.shard_target_mb * 1024 * 1024,
            seed=args.seed,
        )
        if names != image_names or names != web_names:
            raise RuntimeError("prepared mutation format artifacts do not contain the same logical names")
    if not names:
        raise RuntimeError(f"no prepared files found under {base_dir}")

    rows: list[dict] = []
    work_dir = args.output / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    for changed_files in args.changed_files:
        for repeat in range(args.repeats):
            mutations = _planned_replacements(names, changed_files, args.payload_size, args.seed + repeat * 100_000 + changed_files)
            for fmt in args.formats:
                print(f"[format-mutation] format={fmt} changed_files={changed_files} repeat={repeat}", flush=True)
                if fmt == "datasetfs":
                    row = _run_datasetfs_format_mutation(args, base_datasetfs, names, mutations, repeat, changed_files, work_dir)
                elif fmt == "datasetfs_tx":
                    row = _run_datasetfs_tx_format_mutation(args, base_datasetfs, names, mutations, repeat, changed_files, work_dir)
                elif fmt in {"imagefolder", "webdataset"}:
                    base = base_imagefolder if fmt == "imagefolder" else base_webdataset
                    row = _run_directory_format_mutation(args, fmt, base, names, mutations, repeat, changed_files, work_dir)
                else:
                    raise SystemExit(f"unsupported --formats value: {fmt}")
                rows.append(row)
                _write_rows_union(args.output / "summary.csv", rows)
    print(f"[format-mutation] wrote {args.output / 'summary.csv'}", flush=True)


def _run_image_endurance(args) -> None:
    if args.imagefolder_root is None:
        raise SystemExit("--imagefolder-root is required for --scenario image_endurance")
    args.output.mkdir(parents=True, exist_ok=True)
    flat_root = args.dataset_root or (args.output / "flat_datasetfs")
    _prepare_or_load_flat_root(args, flat_root)
    _run_image_endurance_case(
        args,
        flat_root=flat_root,
        output=args.output,
        scenario_name="binary_wal_no_vacuum",
        no_wal=False,
        wal_format="binary",
        auto_vacuum=False,
    )


def _run_snapshot_compare(args) -> None:
    if args.imagefolder_root is None:
        raise SystemExit("--imagefolder-root is required for --scenario snapshot_compare")
    args.output.mkdir(parents=True, exist_ok=True)
    base_root = args.dataset_root or (args.output / "base_flat_datasetfs")
    _prepare_or_load_flat_root(args, base_root)

    scenarios = [
        ("binary_wal_no_vacuum", False),
        ("binary_wal_with_vacuum", True),
    ]
    all_rows: list[dict] = []
    all_samples: list[dict] = []
    for name, auto_vacuum in scenarios:
        scenario_out = args.output / name
        scenario_root = scenario_out / "flat_datasetfs"
        if scenario_root.exists():
            shutil.rmtree(scenario_root)
        _copytree_for_benchmark(base_root, scenario_root)
        (scenario_root / "wal.log").unlink(missing_ok=True)
        rows, samples = _run_image_endurance_case(
            args,
            flat_root=scenario_root,
            output=scenario_out,
            scenario_name=name,
            no_wal=False,
            wal_format="binary",
            auto_vacuum=auto_vacuum,
        )
        all_rows.extend(rows)
        all_samples.extend(samples)
        _write_rows_union(args.output / "summary.csv", all_rows)
        _write_rows_union(args.output / "system_timeseries.csv", all_samples)
    print(f"[snapshot-compare] wrote {args.output / 'summary.csv'}", flush=True)


def _run_vacuum_matrix(args) -> None:
    if args.imagefolder_root is None:
        raise SystemExit("--imagefolder-root is required for --scenario vacuum_matrix")
    args.output.mkdir(parents=True, exist_ok=True)
    base_root = args.dataset_root or (args.output / "base_flat_datasetfs")
    _prepare_or_load_flat_root(args, base_root)

    scenarios = [
        ("binary_wal_no_vacuum", False, "binary", False),
        ("binary_wal_with_vacuum", False, "binary", True),
        ("json_wal_no_vacuum", False, "json", False),
        ("json_wal_with_vacuum", False, "json", True),
    ]
    all_rows: list[dict] = []
    all_samples: list[dict] = []
    for repeat in range(args.repeats):
        run_order = list(scenarios)
        if not args.no_randomize_vacuum_order:
            random.Random(args.seed + repeat).shuffle(run_order)
        for order_index, (name, no_wal, wal_format, auto_vacuum) in enumerate(run_order):
            case_seed = args.seed + repeat * 100_000
            scenario_out = args.output / f"repeat_{repeat:02d}" / f"{order_index:02d}_{name}"
            scenario_root = scenario_out / "flat_datasetfs"
            if scenario_root.exists():
                shutil.rmtree(scenario_root)
            _copytree_for_benchmark(base_root, scenario_root)
            (scenario_root / "wal.log").unlink(missing_ok=True)
            print(
                f"[vacuum-matrix] repeat={repeat} order={order_index} scenario={name} "
                f"wal={wal_format} vacuum={auto_vacuum}",
                flush=True,
            )
            rows, samples = _run_image_endurance_case(
                args,
                flat_root=scenario_root,
                output=scenario_out,
                scenario_name=name,
                no_wal=no_wal,
                wal_format=wal_format,
                auto_vacuum=auto_vacuum,
                seed=case_seed,
                repeat=repeat,
                order_index=order_index,
            )
            all_rows.extend(rows)
            all_samples.extend(samples)
            _write_rows_union(args.output / "summary.csv", all_rows)
            _write_rows_union(args.output / "system_timeseries.csv", all_samples)
            if args.vacuum_case_cooldown_s > 0:
                time.sleep(args.vacuum_case_cooldown_s)
    print(f"[vacuum-matrix] wrote {args.output / 'summary.csv'}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", choices=["flat_smoke", "image_endurance", "snapshot_compare", "vacuum_matrix", "format_mutation"], default="flat_smoke")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--binary", type=Path, default=Path("bin/datasetfs"))
    p.add_argument("--files", type=int, default=256)
    p.add_argument("--payload-size", type=int, default=4096)
    p.add_argument("--modes", nargs="+", default=["none", "mixed", "add", "delete", "replace"])
    p.add_argument("--rates", nargs="+", type=float, default=[0, 5, 20])
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--sample-interval", type=float, default=0.2)
    p.add_argument("--per-sample-delay", type=float, default=0.0,
                   help="Optional synthetic train-step delay to keep the epoch open under mutation load.")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--keep-output", action="store_true")
    p.add_argument("--dataset-root", type=Path,
                   help="Prepared flat DatasetFS root for image_endurance; default is <output>/flat_datasetfs.")
    p.add_argument("--imagefolder-root", type=Path,
                   help="ImageFolder root used for labels and optional flat DatasetFS preparation.")
    p.add_argument("--prepare-flat", action="store_true",
                   help="Rebuild --dataset-root from --imagefolder-root before image_endurance.")
    p.add_argument("--max-flat-files", type=int,
                   help="Optional cap when preparing the flat image dataset.")
    p.add_argument("--shard-target-mb", type=int, default=256)
    p.add_argument("--training-runs", type=int, default=16)
    p.add_argument("--mutations-per-run", type=int, default=25)
    p.add_argument("--mutation-rate", type=float, default=5.0)
    p.add_argument("--endurance-mutation-mode", choices=["fixed_delete", "mixed", "add", "delete", "replace"], default="fixed_delete",
                   help="Mutation pattern for image_endurance/snapshot_compare. Use mixed for add/delete/replace.")
    p.add_argument("--mutation-payload-pool", type=int, default=256,
                   help="Number of existing image payloads sampled for add/replace mutations.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--image-size", type=int, default=160)
    p.add_argument("--model", default="simplecnn")
    p.add_argument("--max-batches-per-train", type=int)
    p.add_argument("--warmup-batches", type=int, default=1)
    p.add_argument("--train-batch-delay", type=float, default=0.0,
                   help="Optional synthetic delay after each training batch; useful to keep train longer than mutations.")
    p.add_argument("--vacuum-interval", default="1s")
    p.add_argument("--vacuum-threshold", type=float, default=0.05)
    p.add_argument("--vacuum-throttle", type=int, default=0)
    p.add_argument("--no-randomize-vacuum-order", action="store_true",
                   help="For vacuum_matrix, keep the fixed scenario order instead of shuffling scenarios inside each repeat.")
    p.add_argument("--vacuum-case-cooldown-s", type=float, default=0.0,
                   help="Optional pause between vacuum_matrix cases to reduce immediate cross-case interference.")
    p.add_argument("--changed-files", nargs="+", type=int, default=[1, 5, 10, 25, 50, 100],
                   help="Mutation counts for --scenario format_mutation; plotted on the X axis.")
    p.add_argument("--formats", nargs="+", default=["datasetfs", "datasetfs_tx", "imagefolder", "webdataset"],
                   help="Formats for --scenario format_mutation: datasetfs datasetfs_tx imagefolder webdataset.")
    p.add_argument("--prepared-dir", type=Path,
                   help="Reusable prepared flat artifacts for format_mutation; contains datasetfs, imagefolder, webdataset.")
    p.add_argument("--rebuild-prepared", action="store_true",
                   help="Rebuild --prepared-dir instead of reusing existing prepared artifacts.")
    p.add_argument("--vacuum-after-run", action="store_true",
                   help="For format_mutation, compact each DatasetFS scratch root after its measured mutation run.")
    p.add_argument("--cleanup-work-dir", action="store_true",
                   help="For format_mutation, delete each per-measurement scratch root after recording the row.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    needs_fuse = args.scenario != "format_mutation" or any(fmt in args.formats for fmt in ("datasetfs", "datasetfs_tx"))
    if needs_fuse and (sys.platform != "darwin" or not Path("/Library/Filesystems/macfuse.fs").exists()):
        raise SystemExit("mutation benchmark requires macFUSE on macOS")
    if needs_fuse and not args.binary.exists():
        raise SystemExit(f"datasetfs binary not found: {args.binary}")
    if args.output.exists() and not args.keep_output:
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, args.output / "mutation_bench.py")

    if args.scenario == "image_endurance":
        _run_image_endurance(args)
        return
    if args.scenario == "snapshot_compare":
        _run_snapshot_compare(args)
        return
    if args.scenario == "vacuum_matrix":
        _run_vacuum_matrix(args)
        return
    if args.scenario == "format_mutation":
        _run_format_mutation(args)
        return

    rows = []
    for mode in args.modes:
        rates = [0.0] if mode == "none" else args.rates
        for rate in rates:
            for repeat in range(args.repeats):
                print(f"[mutation-bench] mode={mode} rate={rate}/s repeat={repeat}", flush=True)
                row = _run_case(args, mode, rate, repeat, args.output)
                rows.append(row)
                _write_rows(args.output / "summary.csv", rows)

    print(f"[mutation-bench] wrote {args.output / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
