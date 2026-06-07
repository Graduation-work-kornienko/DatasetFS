"""Concurrent training + FUSE mutation benchmark (G3/G13).

This is the benchmark version of tests/test_mutation_consistency.py: it drives
the real FUSE mutation path while a DatasetFS epoch is being drained, records
snapshot-consistency violations, throughput, mutation latency, and system usage.
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import os
import random
import shutil
import signal
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
    def __init__(self, binary: Path, root: Path, mount: Path, log_dir: Path):
        self.binary = binary
        self.root = root
        self.mount = mount
        self.log_dir = log_dir
        self.proc: subprocess.Popen | None = None
        self.log_file = None

    def start(self) -> None:
        _cleanup_tmp_files()
        _force_unmount(self.mount)
        self.mount.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"daemon-mutation-{int(time.time() * 1000)}.log"
        self.log_file = open(self.log_path, "w")
        self.proc = subprocess.Popen(
            [str(self.binary), "daemon", "--mount", str(self.mount), "--root", str(self.root)],
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


def _create_empty_datasetfs_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.jsonl").write_text(
        json.dumps({"version": "1.0", "shards_meta": {}, "files": {}}),
        encoding="utf-8",
    )


def _label_to_idx(imagefolder_root: Path) -> dict[str, int]:
    classes = sorted(p.name for p in imagefolder_root.iterdir() if p.is_dir())
    if not classes:
        raise ValueError(f"no class directories under {imagefolder_root}")
    return {c: i for i, c in enumerate(classes)}


def _image_files(imagefolder_root: Path, max_files: int | None, seed: int) -> list[tuple[Path, str, str]]:
    files_no_name: list[tuple[Path, str]] = []
    for class_dir in sorted(p for p in imagefolder_root.iterdir() if p.is_dir()):
        for p in sorted(class_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                files_no_name.append((p, class_dir.name))
    rng = random.Random(seed)
    rng.shuffle(files_no_name)
    if max_files is not None and max_files > 0:
        files_no_name = files_no_name[:max_files]
    if not files_no_name:
        raise ValueError(f"no image files found under {imagefolder_root}")
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

    (out_root / "metadata.jsonl").write_text(json.dumps(manifest), encoding="utf-8")
    return live_names


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


def _run_image_endurance(args) -> None:
    if args.imagefolder_root is None:
        raise SystemExit("--imagefolder-root is required for --scenario image_endurance")
    args.output.mkdir(parents=True, exist_ok=True)
    flat_root = args.dataset_root or (args.output / "flat_datasetfs")
    if args.prepare_flat or not flat_root.exists():
        print(f"[mutation-endurance] preparing flat DatasetFS at {flat_root}", flush=True)
        live_names = prepare_flat_image_datasetfs(
            args.imagefolder_root,
            flat_root,
            max_files=args.max_flat_files,
            shard_target_bytes=args.shard_target_mb * 1024 * 1024,
            seed=args.seed,
        )
    else:
        # Reusing a prepared root assumes logical paths are already flat.
        meta = json.loads((flat_root / "metadata.jsonl").read_text(encoding="utf-8"))
        live_names = [name for name, m in meta["files"].items() if not m.get("deleted")]

    mount = args.output / "mnt"
    daemon = MountedDaemon(args.binary, flat_root, mount, args.output / "logs")
    label_to_idx = _label_to_idx(args.imagefolder_root)
    rows: list[dict] = []
    events: list[dict] = []
    remaining = list(live_names)
    rng = random.Random(args.seed)

    daemon.start()
    sampler = SystemSampler(
        interval_s=args.sample_interval,
        track_pids=[os.getpid()] + ([daemon.pid] if daemon.pid else []),
        track_labels={"python": os.getpid(), **({"daemon": daemon.pid} if daemon.pid else {})},
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
            "seed": args.seed,
            "modality": "image",
        })
        loader.setup()
        try:
            for run_idx in range(args.training_runs):
                if len(remaining) <= args.mutations_per_run:
                    print("[mutation-endurance] stopping: not enough remaining samples", flush=True)
                    break
                rng.shuffle(remaining)
                mutation_candidates = remaining[:args.mutations_per_run]
                mutator = FixedDeleteMutator(
                    mount,
                    mutation_candidates,
                    count=args.mutations_per_run,
                    rate=args.mutation_rate,
                    seed=args.seed + run_idx,
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
                )
                mutator.join(timeout=max(10.0, args.mutations_per_run / max(args.mutation_rate, 1.0) + 5.0))
                daemon_after = daemon_metrics.snapshot()
                end_s = time.perf_counter() - bench_t0
                del dl

                deleted = set(mutator.deleted)
                remaining = [n for n in remaining if n not in deleted]
                mstats = mutator.stats
                row = {
                    "scenario": "image_endurance",
                    "train_run": run_idx,
                    "train_start_s": start_s,
                    "train_end_s": end_s,
                    "live_samples_before": len(remaining) + len(deleted),
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
                _write_rows_union(args.output / "summary.csv", rows)
                _write_rows_union(args.output / "train_events.csv", events)
        finally:
            loader.teardown()
    finally:
        sampler.stop()
        daemon.stop()

    _write_rows_union(args.output / "system_timeseries.csv", sampler.samples)
    print(f"[mutation-endurance] wrote {args.output / 'summary.csv'}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", choices=["flat_smoke", "image_endurance"], default="flat_smoke")
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
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--image-size", type=int, default=160)
    p.add_argument("--model", default="simplecnn")
    p.add_argument("--max-batches-per-train", type=int)
    p.add_argument("--warmup-batches", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if sys.platform != "darwin" or not Path("/Library/Filesystems/macfuse.fs").exists():
        raise SystemExit("mutation benchmark requires macFUSE on macOS")
    if not args.binary.exists():
        raise SystemExit(f"datasetfs binary not found: {args.binary}")
    if args.output.exists() and not args.keep_output:
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, args.output / "mutation_bench.py")

    if args.scenario == "image_endurance":
        _run_image_endurance(args)
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
