"""Synthetic pipeline memory benchmark.

This benchmark is intentionally small and synthetic: it keeps one DatasetFS
daemon alive, repeatedly creates fresh Python DatasetFS sessions, drains a fixed
dataset, and records daemon RSS. Optional bounded FUSE replacements exercise the
real mutation path without growing the logical dataset.
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import psutil
import requests

from clients.python import DatasetFS
from scripts.datasets.datasetfs_writer import DatasetFSWriter


DAEMON_URL = "http://localhost:51409"


def _identity(x):
    return x


def _bytes(x) -> bytes:
    return bytes(x)


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


class Daemon:
    def __init__(self, binary: Path, root: Path, output: Path, mount: Path | None):
        self.binary = binary
        self.root = root
        self.output = output
        self.mount = mount
        self.proc: subprocess.Popen | None = None
        self.log_file = None
        self.log_path: Path | None = None

    def start(self) -> None:
        _cleanup_tmp_files()
        argv = [str(self.binary), "daemon", "--root", str(self.root)]
        if self.mount is None:
            argv.insert(2, "--no-mount")
        else:
            _force_unmount(self.mount)
            self.mount.mkdir(parents=True, exist_ok=True)
            argv[2:2] = ["--mount", str(self.mount)]
        log_dir = self.output / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"daemon-pipeline-memory-{int(time.time() * 1000)}.log"
        self.log_file = open(self.log_path, "w")
        self.proc = subprocess.Popen(
            argv,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
        _wait_for_healthz()
        if self.mount is not None:
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
        if self.mount is not None:
            _force_unmount(self.mount)
        _cleanup_tmp_files()

    @property
    def pid(self) -> int | None:
        if self.proc is not None and self.proc.poll() is None:
            return self.proc.pid
        return None


def _process_rss(pid: int) -> int:
    try:
        return psutil.Process(pid).memory_info().rss
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0


def _slope_per_cycle(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values)) / denom


def _write_rows(path: Path, rows: Iterable[dict]) -> None:
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


def _prepare_root(root: Path, *, files: int, payload_size: int, shard_target_bytes: int) -> None:
    payload = b"x" * payload_size
    with DatasetFSWriter(root, shard_target_bytes=shard_target_bytes) as writer:
        for i in range(files):
            writer.add(f"sample_{i:04d}.bin", payload, {"label": "x"})


def _replace_files(mount: Path, cycle: int, count: int, files: int, payload_size: int) -> None:
    for j in range(count):
        idx = (cycle * count + j) % files
        name = f"sample_{idx:04d}.bin"
        data = f"replace-{cycle:05d}-{j:03d}".encode().ljust(payload_size, b"r")
        path = mount / name
        path.unlink(missing_ok=True)
        with open(path, "wb") as f:
            f.write(data)


def _drain_session(cycle: int, timeout_s: float) -> tuple[int, float]:
    start = time.perf_counter()
    ds = DatasetFS(num_workers=0, seed=cycle, decode_fn=_bytes, transform=_identity, timeout_seconds=timeout_s)
    count = sum(1 for _ in ds)
    del ds
    gc.collect()
    return count, time.perf_counter() - start


def _run_mode(args: argparse.Namespace, mode: str, root: Path, output: Path) -> tuple[list[dict], dict]:
    mount = output / f"mnt_{mode}" if mode == "bounded_replace" else None
    daemon = Daemon(args.binary, root, output, mount)
    rows: list[dict] = []
    daemon.start()
    try:
        pid = daemon.pid
        if pid is None:
            raise RuntimeError("daemon is not running")
        for cycle in range(args.cycles):
            if mode == "bounded_replace":
                _replace_files(mount, cycle, args.replacements_per_cycle, args.files, args.payload_size)
            count, elapsed_s = _drain_session(cycle, args.timeout)
            rss = _process_rss(pid)
            row = {
                "scenario": "pipeline_memory",
                "mode": mode,
                "cycle": cycle,
                "samples": count,
                "drain_elapsed_s": elapsed_s,
                "samples_per_s": count / elapsed_s if elapsed_s > 0 else 0.0,
                "daemon_rss_bytes": rss,
                "daemon_rss_mib": rss / (1024 * 1024),
                "replacements": args.replacements_per_cycle if mode == "bounded_replace" else 0,
            }
            rows.append(row)
            _write_rows(output / "memory_timeseries.csv", rows)
            print(
                f"[pipeline-memory] mode={mode} cycle={cycle:03d} samples={count} "
                f"rss={row['daemon_rss_mib']:.1f} MiB elapsed={elapsed_s:.3f}s",
                flush=True,
            )
    finally:
        daemon.stop()

    warm = rows[min(args.warmup_cycles, len(rows)):]
    rss_values = [float(r["daemon_rss_bytes"]) for r in warm]
    elapsed_values = [float(r["drain_elapsed_s"]) for r in warm]
    summary = {
        "scenario": "pipeline_memory",
        "mode": mode,
        "cycles": len(rows),
        "warmup_cycles": args.warmup_cycles,
        "files": args.files,
        "payload_size": args.payload_size,
        "replacements_per_cycle": args.replacements_per_cycle if mode == "bounded_replace" else 0,
        "rss_min_mib": min(rss_values) / (1024 * 1024) if rss_values else 0.0,
        "rss_max_mib": max(rss_values) / (1024 * 1024) if rss_values else 0.0,
        "rss_growth_mib": (max(rss_values) - min(rss_values)) / (1024 * 1024) if rss_values else 0.0,
        "rss_slope_mib_per_cycle": _slope_per_cycle(rss_values) / (1024 * 1024) if rss_values else 0.0,
        "drain_elapsed_mean_s": sum(elapsed_values) / len(elapsed_values) if elapsed_values else 0.0,
    }
    return rows, summary


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.clean_output:
        for child in args.output.iterdir():
            if child.is_dir():
                import shutil
                shutil.rmtree(child)
            else:
                child.unlink()
    if not args.binary.exists():
        raise SystemExit(f"datasetfs binary not found: {args.binary}")
    if "bounded_replace" in args.modes and (sys.platform != "darwin" or not Path("/Library/Filesystems/macfuse.fs").exists()):
        raise SystemExit("bounded_replace mode requires macFUSE on macOS")

    root = args.output / "synthetic_datasetfs"
    if root.exists():
        import shutil
        shutil.rmtree(root)
    _prepare_root(root, files=args.files, payload_size=args.payload_size, shard_target_bytes=args.shard_target_kb * 1024)

    all_rows: list[dict] = []
    summaries: list[dict] = []
    for mode in args.modes:
        rows, summary = _run_mode(args, mode, root, args.output)
        all_rows.extend(rows)
        summaries.append(summary)
        _write_rows(args.output / "memory_timeseries.csv", all_rows)
        _write_rows(args.output / "summary.csv", summaries)
    print(f"[pipeline-memory] wrote {args.output / 'summary.csv'}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--binary", type=Path, default=Path("bin/datasetfs"))
    p.add_argument("--modes", nargs="+", default=["no_mutation", "bounded_replace"], choices=["no_mutation", "bounded_replace"])
    p.add_argument("--cycles", type=int, default=80)
    p.add_argument("--warmup-cycles", type=int, default=10)
    p.add_argument("--files", type=int, default=9)
    p.add_argument("--payload-size", type=int, default=2048)
    p.add_argument("--shard-target-kb", type=int, default=64)
    p.add_argument("--replacements-per-cycle", type=int, default=3)
    p.add_argument("--timeout", type=float, default=1.0)
    p.add_argument("--clean-output", action="store_true")
    return p.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
