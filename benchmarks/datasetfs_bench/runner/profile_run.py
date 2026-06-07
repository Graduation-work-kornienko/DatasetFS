"""Profile DatasetFS under contention to localize hotspots.

Phase 3 follow-up: workers-sweep showed daemon p50 latency growing 8× from
w=0 to w=8 (16→125 ms), and the MVP smoke showed DFS slower than WebDataset
despite serving everything from page cache. This harness captures the
profiles needed to localize *where* time is spent.

Captures, simultaneously, during one ~30 s training-shaped run:
  - **Daemon-side** (Go): CPU profile, mutex profile, block profile,
    goroutine dump, heap snapshot. Via /debug/pprof/*.
  - **Python-side**: cProfile of the iteration loop, scoped to
    DataLoader-driven work.

Usage:
    python -m benchmarks.datasetfs_bench.runner.profile_run \\
        --output profiling/<stamp> \\
        --num-workers 8 \\
        --batches 60

Then:
    /path/to/go tool pprof -top profiling/<stamp>/daemon_cpu.pb.gz
    /path/to/go tool pprof -top profiling/<stamp>/daemon_mutex.pb.gz
    python -m pstats profiling/<stamp>/python.prof  # then `sort cumulative`, `stats 30`
"""
from __future__ import annotations

import argparse
import cProfile
import functools
import os
import pstats
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clients.python import DatasetFS
from benchmarks.datasetfs_bench.runner.daemon_ctl import (
    cleanup_tmp_files,
    wait_for_healthz,
)


DAEMON_URL = "http://localhost:51409"
DEFAULT_BATCH_SIZE = 64
DEFAULT_IMAGE_SIZE = 96
GO_PATH = "/Users/true-danil-12/.ya/tools/v4/11505891785/bin/go"


def _to_rgb(img):
    return img.convert("RGB")


def _build_transform(image_size: int) -> T.Compose:
    return T.Compose([
        T.Lambda(_to_rgb),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])


def _dfs_collate(items, label_to_idx):
    images = torch.stack([it["image"] for it in items])
    labels = torch.tensor([label_to_idx[it["label"]] for it in items], dtype=torch.long)
    return images, labels


def _label_index(imagefolder_root: Path) -> dict[str, int]:
    classes = sorted(p.name for p in imagefolder_root.iterdir() if p.is_dir())
    return {c: i for i, c in enumerate(classes)}


def _start_daemon(
    binary: Path, root: Path, log_path: Path,
    mutex_rate: int, block_rate: int,
) -> subprocess.Popen:
    cleanup_tmp_files()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w")
    cmd = [
        str(binary), "daemon", "--no-mount", "--no-wal", "--root", str(root),
        f"--mutex-profile-rate={mutex_rate}",
        f"--block-profile-rate={block_rate}",
    ]
    print(f"[profile] starting daemon: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd, cwd=REPO_ROOT,
        stdout=log_file, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )
    wait_for_healthz(DAEMON_URL, timeout=30.0)
    print(f"[profile] daemon ready (pid={proc.pid}, log={log_path})", flush=True)
    return proc


def _stop_daemon(proc: subprocess.Popen) -> None:
    import signal
    if proc.poll() is not None:
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    cleanup_tmp_files()


def _capture_cpu_profile(out_path: Path, seconds: int) -> None:
    """Hit /debug/pprof/profile?seconds=N — blocks for N seconds."""
    url = f"{DAEMON_URL}/debug/pprof/profile?seconds={seconds}"
    print(f"[profile] CPU profile starting ({seconds} s) → {out_path}", flush=True)
    r = requests.get(url, timeout=seconds + 30)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    print(f"[profile] CPU profile saved ({len(r.content) / 1024:.0f} KB)", flush=True)


def _snapshot_profile(name: str, out_path: Path) -> None:
    """Fetch a non-blocking pprof endpoint (mutex, block, goroutine, heap)."""
    url = f"{DAEMON_URL}/debug/pprof/{name}?debug=0"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    print(f"[profile] {name} saved ({len(r.content) / 1024:.0f} KB) → {out_path}",
          flush=True)


def _run_iteration(
    label_to_idx: dict[str, int],
    num_workers: int,
    batch_size: int,
    image_size: int,
    max_batches: int,
    decode_mode: str = "raw",
) -> tuple[int, float]:
    """Iterate the DataLoader for `max_batches`, returning (samples, elapsed_s)."""
    if decode_mode == "rgb_uint8":
        # Daemon already decoded+resized to (image_size, image_size, 3) uint8.
        # Client transform is just ToTensor — no PIL on the per-sample path.
        ds = DatasetFS(
            num_workers=num_workers, seed=0,
            transform=T.ToTensor(),
            decode_mode="rgb_uint8", decode_image_size=image_size,
        )
    else:
        ds = DatasetFS(
            num_workers=num_workers, seed=0,
            transform=_build_transform(image_size),
        )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=functools.partial(_dfs_collate, label_to_idx=label_to_idx),
    )
    t0 = time.perf_counter()
    samples = 0
    for i, (x, _) in enumerate(loader):
        samples += x.shape[0]
        if i + 1 >= max_batches:
            break
    elapsed = time.perf_counter() - t0
    return samples, elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--batches", type=int, default=60,
                        help="Cap iteration at this many batches.")
    parser.add_argument("--cpu-seconds", type=int, default=25,
                        help="Length of the CPU pprof window (s).")
    parser.add_argument("--decode-mode", default="raw",
                        choices=["raw", "rgb_uint8"],
                        help="raw = client PIL decode; rgb_uint8 = daemon decode.")
    parser.add_argument("--dataset",
                        default="data/formats/imagenette/datasetfs")
    parser.add_argument("--imagefolder-root",
                        default="data/formats/imagenette/imagefolder")
    args = parser.parse_args()

    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    daemon_binary = REPO_ROOT / "bin" / "datasetfs"
    daemon_log = out_dir / "daemon.log"

    label_to_idx = _label_index(REPO_ROOT / args.imagefolder_root)
    print(f"[profile] {len(label_to_idx)} classes", flush=True)

    # Mutex rate 5 = sample 1-in-5 lock events (~1-3% overhead).
    # Block rate 10000 = sample blocking events >10 µs (~negligible overhead).
    proc = _start_daemon(
        binary=daemon_binary,
        root=REPO_ROOT / args.dataset,
        log_path=daemon_log,
        mutex_rate=5,
        block_rate=10000,
    )

    cpu_profile_path = out_dir / "daemon_cpu.pb.gz"
    cpu_err: list[BaseException] = []

    def _cpu_capture():
        try:
            _capture_cpu_profile(cpu_profile_path, seconds=args.cpu_seconds)
        except BaseException as e:
            cpu_err.append(e)

    try:
        # Start CPU-profile capture in a background thread. It only triggers
        # daemon-side sampling — Python work proceeds in parallel.
        cpu_thread = threading.Thread(target=_cpu_capture, daemon=True)
        cpu_thread.start()

        # Give the CPU capture a 1 s head-start so the first batches of work
        # land inside the profiling window (avoid measuring only the tail).
        time.sleep(1.0)

        # cProfile the iteration loop only (excludes daemon setup, model
        # construction, etc.) — Python-side hotspots will be on samples that
        # actually flow through DataLoader workers + collate.
        py_profile = cProfile.Profile()
        py_profile.enable()
        try:
            samples, elapsed = _run_iteration(
                label_to_idx=label_to_idx,
                num_workers=args.num_workers,
                batch_size=args.batch_size,
                image_size=args.image_size,
                max_batches=args.batches,
                decode_mode=args.decode_mode,
            )
        finally:
            py_profile.disable()

        sps = samples / elapsed if elapsed > 0 else 0.0
        print(f"\n[profile] iteration: {samples} samples in {elapsed:.2f} s "
              f"= {sps:.1f} samples/sec", flush=True)

        # Wait for CPU profile thread to finish (it blocks for cpu_seconds).
        cpu_thread.join(timeout=args.cpu_seconds + 30)
        if cpu_err:
            raise cpu_err[0]

        # Snapshot the remaining profiles right after the workload — they
        # accumulate from daemon start, so snapshot-now reflects the work
        # just done.
        for name in ("mutex", "block", "goroutine", "heap", "allocs"):
            _snapshot_profile(name, out_dir / f"daemon_{name}.pb.gz")

        # Write Python profile artifacts.
        py_profile_path = out_dir / "python.prof"
        py_profile.dump_stats(py_profile_path)
        print(f"[profile] python cProfile saved → {py_profile_path}", flush=True)

        # Also write a human-readable Python top-30 right here.
        py_txt = out_dir / "python_top30.txt"
        with open(py_txt, "w") as f:
            ps = pstats.Stats(py_profile, stream=f).strip_dirs().sort_stats("cumulative")
            ps.print_stats(30)
            f.write("\n\n--- TOTTIME ORDER ---\n\n")
            ps.sort_stats("tottime").print_stats(30)
        print(f"[profile] python top-30 saved → {py_txt}", flush=True)

        # Persist run config so we can correlate later.
        (out_dir / "run.txt").write_text(
            f"num_workers={args.num_workers}\n"
            f"batch_size={args.batch_size}\n"
            f"image_size={args.image_size}\n"
            f"batches={args.batches}\n"
            f"dataset={args.dataset}\n"
            f"samples={samples}\n"
            f"elapsed_s={elapsed:.3f}\n"
            f"samples_per_sec={sps:.1f}\n"
        )

    finally:
        _stop_daemon(proc)

    # Last: print quick hints for next steps.
    print("\n[profile] DONE. Inspect with:", flush=True)
    print(f"  {GO_PATH} tool pprof -top {cpu_profile_path}", flush=True)
    print(f"  {GO_PATH} tool pprof -top {out_dir}/daemon_mutex.pb.gz", flush=True)
    print(f"  {GO_PATH} tool pprof -top {out_dir}/daemon_block.pb.gz", flush=True)
    print(f"  cat {py_txt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
