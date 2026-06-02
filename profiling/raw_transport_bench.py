"""Standalone baseline harness for opt 03 (pipeline transport).

Measures the DatasetFS *transport* path (pipe framing + per-item dict build +
refcount writes + SHM copy) on a CHEAP-decode dataset, so the cost is NOT masked
by a heavy PIL decode the way the image bench is. Two decode modes:

  - noop      : decode_fn returns the buffer length (no real work) — pure transport ceiling.
  - soundfile : real audio decode (sf.read on WAV) — the realistic non-image path.

Run BEFORE and AFTER the wire/zero-copy/refcount changes with identical args.

    python profiling/raw_transport_bench.py --decode noop --batches 200
    python profiling/raw_transport_bench.py --decode soundfile --batches 200
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clients.python import DatasetFS
from benchmarks.datasetfs_bench.runner.daemon_ctl import DaemonManager


def _noop_decode(buf):
    # Touch the buffer minimally so we don't get optimized away, but do no real
    # decode — isolates the transport cost.
    return len(buf)


def _identity(x):
    return x


def _soundfile_decode(buf):
    import soundfile as sf
    try:
        data, _sr = sf.read(io.BytesIO(buf), dtype="float32", always_2d=True)
    except Exception:
        return None
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode", choices=["noop", "soundfile"], default="noop")
    ap.add_argument("--batches", type=int, default=200,
                    help="stop after this many emitted (collated) batches")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--root", default="data/formats/speech_commands/datasetfs")
    args = ap.parse_args()

    binary = REPO_ROOT / "bin" / "datasetfs"
    root = REPO_ROOT / args.root

    decode_fn = _noop_decode if args.decode == "noop" else _soundfile_decode

    daemon = DaemonManager(binary=binary, root_path=root, cwd=REPO_ROOT)
    daemon.start()
    try:
        ds = DatasetFS(
            num_workers=args.num_workers,
            seed=0,
            decode_fn=decode_fn,
            transform=_identity,
        )
        # Iterate the IterableDataset directly (single stream) and count samples.
        t0 = time.perf_counter()
        samples = 0
        for _ in ds:
            samples += 1
            if samples >= args.batches * 64:
                break
        elapsed = time.perf_counter() - t0
        sps = samples / elapsed if elapsed > 0 else 0.0
        print(f"\n[transport] decode={args.decode} num_workers={args.num_workers} "
              f"samples={samples} elapsed={elapsed:.3f}s "
              f"samples_per_sec={sps:.1f}", flush=True)
    finally:
        daemon.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
