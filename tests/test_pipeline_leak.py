"""Regression test for the opt-03 slot-leak fix.

Before opt 03, the Python client decremented a slot's refcount only on the
*yield* path — every `continue` for a skipped sample (decode returned None,
size mismatch, transform raised) bypassed the decrement. A slot containing any
skipped sample therefore never reached refcount 0, so the Go planner never
recycled it. With more shards than shared-memory slots (Speech Commands has 37
shards vs 9 slots, so a single worker must recycle), enough skips would exhaust
the slots and stall the epoch — it would then end only via the 30 s idle
timeout, having served a small fraction of the data.

opt 03 counts EVERY item toward its slot (skipped or not) and decrements once
per slot per frame, so slots always recycle. This test drives a decode_fn that
skips half the samples and asserts the epoch still traverses the whole dataset
(≈ half the items survive), which is only possible if slots recycle.

A no-op decode_fn is used (no real audio decode) so the drain runs at transport
speed — the test is about slot lifecycle, not decoding.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import psutil
import pytest

from clients.python import DatasetFS
from scripts.datasets.datasetfs_writer import DatasetFSWriter
from tests.conftest import DaemonManager

pytestmark = pytest.mark.timeout(120)


def _identity(x):
    return x


def _bytes(x):
    return bytes(x)


def _process_rss(pid: int) -> int:
    try:
        return psutil.Process(pid).memory_info().rss
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0


def _slope_per_cycle(values: list[int]) -> float:
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


def test_skipped_samples_do_not_leak_slots(daemon_speech_commands):
    """With more shards than slots, skipping half the samples must NOT stall the
    epoch: every slot still recycles, so ≈ half the dataset survives."""
    url = daemon_speech_commands.url

    def drain(skip_fn):
        # Fresh session each drain (re-init the daemon's loading pipelines).
        ds = DatasetFS(num_workers=0, seed=0, decode_fn=skip_fn, transform=_identity,
                       timeout_seconds=20.0, daemon_url=url)
        return sum(1 for _ in ds)

    # Baseline: keep everything → full dataset size.
    full = drain(lambda buf: 1)
    assert full > 9 * 64, f"expected a multi-slot dataset, got only {full} samples"

    # Skip every other sample. A per-call counter is fine here (num_workers=0,
    # single process). If skipped items leaked slots, the epoch would stall after
    # ~9 slots and `kept` would be a small fraction, not ≈ full/2.
    state = {"i": 0}

    def skip_half(buf):
        state["i"] += 1
        return None if state["i"] % 2 == 0 else 1

    kept = drain(skip_half)

    # Survivors should be ≈ half of the full dataset. A leak would strand most
    # slots and collapse this far below half.
    assert 0.4 * full <= kept <= 0.6 * full, (
        f"kept={kept} not ≈ half of full={full} — slots likely leaked on skip"
    )


def test_pipeline_rss_plateaus_across_many_session_restarts(tmp_path, daemon_binary, repo_root):
    """Specialized memory-leak check for repeated pipeline/session restarts.

    Endurance benchmarks show whether throughput remains acceptable, but they do
    not directly classify memory growth. This test keeps one daemon process alive,
    repeatedly reinitializes DatasetFS loading sessions, drains a multi-shard
    dataset, and checks the post-drain daemon RSS curve after allocator warmup.
    """

    root = tmp_path / "synthetic_datasetfs"
    payload = b"x" * 2048
    with DatasetFSWriter(root, shard_target_bytes=64 * 1024) as writer:
        for i in range(9):
            writer.add(f"sample_{i:04d}.bin", payload, {"label": "x"})

    manager = DaemonManager(binary=daemon_binary, root_path=root, cwd=repo_root)
    manager.start()
    try:
        pid = manager.pid
        assert pid is not None
        rss_curve: list[int] = []
        counts: list[int] = []
        cycles = 60
        for cycle in range(cycles):
            ds = DatasetFS(num_workers=0, seed=cycle, decode_fn=_bytes, transform=_identity, timeout_seconds=1.0)
            count = sum(1 for _ in ds)
            del ds
            gc.collect()
            counts.append(count)
            rss = _process_rss(pid)
            rss_curve.append(rss)
            print(f"[pipeline-rss] cycle={cycle:02d} count={count} daemon_rss={rss / 1024 / 1024:.1f} MiB", flush=True)

        assert all(c == counts[0] for c in counts), f"session drains returned inconsistent counts: {counts}"
        warm = rss_curve[5:]
        growth = max(warm) - min(warm)
        slope = _slope_per_cycle(warm)
        print(
            f"[pipeline-rss] warm_growth={growth / 1024 / 1024:.1f} MiB "
            f"slope={slope / 1024 / 1024:.3f} MiB/cycle",
            flush=True,
        )
        assert growth <= 96 * 1024 * 1024, (
            "daemon RSS did not plateau across repeated DatasetFS sessions: "
            f"growth={growth / 1024 / 1024:.1f} MiB, curve MiB="
            f"{[round(v / 1024 / 1024, 1) for v in rss_curve]}"
        )
        assert slope <= 1.0 * 1024 * 1024, (
            "daemon RSS has a positive per-session trend after warmup: "
            f"slope={slope / 1024 / 1024:.3f} MiB/cycle, curve MiB="
            f"{[round(v / 1024 / 1024, 1) for v in rss_curve]}"
        )
    finally:
        manager.stop()


def test_pipeline_rss_plateaus_with_bounded_fuse_replacements(tmp_path, daemon_binary, repo_root):
    """RSS plateau check while data changes through the real mutation path.

    This intentionally uses bounded replace-only mutations over existing names.
    Add/delete workloads can legitimately grow WAL/delta state, so they are less
    useful as a leak gate. Replacing a fixed file set still exercises FUSE writes,
    WAL replay visibility, and repeated pipeline sessions without unbounded
    logical dataset growth.
    """

    if sys.platform != "darwin" or not Path("/Library/Filesystems/macfuse.fs").exists():
        pytest.skip("FUSE mutation RSS check requires macFUSE on macOS")

    root = tmp_path / "mutable_datasetfs"
    mount = tmp_path / "mnt"
    payload = b"x" * 2048
    with DatasetFSWriter(root, shard_target_bytes=64 * 1024) as writer:
        for i in range(9):
            writer.add(f"sample_{i:04d}.bin", payload, {"label": "x"})

    manager = DaemonManager(binary=daemon_binary, root_path=root, cwd=repo_root, mount_point=mount)
    manager.start()
    try:
        pid = manager.pid
        assert pid is not None
        rss_curve: list[int] = []
        counts: list[int] = []
        cycles = 50
        replacements_per_cycle = 3
        for cycle in range(cycles):
            for j in range(replacements_per_cycle):
                idx = (cycle * replacements_per_cycle + j) % 9
                name = f"sample_{idx:04d}.bin"
                data = f"replace-{cycle:03d}-{j:02d}".encode().ljust(2048, b"r")
                path = mount / name
                path.unlink(missing_ok=True)
                with open(path, "wb") as f:
                    f.write(data)

            ds = DatasetFS(num_workers=0, seed=cycle, decode_fn=_bytes, transform=_identity, timeout_seconds=1.0)
            count = sum(1 for _ in ds)
            del ds
            gc.collect()
            counts.append(count)
            rss = _process_rss(pid)
            rss_curve.append(rss)
            print(
                f"[pipeline-rss-mutate] cycle={cycle:02d} replacements={replacements_per_cycle} "
                f"count={count} daemon_rss={rss / 1024 / 1024:.1f} MiB",
                flush=True,
            )

        assert all(c == counts[0] for c in counts), f"mutating session drains returned inconsistent counts: {counts}"
        warm = rss_curve[8:]
        growth = max(warm) - min(warm)
        slope = _slope_per_cycle(warm)
        print(
            f"[pipeline-rss-mutate] warm_growth={growth / 1024 / 1024:.1f} MiB "
            f"slope={slope / 1024 / 1024:.3f} MiB/cycle",
            flush=True,
        )
        assert growth <= 128 * 1024 * 1024, (
            "daemon RSS did not plateau during bounded FUSE replacements: "
            f"growth={growth / 1024 / 1024:.1f} MiB, curve MiB="
            f"{[round(v / 1024 / 1024, 1) for v in rss_curve]}"
        )
        assert slope <= 1.5 * 1024 * 1024, (
            "daemon RSS has a positive per-cycle trend during bounded FUSE replacements: "
            f"slope={slope / 1024 / 1024:.3f} MiB/cycle, curve MiB="
            f"{[round(v / 1024 / 1024, 1) for v in rss_curve]}"
        )
    finally:
        manager.stop()
