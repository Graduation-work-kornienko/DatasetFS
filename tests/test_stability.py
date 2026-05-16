"""Phase 3 long-running stability test.

Runs ~20 epochs against a single daemon process to surface slow-growing
defects that single-epoch correctness tests miss:

  - **Throughput retention**: samples/sec of late epochs must stay within
    15% of an early-epoch baseline. Catches refcount drift (slot pool
    shrinking), goroutine leaks (per-epoch overhead growing), or any
    cache/allocator pathology that makes successive sessions slower.

  - **Daemon RSS growth**: daemon RSS at end must not exceed 2× the
    early-epoch RSS. Catches goroutine leaks (each leaked goroutine ~8 KB
    stack), shm allocator never freeing its buffers across sessions, or
    other unbounded growth.

Each epoch re-initializes the loading session via `POST /initialize_loading`
(which is how a real training run would behave across checkpoint restarts).
The daemon process itself stays up the entire time, so leaks accumulate.

Runtime ~3-5 min CPU. Marked with a long timeout.
"""
from __future__ import annotations

import functools
import time
from pathlib import Path

import psutil
import pytest
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader

from clients.python import DatasetFS
from tests.helpers import imagefolder_index


pytestmark = pytest.mark.timeout(1200)

IMG_SIZE = 64
BATCH_SIZE = 32
NUM_WORKERS = 2
NUM_EPOCHS = 20
MAX_BATCHES_PER_EPOCH = 30  # caps each epoch ~5-10 s so the test stays manageable


def _to_rgb(img):
    return img.convert("RGB")


def _build_transform() -> T.Compose:
    return T.Compose([
        T.Lambda(_to_rgb),
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
    ])


def _dfs_collate(items, label_to_idx):
    images = torch.stack([it["image"] for it in items])
    labels = torch.tensor(
        [label_to_idx[it["label"]] for it in items],
        dtype=torch.long,
    )
    return images, labels


def _build_loader(label_to_idx: dict[str, int], seed: int) -> DataLoader:
    ds = DatasetFS(num_workers=NUM_WORKERS, seed=seed, transform=_build_transform())
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=functools.partial(_dfs_collate, label_to_idx=label_to_idx),
    )


def _label_index(imagefolder_root: Path) -> dict[str, int]:
    truth = imagefolder_index(imagefolder_root)
    classes = sorted(set(truth.values()))
    return {c: i for i, c in enumerate(classes)}


def _process_tree_rss(pid: int) -> int:
    """Sum RSS of `pid` and all its descendants. Resilient to children
    that disappear mid-walk (DataLoader workers spawn/exit each epoch)."""
    total = 0
    try:
        proc = psutil.Process(pid)
        total += proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return total


def _run_one_epoch(loader: DataLoader) -> tuple[float, int]:
    """Iterate the loader, return (samples_per_sec, n_samples).

    Times only the iteration, not loader construction (which includes
    daemon /initialize_loading and worker spawn — bursty and irrelevant
    to steady-state throughput retention)."""
    t0 = time.perf_counter()
    n_samples = 0
    n_batches = 0
    for batch in loader:
        x, _ = batch
        n_samples += x.shape[0]
        n_batches += 1
        if n_batches >= MAX_BATCHES_PER_EPOCH:
            break
    elapsed = time.perf_counter() - t0
    sps = n_samples / elapsed if elapsed > 0 else 0.0
    return sps, n_samples


def test_daemon_stability_across_20_epochs(daemon, imagenette_prepared):
    """20 sessions × 1 epoch on the same daemon process. Surfaces accumulating
    defects (refcount drift, goroutine/FD leaks, daemon memory leak)."""
    label_to_idx = _label_index(imagenette_prepared["imagefolder"])

    daemon_pid = daemon.pid
    assert daemon_pid is not None, "daemon should be running"

    per_epoch_sps: list[float] = []
    per_epoch_rss: list[int] = []

    initial_rss = _process_tree_rss(daemon_pid)
    print(f"\n[stability] initial daemon RSS = {initial_rss / 1024 / 1024:.1f} MB",
          flush=True)

    for epoch in range(NUM_EPOCHS):
        loader = _build_loader(label_to_idx, seed=epoch)
        sps, n_samples = _run_one_epoch(loader)
        # Tear down workers between epochs so pipes close cleanly before the
        # next /initialize_loading call.
        del loader

        # Sample daemon RSS AFTER worker teardown (excludes transient peaks
        # from spawn-mode worker processes).
        rss = _process_tree_rss(daemon_pid)

        per_epoch_sps.append(sps)
        per_epoch_rss.append(rss)
        print(
            f"[stability] epoch {epoch:2d}: "
            f"samples/sec={sps:7.1f}  "
            f"n_samples={n_samples:4d}  "
            f"daemon_rss={rss / 1024 / 1024:6.1f} MB",
            flush=True,
        )

    # Throughput retention: epoch 2 is our baseline (skips warmup epochs 0/1
    # where worker spawn + cold caches dominate). Late-window mean must stay
    # within 85% of that baseline.
    baseline = per_epoch_sps[2]
    late_window = per_epoch_sps[-3:]
    late_mean = sum(late_window) / len(late_window)
    retention = late_mean / baseline if baseline > 0 else 0.0
    print(
        f"\n[stability] retention: baseline(epoch=2)={baseline:.1f} sps, "
        f"late_mean(last 3)={late_mean:.1f} sps, ratio={retention:.2%}",
        flush=True,
    )
    assert retention >= 0.85, (
        f"throughput degraded over {NUM_EPOCHS} epochs: "
        f"baseline={baseline:.1f}, late_mean={late_mean:.1f}, "
        f"retention={retention:.2%} (threshold 85%). Full curve: {per_epoch_sps}"
    )

    # RSS growth: late RSS must stay below 2× early RSS. Compare end vs the
    # early-but-not-cold-start window (epoch 2) to avoid measuring just the
    # one-shot pipeline init cost.
    early_rss = per_epoch_rss[2]
    final_rss = per_epoch_rss[-1]
    growth = final_rss / early_rss if early_rss > 0 else 0.0
    print(
        f"[stability] RSS: early(epoch=2)={early_rss/1024/1024:.1f} MB, "
        f"final={final_rss/1024/1024:.1f} MB, growth={growth:.2f}×",
        flush=True,
    )
    assert growth <= 2.0, (
        f"daemon RSS grew {growth:.2f}× over {NUM_EPOCHS} epochs: "
        f"early={early_rss/1024/1024:.1f} MB, final={final_rss/1024/1024:.1f} MB. "
        f"Likely goroutine/FD leak across sessions. Full RSS curve (MB): "
        f"{[round(r/1024/1024, 1) for r in per_epoch_rss]}"
    )
