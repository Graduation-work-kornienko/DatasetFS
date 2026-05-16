"""Phase 1 correctness tests.

Validates that DatasetFS serves the complete dataset, without duplicates,
that multi-worker partitions are disjoint, and that shuffling actually shuffles.
"""
from __future__ import annotations

import os
import random
from collections import Counter
from pathlib import Path

import pytest
import requests
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, get_worker_info

from clients.python import DatasetFS

from tests.helpers import (
    decode_with,
    hash_tensor,
    imagefolder_index,
    imagefolder_paths,
    path_key,
)


pytestmark = pytest.mark.timeout(900)


# Module-level so DataLoader workers (spawn) can pickle them.
def _to_rgb(img):
    return img.convert("RGB")


_BYTE_TEST_TRANSFORM = T.Compose([
    T.Lambda(_to_rgb),
    T.Resize((64, 64)),
    T.ToTensor(),
])


def _tagging_collate(items):
    info = get_worker_info()
    wid = info.id if info is not None else 0
    return {
        "worker_id": wid,
        "paths": [it["path"] for it in items],
        "labels": [it.get("label") for it in items],
    }


def _path_tensor_collate(items):
    """For byte-level integrity tests — keep path + tensor paired."""
    return [(it["path"], it["image"]) for it in items]


def _read_all(num_workers: int, batch_size: int = 64):
    """Drain one epoch of DatasetFS via DataLoader. Returns list of (worker_id, path, label)."""
    ds = DatasetFS(num_workers=num_workers)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_tagging_collate,
    )
    samples = []
    for batch in loader:
        wid = batch["worker_id"]
        for path, label in zip(batch["paths"], batch["labels"]):
            samples.append((wid, path, label))
    return samples


def _basenames(samples) -> list[str]:
    """Returns canonical class/filename keys (matches imagefolder_index keys)."""
    return [path_key(p) for _, p, _ in samples]


def test_completeness_workers0(daemon, imagenette_prepared):
    """Reading via DatasetFS with num_workers=0 must yield every file from the source."""
    truth = imagefolder_index(imagenette_prepared["imagefolder"])
    samples = _read_all(num_workers=0)
    got = _basenames(samples)
    assert set(got) == set(truth.keys()), (
        f"missing: {set(truth.keys()) - set(got)}; extra: {set(got) - set(truth.keys())}"
    )
    # Spot-check labels
    label_index = {path_key(p): label for _, p, label in samples}
    for name, expected_class in truth.items():
        if label_index.get(name) is not None:
            assert label_index[name] == expected_class, name


def test_completeness_workers4(daemon, imagenette_prepared):
    """Same but with num_workers=4 — the critical multi-worker assertion."""
    truth = imagefolder_index(imagenette_prepared["imagefolder"])
    samples = _read_all(num_workers=4)
    got = _basenames(samples)
    assert set(got) == set(truth.keys()), (
        f"missing: {set(truth.keys()) - set(got)}; extra: {set(got) - set(truth.keys())}"
    )


def test_no_duplicates_workers0(daemon, imagenette_prepared):
    samples = _read_all(num_workers=0)
    counts = Counter(_basenames(samples))
    dups = {k: c for k, c in counts.items() if c > 1}
    assert not dups, f"duplicates: {list(dups.items())[:10]}"


def test_no_duplicates_workers4(daemon, imagenette_prepared):
    samples = _read_all(num_workers=4)
    counts = Counter(_basenames(samples))
    dups = {k: c for k, c in counts.items() if c > 1}
    assert not dups, f"duplicates: {list(dups.items())[:10]}"


def test_multiworker_disjoint(daemon, imagenette_prepared):
    """No file should be seen by more than one worker."""
    samples = _read_all(num_workers=4)
    by_worker: dict[int, set[str]] = {}
    for wid, path, _ in samples:
        by_worker.setdefault(wid, set()).add(path_key(path))

    assert len(by_worker) == 4, f"expected 4 workers, got {sorted(by_worker)}"

    # Every pairwise intersection must be empty
    worker_ids = sorted(by_worker)
    for i in range(len(worker_ids)):
        for j in range(i + 1, len(worker_ids)):
            inter = by_worker[worker_ids[i]] & by_worker[worker_ids[j]]
            assert not inter, (
                f"workers {worker_ids[i]} and {worker_ids[j]} share {len(inter)} files: "
                f"e.g. {list(inter)[:3]}"
            )

    # Union must cover the dataset
    union = set().union(*by_worker.values())
    truth = set(imagefolder_index(imagenette_prepared["imagefolder"]).keys())
    assert union == truth


def test_shuffle_changes_order(daemon, imagenette_prepared):
    """Two epochs should produce different orderings of the same set."""
    epoch1 = _basenames(_read_all(num_workers=4))
    daemon.restart()
    epoch2 = _basenames(_read_all(num_workers=4))

    assert set(epoch1) == set(epoch2), "epochs see different sets"
    assert len(epoch1) == len(epoch2)

    same_pos = sum(1 for a, b in zip(epoch1, epoch2) if a == b)
    # With a real shuffle, fewer than 5% of positions should match by chance
    assert same_pos / len(epoch1) < 0.05, (
        f"epochs nearly identical: {same_pos}/{len(epoch1)} positions match"
    )


def test_repeatable_across_3_epochs(daemon, imagenette_prepared):
    """Each epoch covers the full dataset, regardless of which epoch."""
    truth = set(imagefolder_index(imagenette_prepared["imagefolder"]).keys())
    for epoch in range(3):
        if epoch > 0:
            daemon.restart()
        seen = set(_basenames(_read_all(num_workers=4)))
        assert seen == truth, f"epoch {epoch}: missing {len(truth - seen)} files"


# ---------- P0: byte-level integrity + per-file labels ----------


def test_labels_match_per_sample(daemon, imagenette_prepared):
    """For EVERY sample yielded by DatasetFS, the label must equal the source
    class folder it came from. Stronger than test_completeness's spot-check."""
    truth = imagefolder_index(imagenette_prepared["imagefolder"])
    samples = _read_all(num_workers=4)

    assert samples, "no samples received"
    mismatches: list[tuple[str, str, str]] = []
    for _, path, label in samples:
        name = path_key(path)
        expected = truth.get(name)
        if expected is None:
            mismatches.append((name, "<missing in truth>", str(label)))
        elif label is None:
            mismatches.append((name, expected, "<None>"))
        elif label != expected:
            mismatches.append((name, expected, str(label)))

    assert not mismatches, (
        f"{len(mismatches)} label mismatches; first 5: {mismatches[:5]}"
    )


def test_bytes_match_random_sample(daemon, imagenette_prepared):
    """Hash decoded tensors served by DatasetFS and compare against decoding
    the same files directly from the source ImageFolder with an identical
    transform. Catches: wrong shard offset, wrong slot byte range, wrong
    metadata→data association, corruption in shared memory."""
    truth_paths = imagefolder_paths(imagenette_prepared["imagefolder"])

    ds = DatasetFS(num_workers=0, transform=_BYTE_TEST_TRANSFORM)
    loader = DataLoader(
        ds,
        batch_size=64,
        num_workers=0,
        collate_fn=_path_tensor_collate,
    )

    dfs_hashes: dict[str, str] = {}
    for batch in loader:
        for path, tensor in batch:
            name = path_key(path)
            assert name not in dfs_hashes, f"duplicate file in DFS output: {name}"
            dfs_hashes[name] = hash_tensor(tensor)

    # Spot-check 100 random files: hash from DFS must equal hash from raw decode.
    rng = random.Random(0)
    sample_names = rng.sample(list(truth_paths.keys()), k=min(100, len(truth_paths)))

    mismatches: list[str] = []
    for name in sample_names:
        if name not in dfs_hashes:
            mismatches.append(f"{name}: not in DFS output")
            continue
        expected = hash_tensor(decode_with(_BYTE_TEST_TRANSFORM, truth_paths[name]))
        if dfs_hashes[name] != expected:
            mismatches.append(
                f"{name}: DFS={dfs_hashes[name][:12]}... expected={expected[:12]}..."
            )

    assert not mismatches, (
        f"{len(mismatches)}/{len(sample_names)} byte mismatches; first 3: {mismatches[:3]}"
    )


# ---------- P1: edge num_workers + re-init ----------


def test_num_workers_one(daemon, imagenette_prepared):
    """num_workers=1 exercises a different worker_info path than 0 (single
    subprocess workflow, get_worker_info() returns non-None)."""
    truth = set(imagefolder_index(imagenette_prepared["imagefolder"]).keys())
    samples = _read_all(num_workers=1)
    got = set(_basenames(samples))
    assert got == truth, f"missing {len(truth - got)}, extra {len(got - truth)}"


def test_num_workers_nine_is_max(daemon, imagenette_prepared):
    """num_workers=9 — exactly the slot count (NumSlots). Boundary case."""
    truth = set(imagefolder_index(imagenette_prepared["imagefolder"]).keys())
    samples = _read_all(num_workers=9)
    got = set(_basenames(samples))
    assert got == truth, f"missing {len(truth - got)}, extra {len(got - truth)}"


def test_num_workers_overflow_rejected(daemon, imagenette_prepared):
    """Daemon must reject num_workers > NumSlots(9) with HTTP 400, not silently
    overflow or hang."""
    resp = requests.post(
        "http://localhost:51409/initialize_loading",
        json={"num_workers": 12},
        timeout=10,
    )
    assert resp.status_code == 400, f"expected 400, got {resp.status_code}: {resp.text}"


def test_seed_param_accepted_and_complete(daemon, imagenette_prepared):
    """Smoke: passing a seed via /initialize_loading must not break the pipeline
    and the full dataset must still come through. The Go-side planner+dealer
    use seeded RNGs in this mode (verified by Go unit tests in pipeline_test.go).

    NOTE: bit-exact end-to-end output order is NOT guaranteed because of
    goroutine scheduling jitter in the Loader→Dealer drain — only shard
    assignment to workers and within-window shuffles are seedable. So we
    assert SET consistency, not sequence consistency."""
    truth = set(imagefolder_index(imagenette_prepared["imagefolder"]).keys())

    ds_a = DatasetFS(num_workers=4, seed=42)
    loader_a = DataLoader(ds_a, batch_size=64, num_workers=4, collate_fn=_tagging_collate)
    got_a = set()
    for batch in loader_a:
        for p in batch["paths"]:
            got_a.add(path_key(p))
    assert got_a == truth, f"seed=42 run A missing {len(truth - got_a)}"

    daemon.restart()
    ds_b = DatasetFS(num_workers=4, seed=42)
    loader_b = DataLoader(ds_b, batch_size=64, num_workers=4, collate_fn=_tagging_collate)
    got_b = set()
    for batch in loader_b:
        for p in batch["paths"]:
            got_b.add(path_key(p))
    assert got_b == truth, f"seed=42 run B missing {len(truth - got_b)}"


def test_seed_validation_rejects_negative(daemon, imagenette_prepared):
    """Python client must reject negative seeds before they reach the daemon."""
    with pytest.raises(ValueError, match="seed"):
        DatasetFS(num_workers=4, seed=-1)


def test_reinit_without_restart(daemon, imagenette_prepared):
    """Two iterations back-to-back WITHOUT killing the daemon — exercises the
    `currentSession.stop() + new session` path in startup_server.go, which is
    the production code path that test_repeatable_across_3_epochs bypasses
    (it does full daemon restart)."""
    truth = set(imagefolder_index(imagenette_prepared["imagefolder"]).keys())

    # First epoch
    seen_1 = set(_basenames(_read_all(num_workers=4)))
    assert seen_1 == truth, f"first epoch missing {len(truth - seen_1)} files"

    # Second epoch — same daemon process, /initialize_loading called again.
    # This is what the IPC server's session.stop() path is for.
    seen_2 = set(_basenames(_read_all(num_workers=4)))
    assert seen_2 == truth, f"second epoch missing {len(truth - seen_2)} files"
