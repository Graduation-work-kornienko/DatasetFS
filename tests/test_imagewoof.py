"""Smoke tests on Imagewoof — a second, harder dataset. Validates that the
fixes derived from Imagenette debugging aren't accidentally specific to that
dataset (e.g., specific shard count, specific class names)."""
from __future__ import annotations

import os
from collections import Counter

import pytest
import torch
from torch.utils.data import DataLoader, get_worker_info

from clients.python import DatasetFS

from tests.helpers import imagefolder_index, path_key


pytestmark = pytest.mark.timeout(900)


def _tagging_collate(items):
    info = get_worker_info()
    wid = info.id if info is not None else 0
    return {
        "worker_id": wid,
        "paths": [it["path"] for it in items],
        "labels": [it.get("label") for it in items],
    }


def _read_all(num_workers: int):
    ds = DatasetFS(num_workers=num_workers)
    loader = DataLoader(
        ds,
        batch_size=64,
        num_workers=num_workers,
        collate_fn=_tagging_collate,
    )
    samples = []
    for batch in loader:
        wid = batch["worker_id"]
        for path, label in zip(batch["paths"], batch["labels"]):
            samples.append((wid, path, label))
    return samples


def test_imagewoof_completeness_workers4(daemon_imagewoof, imagewoof_prepared):
    """Same completeness assertion as Imagenette, but on a different dataset
    with different shard count, class names, and image sizes."""
    truth = imagefolder_index(imagewoof_prepared["imagefolder"])
    samples = _read_all(num_workers=4)
    got = [path_key(p) for _, p, _ in samples]

    assert set(got) == set(truth.keys()), (
        f"imagewoof: missing {len(set(truth.keys()) - set(got))}, "
        f"extra {len(set(got) - set(truth.keys()))}"
    )

    counts = Counter(got)
    dups = {k: c for k, c in counts.items() if c > 1}
    assert not dups, f"imagewoof duplicates: {list(dups.items())[:5]}"


def test_imagewoof_labels_match_per_sample(daemon_imagewoof, imagewoof_prepared):
    """Labels must match the source class folder. Catches data/metadata
    coupling bugs that might be specific to imagewoof's class layout."""
    truth = imagefolder_index(imagewoof_prepared["imagefolder"])
    samples = _read_all(num_workers=4)

    mismatches = []
    for _, path, label in samples:
        name = path_key(path)
        expected = truth.get(name)
        if expected is None or label is None or label != expected:
            mismatches.append((name, expected, label))

    assert not mismatches, (
        f"{len(mismatches)} label mismatches in imagewoof; first 5: {mismatches[:5]}"
    )
