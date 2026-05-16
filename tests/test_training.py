"""Phase 1 training-correctness tests.

These tests prove a real model can learn on the data DatasetFS serves:
  - test_loss_decreases_imagenette: training reduces loss across epochs
  - test_loss_parity_with_imagefolder: same model + seed converges similarly
    on DatasetFS and on raw ImageFolder (proves no data corruption / mislabeling)

They are slow (training-time scales with CPU). Marked with timeout=1800.
"""
from __future__ import annotations

import functools
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from clients.python import DatasetFS

from tests.helpers import imagefolder_index


pytestmark = pytest.mark.timeout(1800)

IMG_SIZE = 96
BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCHS = 3
SEED = 42


class SimpleCNN(nn.Module):
    """Small CNN: 3 conv blocks → adaptive pool → FC. Lifted from main.py."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _label_index(imagefolder_root: Path) -> dict[str, int]:
    truth = imagefolder_index(imagefolder_root)
    classes = sorted(set(truth.values()))
    return {c: i for i, c in enumerate(classes)}


# Module-level helpers so they're picklable when DataLoader spawns workers
# (macOS / Python 3.13 default start method = 'spawn').
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


def _imagefolder_loader(root: Path) -> DataLoader:
    ds = ImageFolder(str(root), transform=_build_transform())
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
    )


def _datasetfs_loader(label_to_idx: dict[str, int]) -> DataLoader:
    ds = DatasetFS(num_workers=NUM_WORKERS, transform=_build_transform())
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=functools.partial(_dfs_collate, label_to_idx=label_to_idx),
    )


def _train_one_epoch(model, loader, optim, loss_fn) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        x, y = batch
        optim.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optim.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def _train(loader_factory, restart_fn, num_classes: int, epochs: int = EPOCHS) -> list[float]:
    """Train SimpleCNN for `epochs`. restart_fn() is called before each epoch
    so DatasetFS can re-init its daemon session (no-op for non-DFS loaders)."""
    _set_seeds(SEED)
    model = SimpleCNN(num_classes=num_classes)
    optim = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    per_epoch = []
    for epoch in range(epochs):
        restart_fn(epoch)
        loader = loader_factory()
        ep_loss = _train_one_epoch(model, loader, optim, loss_fn)
        print(f"  epoch {epoch} mean loss = {ep_loss:.4f}", flush=True)
        per_epoch.append(ep_loss)
    return per_epoch


def test_loss_decreases_imagenette(daemon, imagenette_prepared):
    """Training a SimpleCNN on DatasetFS for `EPOCHS` epochs should show
    monotone-ish loss reduction, and end clearly below the random baseline.
    Proves: labels match images, data is decodable, model sees enough variety
    to learn. The stronger correctness check is test_loss_parity_with_imagefolder."""
    import math

    label_to_idx = _label_index(imagenette_prepared["imagefolder"])

    def restart_fn(epoch: int) -> None:
        if epoch > 0:
            daemon.restart()

    losses = _train(
        loader_factory=lambda: _datasetfs_loader(label_to_idx),
        restart_fn=restart_fn,
        num_classes=len(label_to_idx),
    )

    num_classes = len(label_to_idx)
    random_baseline = math.log(num_classes)  # cross-entropy of uniform distribution

    # 1) Loss must end below random by a clear margin (≥5%).
    assert losses[-1] < random_baseline * 0.95, (
        f"final loss {losses[-1]:.4f} is not clearly below random baseline "
        f"ln({num_classes})={random_baseline:.4f} — model isn't learning"
    )

    # 2) Loss must monotonically improve across epochs (allow ±1% wiggle for noise).
    for i in range(1, len(losses)):
        assert losses[i] < losses[i - 1] * 1.01, (
            f"loss not improving: {losses}"
        )

    # 3) Cumulative drop across all epochs ≥5%.
    drop = (losses[0] - losses[-1]) / losses[0]
    assert drop >= 0.05, (
        f"loss only dropped {drop*100:.1f}% across {EPOCHS} epochs: {losses}"
    )


def test_loss_parity_with_imagefolder(daemon, imagenette_prepared):
    """Train same model + seed twice — once via DatasetFS, once via raw ImageFolder.
    Final losses should be close (within 25%). Proves DatasetFS doesn't corrupt
    data or mis-shuffle in a way that breaks learning."""
    label_to_idx = _label_index(imagenette_prepared["imagefolder"])

    print("\n[parity] training on DatasetFS", flush=True)
    def dfs_restart(epoch: int) -> None:
        if epoch > 0:
            daemon.restart()
    dfs_losses = _train(
        loader_factory=lambda: _datasetfs_loader(label_to_idx),
        restart_fn=dfs_restart,
        num_classes=len(label_to_idx),
    )

    print("[parity] training on ImageFolder", flush=True)
    if_losses = _train(
        loader_factory=lambda: _imagefolder_loader(imagenette_prepared["imagefolder"]),
        restart_fn=lambda epoch: None,
        num_classes=len(label_to_idx),
    )

    dfs_final = dfs_losses[-1]
    if_final = if_losses[-1]
    rel_diff = abs(dfs_final - if_final) / max(if_final, 1e-6)
    print(
        f"[parity] DFS final={dfs_final:.4f}, ImageFolder final={if_final:.4f}, "
        f"rel_diff={rel_diff:.3f}",
        flush=True,
    )
    assert rel_diff < 0.25, (
        f"DatasetFS and ImageFolder produced very different losses: "
        f"DFS={dfs_final:.4f}, IF={if_final:.4f}, rel_diff={rel_diff:.3f}"
    )
