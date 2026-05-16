"""Audio training smoke + parity test on Speech Commands V2.

Trains a tiny CNN on mel-spectrograms of the WAV clips served by DatasetFS.
Validates:
  - Loss decreases (model learns)
  - Loss curve is close to training on raw torchaudio.datasets.SPEECHCOMMANDS
    via ImageFolder-style iteration (parity).

The model is intentionally small + batches per epoch are capped so the test
runs in a few minutes on CPU.
"""
from __future__ import annotations

import functools
import io
import math
import os
import random
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset, IterableDataset
from PIL import Image  # unused but verifies env

from clients.python import DatasetFS

from tests.helpers import imagefolder_index


pytestmark = pytest.mark.timeout(1800)


SAMPLE_RATE = 16000
N_MELS = 32
TARGET_SAMPLES = SAMPLE_RATE   # 1 second
BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCHS = 2
MAX_BATCHES_PER_EPOCH = 200    # ~12% of dataset per epoch; needed for 35-class
SEED = 42


# Module-level (picklable for spawn workers) ----------------------------------


_MEL = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=400,
    hop_length=200,
    n_mels=N_MELS,
)


def _decode_and_melspec(raw_bytes: bytes):
    """WAV bytes -> mel spectrogram tensor (1, N_MELS, T). Pad/trunc to 1 second.

    Uses soundfile rather than torchaudio.load (torchcodec dep not on macOS).
    """
    data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T.copy())  # (C, N)
    if sr != SAMPLE_RATE:
        return None  # unexpected sample rate -> skip
    # Mono (most SC files already are)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # Pad / truncate to TARGET_SAMPLES
    n = waveform.shape[1]
    if n < TARGET_SAMPLES:
        pad = TARGET_SAMPLES - n
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    elif n > TARGET_SAMPLES:
        waveform = waveform[:, :TARGET_SAMPLES]
    return waveform


def _wave_to_melspec(waveform):
    """Apply mel spectrogram + log scale on a (1, TARGET_SAMPLES) tensor."""
    spec = _MEL(waveform)  # (1, N_MELS, T)
    return torch.log(spec + 1e-6)


def _dfs_collate(items, label_to_idx):
    images = torch.stack([it["image"] for it in items])
    labels = torch.tensor(
        [label_to_idx[it["label"]] for it in items],
        dtype=torch.long,
    )
    return images, labels


# Reference loader: torchvision-style ImageFolder iteration over the WAVs ------


class _WavImageFolder(Dataset):
    """Walks the prepared imagefolder, decodes each WAV with torchaudio at
    access time. Used as the ground-truth comparison baseline (independent
    of DatasetFS)."""

    def __init__(self, root: Path, label_to_idx: dict[str, int]):
        self.label_to_idx = label_to_idx
        self.samples: list[tuple[Path, int]] = []
        for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if cls_dir.name not in label_to_idx:
                continue
            for f in cls_dir.iterdir():
                if f.is_file():
                    self.samples.append((f, label_to_idx[cls_dir.name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with open(path, "rb") as f:
            raw = f.read()
        wf = _decode_and_melspec(raw)
        if wf is None:
            wf = torch.zeros(1, TARGET_SAMPLES)
        mel = _wave_to_melspec(wf)
        return mel, label


def _ref_collate(items):
    images = torch.stack([m for m, _ in items])
    labels = torch.tensor([l for _, l in items], dtype=torch.long)
    return images, labels


# Model -----------------------------------------------------------------------


class TinyAudioCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# Training helpers ------------------------------------------------------------


def _set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train_one_epoch(model, loader, optim, loss_fn, max_batches: int) -> float:
    model.train()
    losses: list[float] = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        optim.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optim.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def _label_index(speech_commands_root: Path) -> dict[str, int]:
    truth = imagefolder_index(speech_commands_root)
    classes = sorted(set(truth.values()))
    return {c: i for i, c in enumerate(classes)}


def _dfs_loader(label_to_idx):
    ds = DatasetFS(
        num_workers=NUM_WORKERS,
        decode_fn=_decode_and_melspec,
        transform=_wave_to_melspec,
    )
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=functools.partial(_dfs_collate, label_to_idx=label_to_idx),
    )


def _ref_loader(label_to_idx, imagefolder_root):
    ds = _WavImageFolder(imagefolder_root, label_to_idx)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        collate_fn=_ref_collate,
    )


def _train(loader_factory, restart_fn, num_classes: int) -> list[float]:
    _set_seeds(SEED)
    model = TinyAudioCNN(num_classes=num_classes)
    # Adam over SGD: log-mel features have very different gradient magnitudes
    # across frequency bins, and Adam's per-parameter adaptive LR copes with
    # this much better than vanilla SGD on a 35-class problem. lr=1e-3 is the
    # standard recipe for small audio CNNs.
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    per_epoch = []
    for epoch in range(EPOCHS):
        restart_fn(epoch)
        loader = loader_factory()
        ep_loss = _train_one_epoch(model, loader, optim, loss_fn, MAX_BATCHES_PER_EPOCH)
        print(f"  audio epoch {epoch} mean loss = {ep_loss:.4f}", flush=True)
        per_epoch.append(ep_loss)
    return per_epoch


# Tests -----------------------------------------------------------------------


def test_audio_loss_decreases(daemon_speech_commands, speech_commands_prepared):
    """Training a TinyAudioCNN via DatasetFS on Speech Commands V2 must drop
    loss meaningfully below the random baseline (ln(35) ≈ 3.555)."""
    label_to_idx = _label_index(speech_commands_prepared["imagefolder"])

    def restart_fn(epoch: int) -> None:
        if epoch > 0:
            daemon_speech_commands.restart()

    losses = _train(
        loader_factory=lambda: _dfs_loader(label_to_idx),
        restart_fn=restart_fn,
        num_classes=len(label_to_idx),
    )

    num_classes = len(label_to_idx)
    random_baseline = math.log(num_classes)

    # Below random baseline by ≥5%
    assert losses[-1] < random_baseline * 0.95, (
        f"final loss {losses[-1]:.4f} not clearly below random "
        f"ln({num_classes})={random_baseline:.4f}"
    )
    # Loss should improve across epochs (allow 1% wiggle)
    for i in range(1, len(losses)):
        assert losses[i] < losses[i - 1] * 1.01, f"loss didn't improve: {losses}"


def test_audio_loss_parity_with_imagefolder(daemon_speech_commands, speech_commands_prepared):
    """Train same model + seed twice — once via DatasetFS, once via a raw
    ImageFolder-style iteration with torchaudio. Final losses must be close
    (within 30% — audio is noisier than images, looser bound)."""
    label_to_idx = _label_index(speech_commands_prepared["imagefolder"])

    print("\n[audio parity] training on DatasetFS", flush=True)
    def dfs_restart(epoch: int) -> None:
        if epoch > 0:
            daemon_speech_commands.restart()
    dfs_losses = _train(
        loader_factory=lambda: _dfs_loader(label_to_idx),
        restart_fn=dfs_restart,
        num_classes=len(label_to_idx),
    )

    print("[audio parity] training on raw WAV ImageFolder", flush=True)
    ref_losses = _train(
        loader_factory=lambda: _ref_loader(label_to_idx, speech_commands_prepared["imagefolder"]),
        restart_fn=lambda epoch: None,
        num_classes=len(label_to_idx),
    )

    dfs_final, ref_final = dfs_losses[-1], ref_losses[-1]
    rel_diff = abs(dfs_final - ref_final) / max(ref_final, 1e-6)
    print(
        f"[audio parity] DFS final={dfs_final:.4f}, ref final={ref_final:.4f}, "
        f"rel_diff={rel_diff:.3f}",
        flush=True,
    )
    assert rel_diff < 0.30, (
        f"DFS and raw differ too much: DFS={dfs_final:.4f}, ref={ref_final:.4f}"
    )
