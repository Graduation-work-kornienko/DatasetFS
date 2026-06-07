"""Universal DatasetFS training smoke tests.

These tests build tiny but structurally different DatasetFS roots directly:

* text classification: UTF-8 documents + label metadata;
* audio classification: WAV bytes + label metadata;
* multimodal classification: JPEG image bytes + tabular vector metadata.

The point is not dataset scale; it is proving the DatasetFS transport is generic:
arbitrary object bytes plus JSON metadata can be decoded/collated into the model
contract a training loop needs.
"""
from __future__ import annotations

import functools
import io
import json
import math
import wave
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as T

from clients.python import DatasetFS
from scripts.datasets.datasetfs_writer import DatasetFSWriter
from tests.conftest import DaemonManager


BATCH_SIZE = 8
N_PER_CLASS = 16
SEED = 123


def _build_datasetfs_root(root: Path, objects: list[tuple[str, bytes, dict]]) -> None:
    """Create one uncompressed DatasetFS shard plus JSON manifest."""
    with DatasetFSWriter(root) as writer:
        for name, payload, meta in objects:
            writer.add(name, payload, meta)


def _run_with_daemon(root: Path, daemon_binary: Path, repo_root: Path, fn):
    manager = DaemonManager(binary=daemon_binary, root_path=root, cwd=repo_root)
    manager.start()
    try:
        return fn()
    finally:
        manager.stop()


def _train(model: nn.Module, loader: DataLoader, steps: int = 6) -> list[float]:
    torch.manual_seed(SEED)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.CrossEntropyLoss()
    losses: list[float] = []
    before = [p.detach().clone() for p in model.parameters()]
    for i, (inputs, targets) in enumerate(loader):
        if i >= steps:
            break
        opt.zero_grad()
        if isinstance(inputs, dict):
            out = model(**inputs)
        else:
            out = model(inputs)
        loss = loss_fn(out, targets)
        assert torch.isfinite(loss), f"non-finite loss at step {i}: {loss}"
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    assert losses, "loader yielded no trainable batches"
    changed = any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert changed, "optimizer did not update model parameters"
    return losses


# ---- text -------------------------------------------------------------------


def _make_text_root(root: Path) -> None:
    objects = []
    for i in range(N_PER_CLASS):
        objects.append((f"text_a_{i:03d}.txt", ("alpha " * 24).encode(), {"label": "alpha"}))
        objects.append((f"text_b_{i:03d}.txt", ("zulu " * 24).encode(), {"label": "zulu"}))
    _build_datasetfs_root(root, objects)


def _decode_text(raw) -> torch.Tensor:
    data = bytes(raw).lower()
    hist = torch.zeros(32, dtype=torch.float32)
    for b in data:
        hist[b % 32] += 1.0
    return hist / max(1.0, hist.sum())


def _identity(x):
    return x


def _collate_vector(items, label_to_idx: dict[str, int]):
    x = torch.stack([it["image"] for it in items])
    y = torch.tensor([label_to_idx[it["label"]] for it in items], dtype=torch.long)
    return x, y


class VectorClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 16), nn.ReLU(), nn.Linear(16, num_classes))

    def forward(self, x):
        return self.net(x)


def test_datasetfs_trains_on_text_documents(tmp_path, daemon_binary, repo_root):
    root = tmp_path / "text_datasetfs"
    _make_text_root(root)

    def run():
        ds = DatasetFS(num_workers=0, decode_fn=_decode_text, transform=_identity, timeout_seconds=10)
        loader = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            num_workers=0,
            collate_fn=functools.partial(_collate_vector, label_to_idx={"alpha": 0, "zulu": 1}),
        )
        losses = _train(VectorClassifier(32, 2), loader)
        assert all(math.isfinite(v) for v in losses)

    _run_with_daemon(root, daemon_binary, repo_root, run)


# ---- audio ------------------------------------------------------------------


def _wav_bytes(freq_hz: float, seconds: float = 0.25, sr: int = 8000) -> bytes:
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    wave_f32 = 0.6 * np.sin(2 * np.pi * freq_hz * t)
    pcm = (wave_f32 * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _make_audio_root(root: Path) -> None:
    objects = []
    for i in range(N_PER_CLASS):
        objects.append((f"low_{i:03d}.wav", _wav_bytes(220), {"label": "low"}))
        objects.append((f"high_{i:03d}.wav", _wav_bytes(1760), {"label": "high"}))
    _build_datasetfs_root(root, objects)


def _decode_wav_features(raw) -> torch.Tensor:
    with wave.open(io.BytesIO(bytes(raw)), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
    samples = torch.from_numpy(np.frombuffer(frames, dtype="<i2").astype("float32")) / 32768.0
    chunks = torch.chunk(samples, 16)
    energy = torch.stack([c.abs().mean() for c in chunks])
    zero_cross = (samples[:-1] * samples[1:] < 0).float().mean().view(1)
    return torch.cat([energy, zero_cross])


def test_datasetfs_trains_on_audio_waveforms(tmp_path, daemon_binary, repo_root):
    root = tmp_path / "audio_datasetfs"
    _make_audio_root(root)

    def run():
        ds = DatasetFS(num_workers=0, decode_fn=_decode_wav_features, transform=_identity, timeout_seconds=10)
        loader = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            num_workers=0,
            collate_fn=functools.partial(_collate_vector, label_to_idx={"low": 0, "high": 1}),
        )
        losses = _train(VectorClassifier(17, 2), loader)
        assert all(math.isfinite(v) for v in losses)

    _run_with_daemon(root, daemon_binary, repo_root, run)


# ---- image + tabular multimodal ---------------------------------------------


_IMG_TF = T.Compose([T.Resize((32, 32)), T.ToTensor()])


def _jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    img = Image.new("RGB", (48, 48), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _make_multimodal_root(root: Path) -> None:
    objects = []
    for i in range(N_PER_CLASS):
        objects.append((
            f"red_tab_{i:03d}.jpg",
            _jpeg_bytes((220, 40 + i % 20, 40)),
            {"label": "red", "tab": [1.0, 0.0, float(i) / N_PER_CLASS, 0.25]},
        ))
        objects.append((
            f"blue_tab_{i:03d}.jpg",
            _jpeg_bytes((40, 40 + i % 20, 220)),
            {"label": "blue", "tab": [0.0, 1.0, float(i) / N_PER_CLASS, 0.75]},
        ))
    _build_datasetfs_root(root, objects)


def _decode_image(raw):
    return Image.open(io.BytesIO(bytes(raw))).convert("RGB")


def _collate_multimodal(items, label_to_idx: dict[str, int]):
    inputs = {
        "image": torch.stack([it["image"] for it in items]),
        "tab": torch.tensor([it["tab"] for it in items], dtype=torch.float32),
    }
    y = torch.tensor([label_to_idx[it["label"]] for it in items], dtype=torch.long)
    return inputs, y


class TinyFusion(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.image = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.tab = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
        self.head = nn.Linear(16, num_classes)

    def forward(self, image, tab):
        return self.head(torch.cat([self.image(image), self.tab(tab)], dim=1))


def test_datasetfs_trains_on_image_tabular_multimodal(tmp_path, daemon_binary, repo_root):
    root = tmp_path / "multimodal_datasetfs"
    _make_multimodal_root(root)

    def run():
        ds = DatasetFS(num_workers=0, decode_fn=_decode_image, transform=_IMG_TF, timeout_seconds=10)
        loader = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            num_workers=0,
            collate_fn=functools.partial(_collate_multimodal, label_to_idx={"red": 0, "blue": 1}),
        )
        losses = _train(TinyFusion(2), loader)
        assert all(math.isfinite(v) for v in losses)

    _run_with_daemon(root, daemon_binary, repo_root, run)
