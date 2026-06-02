"""Shared transform + collate helpers for image and audio loaders.

All defined at module level so DataLoader workers (`spawn` start method on
macOS) can pickle them.
"""
from __future__ import annotations

import functools
import io

import torch
import torchvision.transforms as T


def _to_rgb(img):
    return img.convert("RGB")


# ---- audio (Speech Commands) decode + transform -----------------------------
# Mirrors tests/test_speech_commands_training.py. Used to measure the *raw*
# pipeline path (opt 03): audio decode is cheap (soundfile), so the shared
# transport — not decode — dominates per-sample cost, unlike the image path
# where PIL masks it. soundfile is used directly because torchaudio.load routes
# through torchcodec, which has no macOS wheels.

AUDIO_SAMPLE_RATE = 16000
AUDIO_TARGET_SAMPLES = 16000  # 1 second
AUDIO_N_MELS = 32
AUDIO_N_FFT = 400
AUDIO_HOP = 200

# Lazily-built per process so the nn.Module isn't pickled across the spawn
# boundary; each worker constructs its own on first use.
_MEL = None


def _mel():
    global _MEL
    if _MEL is None:
        import torchaudio
        _MEL = torchaudio.transforms.MelSpectrogram(
            sample_rate=AUDIO_SAMPLE_RATE, n_fft=AUDIO_N_FFT,
            hop_length=AUDIO_HOP, n_mels=AUDIO_N_MELS,
        )
    return _MEL


def audio_decode_fn(raw):
    """WAV bytes/buffer → (1, AUDIO_TARGET_SAMPLES) mono waveform tensor, padded/
    truncated to 1 s. Returns None on an unexpected sample rate (exercises the
    skip path — opt 03 fixed the slot-leak it used to cause). Accepts a memoryview
    (zero-copy SHM slice); soundfile copies it into its own buffer."""
    import soundfile as sf
    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception:
        return None
    if sr != AUDIO_SAMPLE_RATE:
        return None
    waveform = torch.from_numpy(data.T.copy())  # (C, N)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    n = waveform.shape[1]
    if n < AUDIO_TARGET_SAMPLES:
        waveform = torch.nn.functional.pad(waveform, (0, AUDIO_TARGET_SAMPLES - n))
    elif n > AUDIO_TARGET_SAMPLES:
        waveform = waveform[:, :AUDIO_TARGET_SAMPLES]
    return waveform


def audio_melspec_transform(waveform):
    """(1, N) waveform → (1, n_mels, T) log-mel spectrogram, the model input."""
    spec = _mel()(waveform)
    return torch.log(spec + 1e-6)


def make_image_transform(image_size: int) -> T.Compose:
    """Canonical transform shared by all image loaders so they produce
    identical pixel statistics — eliminates a confounding variable from
    throughput comparisons."""
    return T.Compose([
        T.Lambda(_to_rgb),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])


def make_rgb_uint8_transform() -> T.Compose:
    """Transform for the DatasetFS server-side-decode (rgb_uint8) path: the
    daemon already produced (H, W, 3) uint8 RGB, so just ToTensor — which
    permutes HWC→CHW, casts to float32, and rescales to [0, 1]. Skips PIL.
    """
    return T.Compose([T.ToTensor()])


# ---- collate functions ------------------------------------------------------


def dfs_collate(items, label_to_idx):
    """DatasetFS yields dicts with `image` (transformed tensor) and `label` (string)."""
    images = torch.stack([it["image"] for it in items])
    targets = torch.tensor(
        [label_to_idx[it["label"]] for it in items],
        dtype=torch.long,
    )
    return images, targets


def imagefolder_collate(items):
    """torchvision.datasets.ImageFolder yields (image_tensor, int_label) tuples."""
    images = torch.stack([img for img, _ in items])
    targets = torch.tensor([lbl for _, lbl in items], dtype=torch.long)
    return images, targets


def webdataset_collate(items, label_to_idx):
    """WebDataset shards we built emit (image_tensor, label_str) after decoding."""
    images = torch.stack([img for img, _ in items])
    targets = torch.tensor(
        [label_to_idx[lbl] for _, lbl in items],
        dtype=torch.long,
    )
    return images, targets


def bound_dfs_collate(label_to_idx):
    return functools.partial(dfs_collate, label_to_idx=label_to_idx)


def bound_wds_collate(label_to_idx):
    return functools.partial(webdataset_collate, label_to_idx=label_to_idx)
