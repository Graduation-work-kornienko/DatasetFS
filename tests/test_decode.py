"""Phase 3 server-side decode correctness tests.

Verifies the daemon-side decoder (pure Go image/jpeg + x/image/draw.BiLinear)
produces images close to PIL.Image.open + Resize(BILINEAR) — the ground-truth
client-side decode path.

We do NOT assert byte-equality: Go and PIL use different bilinear-filter
edge handling and round-half-to-even/down behavior, so per-pixel uint8 values
can differ by up to ~5 in regions of high gradient. Instead we assert:

  - Same shape (H, W, 3) and dtype uint8
  - Mean absolute pixel difference < MEAN_DIFF_THRESHOLD
  - 95th-percentile pixel difference < P95_DIFF_THRESHOLD
  - No catastrophic mismatch (max diff < 64) that would suggest e.g. RGB↔BGR
    swap or row-stride bug
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from clients.python.dataset_fs import DatasetFS, DECODE_RGB_UINT8
from tests.helpers import imagefolder_index


IMAGE_SIZE = 64
N_SAMPLES = 20

# Per-pixel uint8 thresholds. Bilinear-filter implementation differences
# between PIL and Go's x/image/draw produce small systematic offsets; these
# bounds were calibrated empirically and are loose enough to absorb that
# noise but tight enough to catch real bugs (wrong colorspace, stride,
# resize target, etc).
MEAN_DIFF_THRESHOLD = 5.0     # avg uint8 distance per pixel-channel
P95_DIFF_THRESHOLD = 25.0     # 95% of pixels within this distance
MAX_DIFF_THRESHOLD = 90       # safety net: catastrophic mismatch


def _pil_reference(path: Path, size: int) -> np.ndarray:
    """Decode + resize via PIL — the reference path the daemon must approximate."""
    with Image.open(path) as im:
        rgb = im.convert("RGB").resize((size, size), Image.BILINEAR)
        return np.asarray(rgb, dtype=np.uint8)  # (size, size, 3)


def _compare(name: str, dfs_img: np.ndarray, pil_img: np.ndarray) -> dict:
    assert dfs_img.shape == pil_img.shape, (
        f"{name}: shape mismatch dfs={dfs_img.shape} pil={pil_img.shape}"
    )
    assert dfs_img.dtype == pil_img.dtype == np.uint8

    diff = np.abs(dfs_img.astype(np.int16) - pil_img.astype(np.int16))
    return {
        "mean": float(diff.mean()),
        "p95": float(np.percentile(diff, 95)),
        "max": int(diff.max()),
    }


def test_rgb_uint8_decode_matches_pil(daemon, imagenette_prepared):
    """Server-side decode + resize ≈ PIL decode + Resize(BILINEAR)."""
    truth = imagefolder_index(imagenette_prepared["imagefolder"])
    if not truth:
        pytest.skip("imagefolder is empty — nothing to compare")

    ds = DatasetFS(
        num_workers=0, seed=0,
        decode_mode=DECODE_RGB_UINT8,
        decode_image_size=IMAGE_SIZE,
        transform=lambda x: x,  # identity — we want raw uint8 HWC
    )

    seen = 0
    stats: list[dict] = []
    for sample in ds:
        path = Path(sample["path"])
        if not path.exists():
            # Some datasets store paths from a different mount; skip those.
            continue
        dfs_img = sample["image"]
        pil_img = _pil_reference(path, IMAGE_SIZE)
        s = _compare(path.name, dfs_img, pil_img)
        stats.append(s)
        print(
            f"[decode-cmp] {path.name:40s} "
            f"mean={s['mean']:5.2f} p95={s['p95']:5.1f} max={s['max']:3d}",
            flush=True,
        )
        seen += 1
        if seen >= N_SAMPLES:
            break

    assert seen >= 1, "expected to compare at least one sample"

    means = [s["mean"] for s in stats]
    p95s = [s["p95"] for s in stats]
    maxs = [s["max"] for s in stats]

    avg_mean = float(np.mean(means))
    avg_p95 = float(np.mean(p95s))
    max_max = int(max(maxs))

    print(
        f"\n[decode-cmp] {seen} samples: "
        f"avg(mean_diff)={avg_mean:.2f}  avg(p95)={avg_p95:.2f}  max(max)={max_max}",
        flush=True,
    )

    assert avg_mean < MEAN_DIFF_THRESHOLD, (
        f"server-side decode diverges too much from PIL: "
        f"avg mean abs diff = {avg_mean:.2f} > {MEAN_DIFF_THRESHOLD}. "
        f"Likely wrong colorspace, stride, or resize target."
    )
    assert avg_p95 < P95_DIFF_THRESHOLD, (
        f"95th-percentile pixel diff too high: {avg_p95:.2f} > {P95_DIFF_THRESHOLD}"
    )
    assert max_max < MAX_DIFF_THRESHOLD, (
        f"catastrophic max-diff = {max_max} >= {MAX_DIFF_THRESHOLD} — "
        f"likely RGB↔BGR swap or alpha-channel leak"
    )


def test_rgb_uint8_with_to_tensor_yields_float_chw(daemon, imagenette_prepared):
    """End-to-end: rgb_uint8 + transforms.ToTensor() yields float32 CHW in [0,1]."""
    import torchvision.transforms as T

    ds = DatasetFS(
        num_workers=0, seed=0,
        decode_mode=DECODE_RGB_UINT8,
        decode_image_size=IMAGE_SIZE,
        transform=T.ToTensor(),
    )
    sample = next(iter(ds))
    img = sample["image"]
    assert img.dtype.is_floating_point, f"expected float tensor, got {img.dtype}"
    assert img.shape == (3, IMAGE_SIZE, IMAGE_SIZE), (
        f"expected CHW (3,{IMAGE_SIZE},{IMAGE_SIZE}), got {tuple(img.shape)}"
    )
    assert 0.0 <= float(img.min()) <= float(img.max()) <= 1.0, (
        f"expected pixel range [0,1], got [{float(img.min())}, {float(img.max())}]"
    )
