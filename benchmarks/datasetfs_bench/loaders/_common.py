"""Shared transform + collate helpers for image loaders.

All defined at module level so DataLoader workers (`spawn` start method on
macOS) can pickle them.
"""
from __future__ import annotations

import functools

import torch
import torchvision.transforms as T


def _to_rgb(img):
    return img.convert("RGB")


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
