"""HuggingFace `datasets` — Arrow-backed, memory-mapped columnar store.

The on-disk dataset (built in prepare_huggingface) has an `image` column (HF
decodes it to PIL on access) and a `label` ClassLabel column. We convert the
label index back to its class-name string so it flows through the SAME runtime
label_to_idx as every other format.
"""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from torch.utils.data import DataLoader, Dataset

from .base import BaseLoader, FormatUnavailable
from ._common import bound_labeled_collate, make_image_transform


class _HFWrap(Dataset):
    def __init__(self, hfds, transform, idx2name):
        self.ds = hfds
        self.transform = transform
        self.idx2name = idx2name

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        rec = self.ds[i]
        return self.transform(rec["image"]), self.idx2name[rec["label"]]


class HuggingFaceLoader(BaseLoader):
    name: ClassVar[str] = "huggingface"

    def setup(self) -> None:
        root = Path(self.spec["root"])
        if not (root / "dataset_info.json").exists() and not (root / ".done").exists():
            raise FormatUnavailable(f"HuggingFace dataset not prepared at {root}")
        try:
            from datasets import load_from_disk
        except ImportError as e:
            raise FormatUnavailable("`datasets` not installed") from e

        self._hfds = load_from_disk(str(root))
        names = self._hfds.features["label"].names
        self._idx2name = {i: n for i, n in enumerate(names)}
        self._transform = make_image_transform(self.image_size)

    def make_loader(self) -> DataLoader:
        ds = _HFWrap(self._hfds, self._transform, self._idx2name)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=bound_labeled_collate(self.label_to_idx),
            persistent_workers=False,
            pin_memory=False,
        )

    def teardown(self) -> None:
        self._hfds = None
