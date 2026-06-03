"""torchvision DatasetFolder — naive baseline (one file open per sample).

Images go through torchvision's ImageFolder (PIL). Audio uses the generic
DatasetFolder with a raw-bytes loader + the shared per-modality decoder, so the
"open every file" baseline exists for both modalities and feeds the model the
same tensors every other format does.
"""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, ImageFolder

from .base import BaseLoader, FormatUnavailable
from ._common import imagefolder_collate, make_image_transform, make_sample_decoder


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


class ImageFolderLoader(BaseLoader):
    name: ClassVar[str] = "imagefolder"

    def setup(self) -> None:
        root = Path(self.spec["root"])
        if not root.exists():
            raise FormatUnavailable(f"ImageFolder root does not exist: {root}")
        modality = self.spec.get("modality", "image")
        if modality == "audio":
            # DatasetFolder loads raw bytes; the per-modality decoder (soundfile
            # → log-mel) is the transform.
            decode = make_sample_decoder("audio", self.image_size)
            self._dataset = DatasetFolder(
                str(root), loader=_read_bytes, extensions=(".wav",), transform=decode,
            )
        else:
            self._transform = make_image_transform(self.image_size)
            self._dataset = ImageFolder(str(root), transform=self._transform)

        # Sanity: the dataset's class order must match our label_to_idx so
        # ground-truth labels line up across loaders.
        their_idx = {name: i for i, name in enumerate(self._dataset.classes)}
        if their_idx != self.label_to_idx:
            raise ValueError(
                f"ImageFolder class indices disagree with config label_to_idx:\n"
                f"  ImageFolder: {their_idx}\n  config: {self.label_to_idx}"
            )

    def make_loader(self) -> DataLoader:
        return DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=imagefolder_collate,
            persistent_workers=False,
            pin_memory=False,
        )

    def teardown(self) -> None:
        # ImageFolder holds nothing that needs explicit cleanup
        self._dataset = None
