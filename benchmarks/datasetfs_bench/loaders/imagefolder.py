"""torchvision.datasets.ImageFolder — naive baseline (one file open per sample)."""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .base import BaseLoader
from ._common import imagefolder_collate, make_image_transform


class ImageFolderLoader(BaseLoader):
    name: ClassVar[str] = "imagefolder"

    def setup(self) -> None:
        root = Path(self.spec["root"])
        if not root.exists():
            raise FileNotFoundError(f"ImageFolder root does not exist: {root}")
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
