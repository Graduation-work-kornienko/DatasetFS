"""TFRecord — TensorFlow's canonical sharded record format, read WITHOUT the
heavy `tensorflow` dep via the lightweight `tfrecord` library.

`TFRecordDataset` is an IterableDataset; with the prebuilt `.index` it shards
records across DataLoader workers and shuffles via an in-memory queue.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import ClassVar

from torch.utils.data import DataLoader

from .base import BaseLoader, FormatUnavailable
from ._common import bound_labeled_collate, make_sample_decoder


def _tf_decode(features: dict, decode):
    """tfrecord `byte` features arrive as uint8 ndarrays. Module-level so it
    pickles into `spawn` workers."""
    raw = bytes(features["image"])
    label = bytes(features["label"]).decode("utf-8")
    return decode(raw), label


class TFRecordLoader(BaseLoader):
    name: ClassVar[str] = "tfrecord"

    def setup(self) -> None:
        root = Path(self.spec["root"])
        self._data = str(root / "data.tfrecord")
        self._index = str(root / "data.index")
        if not Path(self._data).exists():
            raise FormatUnavailable(f"TFRecord not prepared at {root}")
        self._decode = make_sample_decoder(self.spec.get("modality", "image"), self.image_size)

    def make_loader(self) -> DataLoader:
        from tfrecord.torch.dataset import TFRecordDataset

        ds = TFRecordDataset(
            self._data,
            self._index,
            description={"image": "byte", "label": "byte"},
            shuffle_queue_size=1000,
            transform=functools.partial(_tf_decode, decode=self._decode),
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=bound_labeled_collate(self.label_to_idx),
            persistent_workers=False,
            pin_memory=False,
        )

    def teardown(self) -> None:
        pass
