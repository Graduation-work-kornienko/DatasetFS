"""WebDataset — the canonical streaming-shards format. Main competitor."""
from __future__ import annotations

import glob
import io
from pathlib import Path
from typing import ClassVar

from PIL import Image
from torch.utils.data import DataLoader

from .base import BaseLoader
from ._common import bound_wds_collate, make_image_transform


# Module-level so DataLoader workers (`spawn` start method on macOS) can
# pickle them. We bypass webdataset's autodecode entirely because its default
# `.cls` decoder tries `int(data)` — but our class names are ImageNet synsets
# like "n01440764", not integers.
def _decode_sample(sample):
    img = Image.open(io.BytesIO(sample["jpg"]))
    label = sample["cls"]
    if isinstance(label, (bytes, bytearray)):
        label = label.decode("utf-8")
    return img, label


def _identity(x):
    return x


class WebDatasetLoader(BaseLoader):
    name: ClassVar[str] = "webdataset"

    def setup(self) -> None:
        try:
            import webdataset as wds  # noqa: F401
        except ImportError as e:
            raise ImportError("install `webdataset` to use this loader") from e

        root = Path(self.spec["root"])
        shards = sorted(glob.glob(str(root / "shard-*.tar")))
        if not shards:
            raise FileNotFoundError(f"no WebDataset shards found under {root}")
        self._shards = shards
        self._transform = make_image_transform(self.image_size)

    def make_loader(self) -> DataLoader:
        import webdataset as wds

        ds = (
            # shardshuffle=100 = shuffle the shard list with buffer 100 (>= our
            # shard count, so effectively a full reshuffle every epoch).
            wds.WebDataset(self._shards, shardshuffle=100, empty_check=False)
            .shuffle(1000)
            .map(_decode_sample)
            .map_tuple(self._transform, _identity)
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=bound_wds_collate(self.label_to_idx),
            persistent_workers=False,
            pin_memory=False,
        )

    def teardown(self) -> None:
        self._shards = None
