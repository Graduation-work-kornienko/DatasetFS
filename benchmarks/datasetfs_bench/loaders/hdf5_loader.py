"""HDF5 — the scientific-computing array store. Single file, two datasets:
`data` (variable-length uint8 = raw image bytes) and `labels` (strings).

h5py File handles are not picklable, so we read the (small) label list eagerly
in __init__ and open the file lazily per worker for the byte reads.
"""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from torch.utils.data import DataLoader, Dataset

from .base import BaseLoader, FormatUnavailable
from ._common import bound_labeled_collate, make_sample_decoder


def _as_str(x) -> str:
    return x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)


class _HDF5Dataset(Dataset):
    def __init__(self, path: str, decode):
        self.path = path
        self.decode = decode
        self._h5 = None
        import h5py
        with h5py.File(path, "r") as h5:
            self.labels = [_as_str(l) for l in h5["labels"][:]]
        self.n = len(self.labels)

    def _ensure_file(self):
        if self._h5 is None:
            import h5py
            self._h5 = h5py.File(self.path, "r")

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        self._ensure_file()
        raw = self._h5["data"][i].tobytes()
        return self.decode(raw), self.labels[i]


class HDF5Loader(BaseLoader):
    name: ClassVar[str] = "hdf5"

    def setup(self) -> None:
        self._h5_path = Path(self.spec["root"]) / "data.h5"
        if not self._h5_path.exists():
            raise FormatUnavailable(f"HDF5 not prepared at {self._h5_path}")
        self._decode = make_sample_decoder(self.spec.get("modality", "image"), self.image_size)

    def make_loader(self) -> DataLoader:
        ds = _HDF5Dataset(str(self._h5_path), self._decode)
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
        pass
