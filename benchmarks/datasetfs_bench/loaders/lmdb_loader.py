"""LMDB — a memory-mapped key-value store, a common ImageNet-in-one-file format.

One B-tree keyed by zero-padded index; each value is a pickled {data, label}.
The env is opened lazily per worker (an LMDB env/handle isn't picklable across
the `spawn` boundary), so only the path + key list cross into workers.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import ClassVar

from torch.utils.data import DataLoader, Dataset

from .base import BaseLoader, FormatUnavailable
from ._common import bound_labeled_collate, make_sample_decoder


class _LMDBDataset(Dataset):
    def __init__(self, path: str, decode):
        self.path = path
        self.decode = decode
        self._env = None
        import lmdb
        env = lmdb.open(path, readonly=True, lock=False, readahead=False, subdir=True)
        with env.begin() as txn:
            self.keys = pickle.loads(txn.get(b"__keys__"))
        env.close()

    def _ensure_env(self):
        if self._env is None:
            import lmdb
            self._env = lmdb.open(
                self.path, readonly=True, lock=False, readahead=False,
                subdir=True, max_readers=512,
            )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        self._ensure_env()
        with self._env.begin() as txn:
            rec = pickle.loads(txn.get(self.keys[i].encode()))
        return self.decode(rec["data"]), rec["label"]


class LMDBLoader(BaseLoader):
    name: ClassVar[str] = "lmdb"

    def setup(self) -> None:
        root = Path(self.spec["root"])
        if not (root / "data.mdb").exists() and not (root / ".done").exists():
            raise FormatUnavailable(f"LMDB not prepared at {root}")
        self._decode = make_sample_decoder(self.spec.get("modality", "image"), self.image_size)
        self._root = str(root)

    def make_loader(self) -> DataLoader:
        ds = _LMDBDataset(self._root, self._decode)
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
