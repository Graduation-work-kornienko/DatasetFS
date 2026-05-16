"""DatasetFS — the format under test.

Note: this loader assumes the daemon is already running and pointed at
spec["root"]. The runner is responsible for daemon lifecycle (see
`runner/daemon_ctl.py`) so loader setup/teardown is cheap.
"""
from __future__ import annotations

import requests
from typing import ClassVar

from torch.utils.data import DataLoader

from clients.python import DatasetFS

from .base import BaseLoader
from ._common import bound_dfs_collate, make_image_transform


class DatasetFSLoader(BaseLoader):
    name: ClassVar[str] = "datasetfs"

    def setup(self) -> None:
        # Verify daemon is reachable before we try to construct a DatasetFS
        # (which posts /initialize_loading). Cleaner failure mode than
        # bubbling up a generic connection error mid-iteration.
        url = self.spec.get("daemon_url", "http://localhost:51409")
        r = requests.get(f"{url}/healthz", timeout=5)
        r.raise_for_status()

        self._daemon_url = url
        self._transform = make_image_transform(self.image_size)

    def make_loader(self) -> DataLoader:
        ds = DatasetFS(
            num_workers=self.num_workers,
            seed=self.seed,
            transform=self._transform,
            daemon_url=self._daemon_url,
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=bound_dfs_collate(self.label_to_idx),
            persistent_workers=False,
            pin_memory=False,
        )

    def teardown(self) -> None:
        # The runner stops the daemon; nothing to do here.
        pass
