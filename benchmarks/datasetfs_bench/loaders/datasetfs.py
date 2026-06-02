"""DatasetFS — the format under test.

Note: this loader assumes the daemon is already running and pointed at
spec["root"]. The runner is responsible for daemon lifecycle (see
`runner/daemon_ctl.py`) so loader setup/teardown is cheap.

`decode_mode` (spec field, default "raw"):
  - "raw"        : daemon serves raw JPEG bytes, Python decodes via PIL.
  - "rgb_uint8"  : daemon JPEG-decodes + resizes server-side, Python skips
                   PIL entirely and just ToTensor's the uint8 HWC buffer.
"""
from __future__ import annotations

import requests
from typing import ClassVar

from torch.utils.data import DataLoader

from clients.python import DatasetFS
from clients.python.dataset_fs import DECODE_RAW, DECODE_RGB_UINT8

from .base import BaseLoader
from ._common import (
    audio_decode_fn,
    audio_melspec_transform,
    bound_dfs_collate,
    make_image_transform,
    make_rgb_uint8_transform,
)


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
        self._modality = self.spec.get("modality", "image")
        self._decode_mode = self.spec.get("decode_mode", DECODE_RAW)
        if self._decode_mode not in (DECODE_RAW, DECODE_RGB_UINT8):
            raise ValueError(
                f"datasetfs spec.decode_mode must be one of "
                f"{DECODE_RAW!r}/{DECODE_RGB_UINT8!r}, got {self._decode_mode!r}"
            )
        # Audio (and any non-image type) goes through the generic raw transport:
        # decode is cheap client-side, so server-side rgb_uint8 (JPEG-only) is
        # meaningless here. This path is what opt 03 optimized.
        self._decode_fn = None
        if self._modality == "audio":
            if self._decode_mode == DECODE_RGB_UINT8:
                raise ValueError("modality='audio' is incompatible with decode_mode='rgb_uint8'")
            self._decode_fn = audio_decode_fn
            self._transform = audio_melspec_transform
        elif self._decode_mode == DECODE_RGB_UINT8:
            self._transform = make_rgb_uint8_transform()
        else:
            self._transform = make_image_transform(self.image_size)
        # opt 02: decode worker goroutines per pipeline (server-side decode
        # only). 0 = auto (daemon picks NumCPU/num_workers).
        self._decode_parallelism = int(self.spec.get("decode_parallelism", 0) or 0)

    def make_loader(self) -> DataLoader:
        kwargs = dict(
            num_workers=self.num_workers,
            seed=self.seed,
            transform=self._transform,
            daemon_url=self._daemon_url,
        )
        if self._decode_fn is not None:
            kwargs["decode_fn"] = self._decode_fn
        if self._decode_mode == DECODE_RGB_UINT8:
            kwargs["decode_mode"] = DECODE_RGB_UINT8
            kwargs["decode_image_size"] = self.image_size
            if self._decode_parallelism > 0:
                kwargs["decode_parallelism"] = self._decode_parallelism
        ds = DatasetFS(**kwargs)
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
