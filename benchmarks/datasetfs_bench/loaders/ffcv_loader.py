"""FFCV — the GPU-oriented, JIT-compiled dataloader. Linux-only: no macOS
wheels (needs libturbojpeg/opencv). On darwin this loader raises
FormatUnavailable so the runner skips it; the real path runs on a Linux+GPU
host where `requirements-linux.txt` is installed. See HANDOFF.md "Real numbers
on Linux + GPU".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import ClassVar

import torch
from torch.utils.data import DataLoader

from .base import BaseLoader, FormatUnavailable


class FFCVLoader(BaseLoader):
    name: ClassVar[str] = "ffcv"

    def setup(self) -> None:
        if sys.platform == "darwin":
            raise FormatUnavailable("FFCV is Linux-only; not available on macOS")
        beton = Path(self.spec["root"]) / "data.beton"
        if not beton.exists():
            raise FormatUnavailable(f"FFCV .beton not prepared at {beton}")
        try:
            import ffcv  # noqa: F401
        except ImportError as e:
            raise FormatUnavailable(
                "ffcv not installed (pip install -r requirements-linux.txt)"
            ) from e
        self._beton = str(beton)
        idx_path = Path(self.spec["root"]) / "label_to_idx.json"
        self._prep_label_to_idx = json.loads(idx_path.read_text()) if idx_path.exists() else None

    def make_loader(self) -> DataLoader:
        # Built lazily so the heavy FFCV import only happens on Linux.
        from ffcv.loader import Loader, OrderOption
        from ffcv.fields.decoders import SimpleRGBImageDecoder, IntDecoder
        from ffcv.transforms import ToTensor, ToTorchImage, Convert, Squeeze
        import torchvision.transforms as T

        image_pipeline = [
            SimpleRGBImageDecoder(),
            ToTensor(),
            ToTorchImage(),
            Convert(torch.float32),
            T.Resize((self.image_size, self.image_size)),
            T.Lambda(lambda x: x / 255.0),
        ]
        label_pipeline = [IntDecoder(), ToTensor(), Squeeze()]
        return Loader(
            self._beton,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            order=OrderOption.RANDOM,
            pipelines={"image": image_pipeline, "label": label_pipeline},
        )

    def teardown(self) -> None:
        pass
