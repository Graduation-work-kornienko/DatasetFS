"""WebDataset — the canonical streaming-shards format. Main competitor."""
from __future__ import annotations

import functools
import glob
import shlex
from pathlib import Path
from typing import ClassVar

from torch.utils.data import DataLoader

from .base import BaseLoader, FormatUnavailable
from ._common import bound_wds_collate, make_sample_decoder

# Data field key inside each tar sample, by modality (prepare_webdataset stores
# the raw file bytes under its lowercased extension).
_IMAGE_KEYS = ("jpg", "jpeg", "png")
_AUDIO_KEYS = ("wav", "flac")


# Module-level so DataLoader workers (`spawn` start method on macOS) can pickle
# them. We bypass webdataset's autodecode (its `.cls` decoder tries `int(data)`,
# but our class names are synsets like "n01440764") and run our shared per-
# modality decoder so every format feeds the model identical tensors.
def _decode_sample(sample, decode, data_keys):
    data = next((sample[k] for k in data_keys if k in sample), None)
    label = sample["cls"]
    if isinstance(label, (bytes, bytearray)):
        label = label.decode("utf-8")
    return decode(data), label


class WebDatasetLoader(BaseLoader):
    name: ClassVar[str] = "webdataset"

    def setup(self) -> None:
        try:
            import webdataset as wds  # noqa: F401
        except ImportError as e:
            raise ImportError("install `webdataset` to use this loader") from e

        self._shards = self._resolve_shards()
        modality = self.spec.get("modality", "image")
        self._decode = make_sample_decoder(modality, self.image_size)
        self._data_keys = _AUDIO_KEYS if modality == "audio" else _IMAGE_KEYS

    def _resolve_shards(self) -> list[str]:
        """Local tar files by default; HTTP shard URLs when the spec asks for a
        remote source (Phase 5, train-while-streaming). webdataset consumes
        ``http(s)://`` shard URLs directly; for finicky endpoints set
        ``wds_http_mode: curl`` to stream via ``pipe:curl``."""
        # Explicit URL list wins.
        urls = self.spec.get("shard_urls")
        if not urls and self.spec.get("http_base"):
            base = str(self.spec["http_base"]).rstrip("/")
            n = int(self.spec["num_shards"])
            pattern = self.spec.get("shard_pattern", "shard-{i:06d}.tar")
            urls = [f"{base}/{pattern.format(i=i)}" for i in range(n)]
        if urls:
            if self.spec.get("wds_http_mode") == "curl":
                limit = self.spec.get("wds_curl_limit_rate")
                limit_arg = f" --limit-rate {shlex.quote(str(limit))}" if limit else ""
                return [f"pipe:curl -s -L{limit_arg} {shlex.quote(u)}" for u in urls]
            return list(urls)

        root = Path(self.spec["root"])
        shards = sorted(glob.glob(str(root / "shard-*.tar")))
        if not shards:
            raise FormatUnavailable(f"no WebDataset shards found under {root}")
        return shards

    def make_loader(self) -> DataLoader:
        import webdataset as wds

        ds = (
            # shardshuffle=100 = shuffle the shard list with buffer 100 (>= our
            # shard count, so effectively a full reshuffle every epoch).
            wds.WebDataset(self._shards, shardshuffle=100, empty_check=False)
            .shuffle(1000)
            .map(functools.partial(_decode_sample, decode=self._decode, data_keys=self._data_keys))
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
