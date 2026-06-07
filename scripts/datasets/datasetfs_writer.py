"""Small DatasetFS shard/manifest writer used by dataset preparation scripts.

It writes uncompressed tar shards named ``shard_N`` plus ``metadata.jsonl`` in
the manifest shape the Go daemon already loads. Payloads are arbitrary bytes;
metadata is arbitrary JSON-serializable object metadata.
"""
from __future__ import annotations

import io
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any


class DatasetFSWriter:
    def __init__(self, out: Path, shard_target_bytes: int = 512 * 1024 * 1024,
                 overwrite: bool = True):
        self.out = Path(out)
        self.shard_target_bytes = shard_target_bytes
        self.overwrite = overwrite
        self.manifest: dict[str, Any] = {"version": "1.0", "shards_meta": {}, "files": {}}
        self.shard_id = -1
        self.current = 0
        self.tf: tarfile.TarFile | None = None
        self.count = 0
        self.total_payload_bytes = 0

    def __enter__(self) -> "DatasetFSWriter":
        if self.out.exists():
            if not self.overwrite:
                raise FileExistsError(self.out)
            shutil.rmtree(self.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self._next_shard()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.tf is not None:
            self.tf.close()
            self.manifest["shards_meta"][str(self.shard_id)] = {
                "type": "base",
                "total_size": self.current,
            }
        if exc_type is None:
            (self.out / "metadata.jsonl").write_text(json.dumps(self.manifest), encoding="utf-8")
            (self.out / ".done").touch()

    def _next_shard(self) -> None:
        if self.tf is not None:
            self.tf.close()
            self.manifest["shards_meta"][str(self.shard_id)] = {
                "type": "base",
                "total_size": self.current,
            }
        self.shard_id += 1
        self.current = 0
        self.tf = tarfile.open(self.out / f"shard_{self.shard_id}", "w")

    def add(self, name: str, payload: bytes, meta: dict[str, Any] | None = None) -> None:
        if name in self.manifest["files"]:
            raise ValueError(f"duplicate DatasetFS logical path: {name}")
        if self.current > 0 and self.current + len(payload) + 512 > self.shard_target_bytes:
            self._next_shard()
        assert self.tf is not None
        info = tarfile.TarInfo(name=name)
        info.size = len(payload)
        info.mode = 0o600
        self.tf.addfile(info, io.BytesIO(payload))
        self.manifest["files"][name] = {
            "c_id": self.shard_id,
            "offset": self.current + 512,
            "size": len(payload),
            "deleted": False,
            "path": name,
            "meta": meta or {},
        }
        padding = (512 - (len(payload) % 512)) % 512
        self.current += 512 + len(payload) + padding
        self.count += 1
        self.total_payload_bytes += len(payload)

    @property
    def shard_count(self) -> int:
        return self.shard_id + 1 if self.shard_id >= 0 else 0
