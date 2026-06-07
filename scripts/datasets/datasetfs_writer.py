"""Small DatasetFS shard/manifest writer used by dataset preparation scripts.

It writes uncompressed tar shards named ``shard_N`` plus ``metadata.parquet`` in
the manifest shape the Go daemon loads. Payloads are arbitrary bytes; metadata
is arbitrary JSON-serializable object metadata stored as JSON bytes in Parquet.
"""
from __future__ import annotations

import io
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any


def write_parquet_manifest(out: Path, manifest: dict[str, Any]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(out)
    shards = [
        {
            "number": int(shard_id),
            "type": str(info["type"]),
            "total_size": int(info["total_size"]),
        }
        for shard_id, info in sorted(manifest["shards_meta"].items(), key=lambda kv: int(kv[0]))
    ]
    files = [
        {
            "path": path,
            "shard_id": int(info["c_id"]),
            "offset": int(info["offset"]),
            "size": int(info["size"]),
            "deleted": bool(info.get("deleted", False)),
            "object_metadata": json.dumps(info.get("meta") or {}, separators=(",", ":")).encode("utf-8"),
        }
        for path, info in sorted(manifest["files"].items())
    ]
    schema = pa.schema([
        pa.field("version", pa.string(), nullable=False),
        pa.field("shards_meta", pa.list_(pa.struct([
            pa.field("number", pa.int32(), nullable=False),
            pa.field("type", pa.string(), nullable=False),
            pa.field("total_size", pa.int64(), nullable=False),
        ])), nullable=False),
        pa.field("files", pa.list_(pa.struct([
            pa.field("path", pa.string(), nullable=False),
            pa.field("shard_id", pa.int32(), nullable=False),
            pa.field("offset", pa.int64(), nullable=False),
            pa.field("size", pa.int64(), nullable=False),
            pa.field("deleted", pa.bool_(), nullable=False),
            pa.field("object_metadata", pa.binary(), nullable=False),
        ])), nullable=False),
    ])
    table = pa.Table.from_pylist([
        {"version": manifest["version"], "shards_meta": shards, "files": files}
    ], schema=schema)
    pq.write_table(table, out / "metadata.parquet")


def read_parquet_manifest(root: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    table = pq.read_table(Path(root) / "metadata.parquet")
    rows = table.to_pylist()
    if not rows:
        raise ValueError(f"empty DatasetFS manifest: {root}")
    row = rows[0]
    manifest = {"version": row["version"], "shards_meta": {}, "files": {}}
    for shard in row["shards_meta"] or []:
        manifest["shards_meta"][str(int(shard["number"]))] = {
            "type": shard["type"],
            "total_size": int(shard["total_size"]),
        }
    for item in row["files"] or []:
        meta_raw = item.get("object_metadata") or b"{}"
        if isinstance(meta_raw, str):
            meta_raw = meta_raw.encode("utf-8")
        manifest["files"][item["path"]] = {
            "c_id": int(item["shard_id"]),
            "offset": int(item["offset"]),
            "size": int(item["size"]),
            "deleted": bool(item["deleted"]),
            "path": item["path"],
            "meta": json.loads(meta_raw.decode("utf-8")),
        }
    return manifest


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
            self._write_parquet_manifest()
            (self.out / ".done").touch()

    def _write_parquet_manifest(self) -> None:
        write_parquet_manifest(self.out, self.manifest)

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
