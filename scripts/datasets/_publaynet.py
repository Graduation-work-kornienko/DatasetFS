"""PubLayNet acquisition for the >RAM format-matrix + remote-streaming benchmark.

PubLayNet is distributed on HuggingFace as 199 train Parquet shards
(``creative-graphic-design/PubLayNet``, ~506 MB each, ~100 GB total). Each shard
yields ~471 MB of extracted JPEGs (the bytes training actually reads), so we pull
the first ``n_shards`` (default 85 ≈ 40 GB extracted, > the 36 GB RAM → defeats
page cache) and extract the encoded image bytes into a single **dummy class** dir so
the existing ``prepare_formats`` pipeline (which keys everything off a
``<train>/<class>/<file>`` tree filtered by ``ds.classes``) works unchanged.

PubLayNet has no classification label — only COCO layout annotations — so a
single class ``layout`` is used. The benchmark measures I/O / cache / transport
throughput; loss/accuracy are meaningless with one class and are NOT analysed
(documented in the bench configs).

Memory: a 506 MB parquet of images would explode if read whole, so we iterate
**row-group by row-group** and write each image straight to disk, deleting the
parquet immediately after to stay within the local disk budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.datasets._fastai import download_with_progress


HF_REPO = "creative-graphic-design/PubLayNet"
HF_SHARD_URL = (
    "https://huggingface.co/datasets/" + HF_REPO
    + "/resolve/main/data/train-{i:05d}-of-00199.parquet"
)
TOTAL_TRAIN_SHARDS = 199


@dataclass(frozen=True)
class PubLayNetDataset:
    """Duck-compatible with FastaiDataset for everything prepare_formats reads
    (``name``, ``classes``, ``train_subdir``). Acquisition differs — it is HF
    Parquet, not a fastai .tgz — so ``prepare()`` dispatches to ``ensure_publaynet``
    instead of ``ensure_dataset``."""
    name: str = "publaynet"
    classes: tuple[str, ...] = ("layout",)
    train_subdir: str = "train"
    n_shards: int = 85  # ~40 GB extracted (471 MB/shard); override via --n-shards


PUBLAYNET = PubLayNetDataset()


def _sniff_ext(data: bytes) -> str:
    """Pick a file extension from the encoded-image magic bytes."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    return "jpg"  # PubLayNet images are JPEG/PNG; default to jpg


def _extract_image_bytes(cell) -> bytes | None:
    """The HF Image feature is stored in Parquet as a struct {bytes, path}.
    Be defensive about the exact shape pyarrow hands back."""
    if cell is None:
        return None
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    if isinstance(cell, dict):
        b = cell.get("bytes")
        if b:
            return bytes(b)
        path = cell.get("path")
        if path and Path(path).is_file():
            return Path(path).read_bytes()
    return None


def ensure_publaynet(ds: PubLayNetDataset, data_root: Path, n_shards: int | None = None) -> Path:
    """Download + extract the first ``n_shards`` PubLayNet train shards into
    ``data_root/raw/publaynet/train/layout/``. Idempotent (per-shard markers).
    Returns the extracted dataset root (containing ``train/``)."""
    import pyarrow.parquet as pq

    n_shards = n_shards if n_shards is not None else ds.n_shards
    if n_shards > TOTAL_TRAIN_SHARDS:
        n_shards = TOTAL_TRAIN_SHARDS

    raw_dir = data_root / "raw" / "publaynet"
    class_dir = raw_dir / ds.train_subdir / "layout"
    class_dir.mkdir(parents=True, exist_ok=True)
    done_marker = raw_dir / ".publaynet.done"

    if done_marker.exists():
        meta = done_marker.read_text().strip()
        if meta == str(n_shards):
            print(f"[skip] publaynet already prepared ({n_shards} shards) at {raw_dir}", flush=True)
            return raw_dir
        print(f"[publaynet] marker says {meta} shards, want {n_shards}; continuing", flush=True)

    tmp_dir = raw_dir / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    n_images = 0
    for i in range(n_shards):
        shard_marker = raw_dir / f".shard_{i:05d}.done"
        if shard_marker.exists():
            print(f"[skip] publaynet shard {i} already extracted", flush=True)
            continue

        url = HF_SHARD_URL.format(i=i)
        parquet_tmp = tmp_dir / f"train-{i:05d}.parquet"
        download_with_progress(url, parquet_tmp)

        try:
            pf = pq.ParquetFile(parquet_tmp)
            written = 0
            for batch in pf.iter_batches(batch_size=64, columns=["image_id", "image"]):
                ids = batch.column("image_id").to_pylist()
                imgs = batch.column("image").to_pylist()
                for img_id, cell in zip(ids, imgs):
                    data = _extract_image_bytes(cell)
                    if not data:
                        continue
                    ext = _sniff_ext(data)
                    (class_dir / f"{img_id}.{ext}").write_bytes(data)
                    written += 1
            n_images += written
            print(f"[publaynet] shard {i}: extracted {written} images "
                  f"(running total {n_images})", flush=True)
        finally:
            # Disk discipline: drop the 506 MB parquet as soon as it's extracted.
            parquet_tmp.unlink(missing_ok=True)

        shard_marker.touch()

    # Best-effort tmp cleanup.
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    done_marker.write_text(str(n_shards))
    print(f"[done] publaynet: {n_shards} shards → {class_dir} "
          f"({sum(1 for _ in class_dir.iterdir())} files)", flush=True)
    return raw_dir
