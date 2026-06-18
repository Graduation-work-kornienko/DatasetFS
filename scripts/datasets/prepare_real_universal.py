"""Prepare real-world universal DatasetFS datasets.

This module intentionally materializes bounded slices for huge corpora. The data
is real, but capped by --target-gb so the thesis universality matrix stays in
the 1-5GB range instead of turning into a storage project.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import tarfile
from pathlib import Path
from zipfile import ZipFile

from scripts.datasets.datasetfs_writer import DatasetFSWriter


RAW_ROOT = Path("data/raw/real_universal")
FMT_ROOT = Path("data/formats/real_universal")


def _target_bytes(gb: float) -> int:
    return int(gb * 1024 * 1024 * 1024)


class _ParquetTextSink:
    def __init__(self, out: Path, batch_size: int = 10_000):
        self.out = out
        self.batch_size = batch_size
        self.rows: list[dict] = []
        self.writer = None

    def __enter__(self):
        import pyarrow as pa

        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.schema = pa.schema([
            ("path", pa.string()),
            ("text", pa.string()),
            ("label", pa.string()),
        ])
        return self

    def add(self, path: str, text: str, label: str) -> None:
        self.rows.append({"path": path, "text": text, "label": label})
        if len(self.rows) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.out, self.schema, compression="zstd")
        self.writer.write_table(table)
        self.rows.clear()

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.flush()
        if self.writer is not None:
            self.writer.close()


def _download_file(url: str, dest: Path) -> None:
    import requests

    if dest.exists():
        print(f"[skip] {dest} already exists", flush=True)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    tmp.unlink(missing_ok=True)
    done = 0
    next_report = 512 * 1024 * 1024
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if done >= next_report:
                    print(f"[download] {dest.name}: {done / (1024**3):.1f} GiB", flush=True)
                    next_report += 512 * 1024 * 1024
    tmp.replace(dest)
    print(f"[download] done {dest} ({done / (1024**3):.2f} GiB)", flush=True)


def _prepare_hf_text(name: str, dataset_name: str, config: str | None, split: str, text_field: str,
                     label_field: str | None, target_gb: float) -> None:
    from datasets import load_dataset

    out = FMT_ROOT / name / "datasetfs"
    parquet_out = FMT_ROOT / name / "parquet" / "data.parquet"
    raw = RAW_ROOT / name
    raw.mkdir(parents=True, exist_ok=True)
    total = 0
    n = 0
    ds = load_dataset(dataset_name, config, split=split, streaming=True) if config else load_dataset(dataset_name, split=split, streaming=True)
    with DatasetFSWriter(out) as writer, _ParquetTextSink(parquet_out) as parquet:
        for row in ds:
            text = str(row.get(text_field) or "")
            if not text:
                continue
            payload = text.encode("utf-8", errors="replace")
            label = str(row.get(label_field)) if label_field and label_field in row else str(len(text) % 8)
            path = f"{n:09d}.txt"
            writer.add(path, payload, {"label": label, "text_len": len(text)})
            parquet.add(path, text, label)
            total += len(payload)
            n += 1
            if total >= _target_bytes(target_gb):
                break
    (raw / "README.txt").write_text(
        f"Materialized {n} rows / {total} bytes from {dataset_name} {config or ''} {split}\n",
        encoding="utf-8",
    )
    print(f"[done] {name}: {n} rows, {total / (1024**3):.2f} GiB -> {out}", flush=True)


def prepare_wikipedia_en_slice(target_gb: float) -> None:
    _prepare_hf_text(
        "wikipedia_en_slice",
        dataset_name="wikimedia/wikipedia",
        config="20231101.en",
        split="train",
        text_field="text",
        label_field=None,
        target_gb=target_gb,
    )


def prepare_amazon_polarity_slice(target_gb: float) -> None:
    """Prepare a bounded real text-classification corpus.

    Amazon Polarity is compact enough for a 1-2 GiB thesis benchmark slice, has
    real binary sentiment labels, and streams from HuggingFace without storing a
    separate raw archive first.
    """
    from datasets import load_dataset

    name = "amazon_polarity_slice"
    out = FMT_ROOT / name / "datasetfs"
    parquet_out = FMT_ROOT / name / "parquet" / "data.parquet"
    raw = RAW_ROOT / name
    raw.mkdir(parents=True, exist_ok=True)
    total = 0
    n = 0
    next_report = 100_000
    ds = load_dataset("fancyzhx/amazon_polarity", split="train", streaming=True)
    with DatasetFSWriter(out) as writer, _ParquetTextSink(parquet_out) as parquet:
        for row in ds:
            title = str(row.get("title") or "")
            content = str(row.get("content") or "")
            text = f"{title}\n{content}".strip()
            if not text:
                continue
            payload = text.encode("utf-8", errors="replace")
            label = str(row.get("label", ""))
            path = f"{n:09d}.txt"
            writer.add(
                path,
                payload,
                {"label": label, "source": "amazon_polarity", "text_len": len(text)},
            )
            parquet.add(path, text, label)
            total += len(payload)
            n += 1
            if n >= next_report:
                print(f"[{name}] rows={n} payload={total / (1024**3):.2f} GiB", flush=True)
                next_report += 100_000
            if total >= _target_bytes(target_gb):
                break
    (raw / "README.txt").write_text(
        f"Materialized {n} rows / {total} bytes from fancyzhx/amazon_polarity train\n",
        encoding="utf-8",
    )
    print(f"[done] {name}: {n} rows, {total / (1024**3):.2f} GiB -> {out}", flush=True)


def prepare_amazon_reviews_slice(target_gb: float) -> None:
    name = "amazon_reviews_slice"
    import json
    import requests

    source_url = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Home_and_Kitchen.jsonl"
    out = FMT_ROOT / name / "datasetfs"
    parquet_out = FMT_ROOT / name / "parquet" / "data.parquet"
    raw = RAW_ROOT / name
    raw.mkdir(parents=True, exist_ok=True)
    total = 0
    n = 0
    next_report = 100_000
    with requests.get(source_url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with DatasetFSWriter(out) as writer, _ParquetTextSink(parquet_out) as parquet:
            for line in resp.iter_lines(chunk_size=1024 * 1024):
                if not line:
                    continue
                row = json.loads(line)
                text = str(row.get("text") or "")
                if not text:
                    continue
                payload = text.encode("utf-8", errors="replace")
                rating = row.get("rating", "")
                path = f"{n:09d}.txt"
                writer.add(
                    path,
                    payload,
                    {"label": str(rating), "rating": rating, "text_len": len(text)},
                )
                parquet.add(path, text, str(rating))
                total += len(payload)
                n += 1
                if n >= next_report:
                    print(f"[amazon_reviews_slice] rows={n} payload={total / (1024**3):.2f} GiB", flush=True)
                    next_report += 100_000
                if total >= _target_bytes(target_gb):
                    break
    (raw / "README.txt").write_text(
        f"Materialized {n} rows / {total} bytes from {source_url}\n",
        encoding="utf-8",
    )
    print(f"[done] {name}: {n} rows, {total / (1024**3):.2f} GiB -> {out}", flush=True)


def prepare_librispeech_clean_100(target_gb: float | None = None) -> None:
    from datasets import Audio, load_dataset

    name = "librispeech_clean_100"
    raw = RAW_ROOT / "librispeech_clean_100"
    raw.mkdir(parents=True, exist_ok=True)
    out = FMT_ROOT / "librispeech_clean_100" / "datasetfs"
    total = 0
    limit = _target_bytes(target_gb) if target_gb else None
    n = 0
    next_report = 500
    ds = load_dataset("openslr/librispeech_asr", "clean", split="train.100", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    with DatasetFSWriter(out) as writer:
        for row in ds:
            payload = row["audio"].get("bytes") or b""
            if not payload:
                continue
            stem = str(row["id"])
            speaker = str(row["speaker_id"])
            writer.add(
                f"{stem}.flac",
                payload,
                {"label": speaker, "speaker": speaker, "text": row.get("text", "")},
            )
            total += len(payload)
            n += 1
            if n >= next_report:
                print(f"[{name}] clips={n} payload={total / (1024**3):.2f} GiB", flush=True)
                next_report += 500
            if limit and total >= limit:
                break
    (raw / "README.txt").write_text(
        f"Materialized {n} clips / {total} bytes from openslr/librispeech_asr clean train.100\n",
        encoding="utf-8",
    )
    print(f"[done] librispeech_clean_100: {n} clips, {total / (1024**3):.2f} GiB -> {out}", flush=True)


def _not_ready(name: str) -> None:
    raise SystemExit(
        f"{name} preparation needs a selected source/export layout. "
        "Use this script once we decide HF dataset id or local archive format."
    )


def prepare_flickr30k(target_gb: float) -> None:
    name = "flickr30k"
    raw = RAW_ROOT / name
    raw.mkdir(parents=True, exist_ok=True)
    zip_path = raw / "flickr30k-images.zip"
    ann_path = raw / "flickr_annotations_30k.csv"
    base = "https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main"
    _download_file(f"{base}/flickr30k-images.zip", zip_path)
    _download_file(f"{base}/flickr_annotations_30k.csv", ann_path)

    captions: dict[str, list[str]] = {}
    splits: dict[str, str] = {}
    with ann_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("filename") or row.get("image") or ""
            if not filename:
                continue
            caps = row.get("caption") or row.get("raw") or ""
            if caps:
                captions.setdefault(filename, []).append(caps)
            if row.get("split"):
                splits[filename] = row["split"]

    out = FMT_ROOT / name / "datasetfs"
    total = 0
    limit = _target_bytes(target_gb)
    n = 0
    next_report = 1000
    with ZipFile(zip_path) as zf, DatasetFSWriter(out) as writer:
        names = sorted(
            x for x in zf.namelist()
            if x.lower().endswith((".jpg", ".jpeg", ".png"))
            and "__macosx" not in x.lower()
            and not Path(x).name.startswith("._")
        )
        for member in names:
            filename = Path(member).name
            payload = zf.read(member)
            caps = captions.get(filename, [])
            writer.add(
                filename,
                payload,
                {
                    "label": splits.get(filename, "train"),
                    "caption": caps[0] if caps else "",
                    "captions": caps,
                    "filename": filename,
                },
            )
            total += len(payload)
            n += 1
            if n >= next_report:
                print(f"[{name}] images={n} payload={total / (1024**3):.2f} GiB", flush=True)
                next_report += 1000
            if total >= limit:
                break
    (raw / "README.txt").write_text(
        f"Materialized {n} images / {total} bytes from nlphuji/flickr30k\n",
        encoding="utf-8",
    )
    print(f"[done] {name}: {n} images, {total / (1024**3):.2f} GiB -> {out}", flush=True)


def prepare_ucf101_subset(target_gb: float) -> None:
    """Prepare a short-video action-recognition slice.

    The HF archive is a compact real UCF101 subset (~172 MB) from TensorFlow's
    video loading tutorial. It is intentionally small: this dataset is for video
    modality coverage in the universality suite, not for long throughput runs.
    """
    name = "ucf101_subset"
    raw = RAW_ROOT / name
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "UCF101_subset.tar.gz"
    _download_file(
        "https://huggingface.co/datasets/sayakpaul/ucf101-subset/resolve/main/UCF101_subset.tar.gz",
        archive,
    )

    extracted = raw / "extracted"
    if not extracted.exists():
        tmp = raw / "extracting"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(tmp, filter="data")
        tmp.replace(extracted)

    out = FMT_ROOT / name / "datasetfs"
    total = 0
    limit = _target_bytes(target_gb)
    n = 0
    video_exts = {".avi", ".mp4", ".mov", ".mkv", ".webm"}
    with DatasetFSWriter(out) as writer:
        for video in sorted(extracted.rglob("*")):
            if not video.is_file() or video.suffix.lower() not in video_exts:
                continue
            label = video.parent.name if video.parent != extracted else video.stem.split("_")[1]
            payload = video.read_bytes()
            writer.add(
                f"{n:08d}_{video.name}",
                payload,
                {"label": label, "source": "ucf101_subset", "filename": video.name},
            )
            total += len(payload)
            n += 1
            if total >= limit:
                break
    (raw / "README.txt").write_text(
        f"Materialized {n} videos / {total} bytes from sayakpaul/ucf101-subset\n",
        encoding="utf-8",
    )
    print(f"[done] {name}: {n} videos, {total / (1024**3):.2f} GiB -> {out}", flush=True)


def prepare_publaynet_slice(target_gb: float) -> None:
    """Build a bounded DatasetFS slice from prepared PubLayNet image bytes.

    The current PubLayNet raw extractor stores encoded page images under
    ``data/raw/publaynet/train/layout`` and does not persist COCO annotations.
    This still gives a real, large document-image dataset for the universality
    matrix; region metadata is represented as an empty list until annotation
    extraction is added to ``scripts/datasets/_publaynet.py``.
    """
    src = Path("data/raw/publaynet/train/layout")
    if not src.exists():
        raise SystemExit(
            f"PubLayNet raw images not found at {src}. Restore/build PubLayNet raw first."
        )
    out = FMT_ROOT / "publaynet_slice" / "datasetfs"
    limit = _target_bytes(target_gb)
    total = 0
    n = 0
    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    with DatasetFSWriter(out) as writer:
        for img in sorted(src.iterdir()):
            if not img.is_file() or img.suffix.lower() not in image_exts:
                continue
            payload = img.read_bytes()
            writer.add(
                f"{n:09d}_{img.name}",
                payload,
                {
                    "label": "layout",
                    "image_id": img.stem,
                    "regions": [],
                    "region_count": 0,
                },
            )
            total += len(payload)
            n += 1
            if total >= limit:
                break
    print(f"[done] publaynet_slice: {n} images, {total / (1024**3):.2f} GiB -> {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=[
        "wikipedia_en_slice",
        "amazon_polarity_slice",
        "amazon_reviews_slice",
        "librispeech_clean_100",
        "flickr30k",
        "ucf101_subset",
        "publaynet_slice",
    ])
    parser.add_argument("--target-gb", type=float, default=2.0)
    args = parser.parse_args()

    if args.dataset == "wikipedia_en_slice":
        prepare_wikipedia_en_slice(args.target_gb)
    elif args.dataset == "amazon_polarity_slice":
        prepare_amazon_polarity_slice(args.target_gb)
    elif args.dataset == "amazon_reviews_slice":
        prepare_amazon_reviews_slice(args.target_gb)
    elif args.dataset == "librispeech_clean_100":
        prepare_librispeech_clean_100(args.target_gb)
    elif args.dataset == "publaynet_slice":
        prepare_publaynet_slice(args.target_gb)
    elif args.dataset == "flickr30k":
        prepare_flickr30k(args.target_gb)
    elif args.dataset == "ucf101_subset":
        prepare_ucf101_subset(args.target_gb)


if __name__ == "__main__":
    main()
