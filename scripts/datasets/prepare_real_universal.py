"""Prepare real-world universal DatasetFS datasets.

This module intentionally materializes bounded slices for huge corpora. The data
is real, but capped by --target-gb so the thesis universality matrix stays in
the 2-5GB range instead of turning into a storage project.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.datasets.datasetfs_writer import DatasetFSWriter


RAW_ROOT = Path("data/raw/real_universal")
FMT_ROOT = Path("data/formats/real_universal")


def _target_bytes(gb: float) -> int:
    return int(gb * 1024 * 1024 * 1024)


def _prepare_hf_text(name: str, dataset_name: str, config: str | None, split: str, text_field: str,
                     label_field: str | None, target_gb: float) -> None:
    from datasets import load_dataset

    out = FMT_ROOT / name / "datasetfs"
    raw = RAW_ROOT / name
    raw.mkdir(parents=True, exist_ok=True)
    total = 0
    n = 0
    ds = load_dataset(dataset_name, config, split=split, streaming=True) if config else load_dataset(dataset_name, split=split, streaming=True)
    with DatasetFSWriter(out) as writer:
        for row in ds:
            text = str(row.get(text_field) or "")
            if not text:
                continue
            payload = text.encode("utf-8", errors="replace")
            label = str(row.get(label_field)) if label_field and label_field in row else str(len(text) % 8)
            writer.add(f"{n:09d}.txt", payload, {"label": label, "text_len": len(text)})
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


def prepare_amazon_reviews_slice(target_gb: float) -> None:
    _prepare_hf_text(
        "amazon_reviews_slice",
        dataset_name="amazon_reviews_multi",
        config="en",
        split="train",
        text_field="review_body",
        label_field="stars",
        target_gb=target_gb,
    )


def prepare_librispeech_clean_100(target_gb: float | None = None) -> None:
    import torchaudio

    raw = RAW_ROOT / "librispeech_clean_100"
    # torchaudio stores under raw/LibriSpeech/train-clean-100.
    torchaudio.datasets.LIBRISPEECH(str(raw), url="train-clean-100", download=True)
    corpus = raw / "LibriSpeech" / "train-clean-100"
    if not corpus.exists():
        raise RuntimeError(f"LibriSpeech corpus not found after download: {corpus}")

    transcripts = {}
    for txt in corpus.rglob("*.trans.txt"):
        for line in txt.read_text(encoding="utf-8").splitlines():
            key, _, text = line.partition(" ")
            transcripts[key] = text

    out = FMT_ROOT / "librispeech_clean_100" / "datasetfs"
    total = 0
    limit = _target_bytes(target_gb) if target_gb else None
    n = 0
    with DatasetFSWriter(out) as writer:
        for flac in sorted(corpus.rglob("*.flac")):
            payload = flac.read_bytes()
            stem = flac.stem
            speaker = stem.split("-")[0]
            writer.add(
                f"{stem}.flac",
                payload,
                {"label": speaker, "speaker": speaker, "text": transcripts.get(stem, "")},
            )
            total += len(payload)
            n += 1
            if limit and total >= limit:
                break
    print(f"[done] librispeech_clean_100: {n} clips, {total / (1024**3):.2f} GiB -> {out}", flush=True)


def _not_ready(name: str) -> None:
    raise SystemExit(
        f"{name} preparation needs a selected source/export layout. "
        "Use this script once we decide HF dataset id or local archive format."
    )


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
        "amazon_reviews_slice",
        "librispeech_clean_100",
        "flickr30k",
        "publaynet_slice",
    ])
    parser.add_argument("--target-gb", type=float, default=2.0)
    args = parser.parse_args()

    if args.dataset == "wikipedia_en_slice":
        prepare_wikipedia_en_slice(args.target_gb)
    elif args.dataset == "amazon_reviews_slice":
        prepare_amazon_reviews_slice(args.target_gb)
    elif args.dataset == "librispeech_clean_100":
        prepare_librispeech_clean_100(args.target_gb)
    elif args.dataset == "publaynet_slice":
        prepare_publaynet_slice(args.target_gb)
    elif args.dataset == "flickr30k":
        _not_ready(args.dataset)


if __name__ == "__main__":
    main()
