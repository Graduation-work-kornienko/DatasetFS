"""Prepare a fastai dataset in 4 storage formats: ImageFolder, WebDataset, HuggingFace, DatasetFS.

Usage:
    python -m scripts.datasets.prepare_formats                  # both datasets, all formats
    python -m scripts.datasets.prepare_formats imagenette       # one dataset
    python -m scripts.datasets.prepare_formats --formats datasetfs webdataset
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.datasets._fastai import ALL_DATASETS, FastaiDataset, ensure_dataset


ALL_FORMATS = ("imagefolder", "webdataset", "huggingface", "datasetfs")


def _train_dir(ds: FastaiDataset, extracted: Path) -> Path:
    """Root containing class subdirs. May be extracted/train (split layout)
    or extracted/ itself (flat layout, e.g., Speech Commands)."""
    return extracted / ds.train_subdir if ds.train_subdir else extracted


def _list_samples(ds: FastaiDataset, train: Path) -> list[tuple[Path, str]]:
    """Yield (file_path, class_name) for every file in every known class dir.

    Filters strictly by ds.classes — ignores extra dirs like `_background_noise_`
    and stray files like README/LICENSE that some datasets ship.
    """
    classes = set(ds.classes)
    samples: list[tuple[Path, str]] = []
    for class_dir in sorted(p for p in train.iterdir() if p.is_dir()):
        if class_dir.name not in classes:
            continue
        for f in sorted(class_dir.iterdir()):
            if f.is_file():
                samples.append((f, class_dir.name))
    return samples


def prepare_imagefolder(ds: FastaiDataset, extracted: Path, out: Path) -> None:
    """Materialise a clean ImageFolder root containing ONLY ds.classes —
    one symlink per class, so other consumers (torchvision.ImageFolder,
    ground-truth walks) see exactly the same set of files DatasetFS sees."""
    marker = out / ".done"
    # New format = real dir with per-class symlinks. Old format was a single
    # symlink at `out`. If we see a symlink we know we need to re-prep.
    if out.is_dir() and not out.is_symlink() and marker.exists():
        print(f"[skip] {ds.name}/imagefolder already prepared", flush=True)
        return
    if out.is_symlink():
        out.unlink()
    elif out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    train_root = _train_dir(ds, extracted)
    for class_name in ds.classes:
        src = train_root / class_name
        if not src.is_dir():
            raise RuntimeError(f"missing class dir {src} for dataset {ds.name}")
        os.symlink(src, out / class_name, target_is_directory=True)

    marker.touch()
    print(f"[done] {ds.name}/imagefolder ({len(ds.classes)} class symlinks → {train_root})", flush=True)


def prepare_webdataset(ds: FastaiDataset, extracted: Path, out: Path) -> None:
    marker = out / ".done"
    if marker.exists():
        print(f"[skip] {ds.name}/webdataset already prepared", flush=True)
        return
    try:
        from webdataset import ShardWriter
    except ImportError:
        print("[error] `webdataset` package not installed; pip install webdataset", file=sys.stderr)
        raise

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    samples = _list_samples(ds, _train_dir(ds, extracted))
    print(f"[wds] {ds.name}: writing {len(samples)} samples", flush=True)

    pattern = str(out / "shard-%06d.tar")
    with ShardWriter(pattern, maxsize=500 * 1024 * 1024) as sink:
        for idx, (img_path, label) in enumerate(samples):
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            ext = img_path.suffix.lower().lstrip(".") or "jpg"
            if ext in ("jpeg", "jpe"):
                ext = "jpg"
            sink.write({
                "__key__": f"{idx:08d}",
                ext: img_bytes,
                "cls": label.encode("utf-8"),
            })

    marker.touch()
    print(f"[done] {ds.name}/webdataset", flush=True)


def prepare_huggingface(ds: FastaiDataset, extracted: Path, out: Path) -> None:
    marker = out / ".done"
    if marker.exists():
        print(f"[skip] {ds.name}/huggingface already prepared", flush=True)
        return
    try:
        from datasets import ClassLabel, Dataset, Features, Image
    except ImportError:
        print("[error] `datasets` package not installed; pip install datasets", file=sys.stderr)
        raise

    if out.exists():
        shutil.rmtree(out)

    # Don't rely on HF's `imagefolder` auto-builder — its split heuristics get
    # confused by sibling val/ directories. Build the dataset explicitly from
    # the known train file list.
    samples = _list_samples(ds, _train_dir(ds, extracted))
    labels = sorted({label for _, label in samples})
    label_to_idx = {l: i for i, l in enumerate(labels)}

    print(f"[hf] {ds.name}: building Arrow dataset ({len(samples)} samples, {len(labels)} classes)", flush=True)

    def gen():
        for img_path, label in samples:
            yield {"image": str(img_path), "label": label_to_idx[label]}

    features = Features({"image": Image(), "label": ClassLabel(names=labels)})
    hfds = Dataset.from_generator(gen, features=features)
    hfds.save_to_disk(str(out))

    marker.touch()
    print(f"[done] {ds.name}/huggingface", flush=True)


def _ensure_converter_binary(repo_root: Path) -> Path:
    binary = repo_root / "bin" / "dataset_converter"
    binary.parent.mkdir(parents=True, exist_ok=True)
    print(f"[go build] dataset_converter → {binary}", flush=True)
    subprocess.run(
        ["go", "build", "-o", str(binary), "./cmd/dataset_converter"],
        cwd=repo_root,
        check=True,
    )
    return binary


def prepare_datasetfs(ds: FastaiDataset, class_root: Path, out: Path, repo_root: Path) -> None:
    """`class_root`: a directory whose immediate subdirs are class names from
    ds.classes (and ONLY those — typically the prepared imagefolder)."""
    marker = out / ".done"
    if marker.exists():
        print(f"[skip] {ds.name}/datasetfs already prepared", flush=True)
        return
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    binary = _ensure_converter_binary(repo_root)

    print(f"[dfs] {ds.name}: converting {class_root} → {out}", flush=True)
    subprocess.run(
        [str(binary), "dataset-folder", "--source", str(class_root), "--target", str(out)],
        cwd=repo_root,
        check=True,
    )

    marker.touch()
    print(f"[done] {ds.name}/datasetfs", flush=True)


def prepare(ds: FastaiDataset, data_root: Path, repo_root: Path, formats: tuple[str, ...]) -> None:
    extracted = ensure_dataset(ds, data_root)
    formats_root = data_root / "formats" / ds.name

    # ImageFolder is the canonical filtered source — every other format builds
    # from the same per-class symlink tree so they all see exactly the same files.
    imagefolder_path = formats_root / "imagefolder"
    prepare_imagefolder(ds, extracted, imagefolder_path)

    if "webdataset" in formats:
        prepare_webdataset(ds, extracted, formats_root / "webdataset")
    if "huggingface" in formats:
        prepare_huggingface(ds, extracted, formats_root / "huggingface")
    if "datasetfs" in formats:
        # Use the filtered imagefolder so the Go converter sees ONLY ds.classes
        # (avoids treating _background_noise_ etc. as classes for Speech Commands).
        prepare_datasetfs(ds, imagefolder_path, formats_root / "datasetfs", repo_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="Dataset name(s); default: all")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--formats", nargs="*", default=ALL_FORMATS, choices=ALL_FORMATS)
    args = parser.parse_args()

    requested = set(args.names) if args.names else {ds.name for ds in ALL_DATASETS}
    unknown = requested - {ds.name for ds in ALL_DATASETS}
    if unknown:
        parser.error(f"unknown dataset(s): {unknown}")

    data_root = Path(args.data_root).resolve()
    repo_root = Path(__file__).resolve().parent.parent.parent
    formats = tuple(args.formats)

    for ds in ALL_DATASETS:
        if ds.name in requested:
            prepare(ds, data_root, repo_root, formats)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
