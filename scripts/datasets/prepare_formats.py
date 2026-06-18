"""Prepare a fastai dataset in 4 storage formats: ImageFolder, WebDataset, HuggingFace, DatasetFS.

Usage:
    python -m scripts.datasets.prepare_formats                  # both datasets, all formats
    python -m scripts.datasets.prepare_formats imagenette       # one dataset
    python -m scripts.datasets.prepare_formats --formats datasetfs webdataset
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.datasets._fastai import ALL_DATASETS, FastaiDataset, ensure_dataset
from scripts.datasets._publaynet import PUBLAYNET, PubLayNetDataset, ensure_publaynet

# All datasets the CLI can prepare. PubLayNet is acquired from HF Parquet (not a
# fastai .tgz) so it carries its own acquisition path; see prepare()'s dispatch.
KNOWN_DATASETS = ALL_DATASETS + (PUBLAYNET,)


ALL_FORMATS = (
    "imagefolder", "webdataset", "huggingface", "datasetfs",
    # Format-matrix (G1): same files, different storage engines.
    "lmdb", "hdf5", "tfrecord", "ffcv",
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".jpe", ".png"}
AUDIO_SUFFIXES = {".wav"}


def _sample_suffixes(ds: FastaiDataset) -> set[str]:
    return AUDIO_SUFFIXES if ds.name == "speech_commands" else IMAGE_SUFFIXES


def _is_valid_sample_file(path: Path, suffixes: set[str]) -> bool:
    return path.is_file() and not path.name.startswith("._") and path.suffix.lower() in suffixes


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
            if _is_valid_sample_file(f, _sample_suffixes(ds)):
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
        dst = out / class_name
        dst.mkdir(parents=True, exist_ok=True)
        for sample in sorted(src.iterdir()):
            if _is_valid_sample_file(sample, _sample_suffixes(ds)):
                os.symlink(sample, dst / sample.name)

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


# ---- format-matrix engines (G1): same (file, label) set, different storage ----
# All read the identical filtered sample list (_list_samples → ds.classes only),
# store the RAW file bytes + the class-name string, and write a `.done` marker.
# Labels are kept as strings so loaders map them through the SAME runtime
# label_to_idx every other format uses — no prep-time index drift.


def prepare_lmdb(ds: FastaiDataset, extracted: Path, out: Path) -> None:
    """LMDB key-value store: key=f"{idx:08d}", value=pickle{data,label}.
    A `__keys__` entry holds the ordered key list (the dataset length/iteration
    order) so the loader needn't enumerate the env."""
    marker = out / ".done"
    if marker.exists():
        print(f"[skip] {ds.name}/lmdb already prepared", flush=True)
        return
    import lmdb

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    samples = _list_samples(ds, _train_dir(ds, extracted))
    total = sum(p.stat().st_size for p, _ in samples)
    # Generous, sparse map_size: raw bytes + pickle overhead + headroom.
    map_size = int(total * 1.5) + 256 * 1024 * 1024
    print(f"[lmdb] {ds.name}: writing {len(samples)} samples (~{total/1e6:.0f} MB)", flush=True)

    env = lmdb.open(str(out), map_size=map_size, subdir=True, writemap=False)
    keys: list[str] = []
    with env.begin(write=True) as txn:
        for idx, (img_path, label) in enumerate(samples):
            key = f"{idx:08d}"
            with open(img_path, "rb") as f:
                data = f.read()
            txn.put(key.encode(), pickle.dumps({"data": data, "label": label},
                                               protocol=pickle.HIGHEST_PROTOCOL))
            keys.append(key)
        txn.put(b"__keys__", pickle.dumps(keys, protocol=pickle.HIGHEST_PROTOCOL))
    env.sync()
    env.close()

    marker.touch()
    print(f"[done] {ds.name}/lmdb", flush=True)


def prepare_hdf5(ds: FastaiDataset, extracted: Path, out: Path) -> None:
    """Single HDF5 file with a variable-length uint8 `data` dataset (raw file
    bytes per sample) and a string `labels` dataset."""
    marker = out / ".done"
    if marker.exists():
        print(f"[skip] {ds.name}/hdf5 already prepared", flush=True)
        return
    import h5py
    import numpy as np

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    samples = _list_samples(ds, _train_dir(ds, extracted))
    print(f"[hdf5] {ds.name}: writing {len(samples)} samples", flush=True)

    h5_path = out / "data.h5"
    with h5py.File(h5_path, "w") as h5:
        n = len(samples)
        data_ds = h5.create_dataset("data", (n,), dtype=h5py.vlen_dtype(np.uint8))
        label_ds = h5.create_dataset("labels", (n,), dtype=h5py.string_dtype())
        for idx, (img_path, label) in enumerate(samples):
            with open(img_path, "rb") as f:
                raw = f.read()
            data_ds[idx] = np.frombuffer(raw, dtype=np.uint8)
            label_ds[idx] = label

    marker.touch()
    print(f"[done] {ds.name}/hdf5", flush=True)


def prepare_tfrecord(ds: FastaiDataset, extracted: Path, out: Path) -> None:
    """A single `data.tfrecord` of Example{image:bytes, label:bytes} plus the
    `data.index` the tfrecord torch reader needs for sharded random access."""
    marker = out / ".done"
    if marker.exists():
        print(f"[skip] {ds.name}/tfrecord already prepared", flush=True)
        return
    from tfrecord import TFRecordWriter
    from tfrecord.tools.tfrecord2idx import create_index

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    samples = _list_samples(ds, _train_dir(ds, extracted))
    print(f"[tfrecord] {ds.name}: writing {len(samples)} samples", flush=True)

    rec_path = out / "data.tfrecord"
    writer = TFRecordWriter(str(rec_path))
    for img_path, label in samples:
        with open(img_path, "rb") as f:
            data = f.read()
        writer.write({"image": (data, "byte"), "label": (label.encode("utf-8"), "byte")})
    writer.close()

    create_index(str(rec_path), str(out / "data.index"))

    marker.touch()
    print(f"[done] {ds.name}/tfrecord", flush=True)


def prepare_ffcv(ds: FastaiDataset, extracted: Path, out: Path) -> None:
    """FFCV `.beton`. Linux-only: FFCV has no macOS wheels. On darwin we write a
    `.skipped` marker and return so the format-matrix prep stays a no-op there;
    the loader mirrors this gate. See requirements-linux.txt."""
    if sys.platform == "darwin":
        out.mkdir(parents=True, exist_ok=True)
        (out / ".skipped").write_text(
            "FFCV is Linux-only; skipped on darwin. See requirements-linux.txt.\n"
        )
        print(f"[skip] {ds.name}/ffcv (Linux-only, darwin host)", flush=True)
        return

    marker = out / ".done"
    if marker.exists():
        print(f"[skip] {ds.name}/ffcv already prepared", flush=True)
        return
    try:
        import numpy as np
        from ffcv.writer import DatasetWriter
        from ffcv.fields import RGBImageField, IntField
        from PIL import Image
    except ImportError:
        print("[error] ffcv not installed; pip install -r requirements-linux.txt", file=sys.stderr)
        raise

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    samples = _list_samples(ds, _train_dir(ds, extracted))
    labels = sorted({label for _, label in samples})
    label_to_idx = {l: i for i, l in enumerate(labels)}
    (out / "label_to_idx.json").write_text(json.dumps(label_to_idx))

    class _PILDataset:
        def __len__(self):
            return len(samples)

        def __getitem__(self, i):
            path, label = samples[i]
            return np.array(Image.open(path).convert("RGB")), label_to_idx[label]

    beton = out / "data.beton"
    writer = DatasetWriter(str(beton), {
        "image": RGBImageField(write_mode="jpg"),
        "label": IntField(),
    })
    writer.from_indexed_dataset(_PILDataset())

    marker.touch()
    print(f"[done] {ds.name}/ffcv", flush=True)


def _ensure_converter_binary(repo_root: Path) -> Path:
    binary = repo_root / "bin" / "datasetfs"
    binary.parent.mkdir(parents=True, exist_ok=True)
    print(f"[go build] datasetfs → {binary}", flush=True)
    # cgo for libjpeg-turbo: the converter subcommand shares the daemon's binary,
    # which pulls in internal/pipeline. Mirror Makefile's CGO_ENV.
    env = {
        **os.environ,
        "CGO_ENABLED": "1",
        "PKG_CONFIG_PATH": (
            "/opt/homebrew/opt/jpeg-turbo/lib/pkgconfig"
            + (":" + os.environ["PKG_CONFIG_PATH"] if "PKG_CONFIG_PATH" in os.environ else "")
        ),
    }
    subprocess.run(
        ["go", "build", "-o", str(binary), "./cmd/datasetfs"],
        cwd=repo_root,
        env=env,
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
        [str(binary), "converter", "dataset-folder", "--source", str(class_root), "--target", str(out)],
        cwd=repo_root,
        check=True,
    )

    marker.touch()
    print(f"[done] {ds.name}/datasetfs", flush=True)


def prepare(ds, data_root: Path, repo_root: Path, formats: tuple[str, ...],
            n_shards: int | None = None) -> None:
    # Acquisition dispatch: fastai datasets download a .tgz; PubLayNet pulls HF
    # Parquet shards. Both return an extracted root with a <train>/<class>/ tree.
    if isinstance(ds, PubLayNetDataset):
        extracted = ensure_publaynet(ds, data_root, n_shards=n_shards)
    else:
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
    if "lmdb" in formats:
        prepare_lmdb(ds, extracted, formats_root / "lmdb")
    if "hdf5" in formats:
        prepare_hdf5(ds, extracted, formats_root / "hdf5")
    if "tfrecord" in formats:
        prepare_tfrecord(ds, extracted, formats_root / "tfrecord")
    if "ffcv" in formats:
        prepare_ffcv(ds, extracted, formats_root / "ffcv")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="Dataset name(s); default: all fastai datasets")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--formats", nargs="*", default=ALL_FORMATS, choices=ALL_FORMATS)
    parser.add_argument("--n-shards", type=int, default=None,
                        help="PubLayNet only: number of HF train Parquet shards to pull "
                             "(default 85 ≈ 40 GB extracted).")
    parser.add_argument("--rm-raw-after", action="store_true",
                        help="Delete data/raw/<ds> after all formats are built (the "
                             "self-contained formats no longer need it; imagefolder, "
                             "which symlinks into raw, is excluded if this is set).")
    args = parser.parse_args()

    # Default name set = the fastai datasets (PubLayNet is large/explicit → opt-in).
    requested = set(args.names) if args.names else {ds.name for ds in ALL_DATASETS}
    known = {ds.name for ds in KNOWN_DATASETS}
    unknown = requested - known
    if unknown:
        parser.error(f"unknown dataset(s): {unknown}")

    data_root = Path(args.data_root).resolve()
    repo_root = Path(__file__).resolve().parent.parent.parent
    formats = tuple(args.formats)

    for ds in KNOWN_DATASETS:
        if ds.name in requested:
            prepare(ds, data_root, repo_root, formats, n_shards=args.n_shards)
            if args.rm_raw_after:
                raw = data_root / "raw" / ds.name
                if raw.exists():
                    print(f"[rm-raw] {raw}", flush=True)
                    shutil.rmtree(raw)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
