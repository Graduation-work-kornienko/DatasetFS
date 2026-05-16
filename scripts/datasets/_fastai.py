"""Helpers for downloading fastai imageclas datasets."""
from __future__ import annotations

import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass(frozen=True)
class FastaiDataset:
    name: str
    url: str
    tgz_filename: str
    extracted_dir: str
    classes: tuple[str, ...]
    # Layout: where class dirs live relative to extracted_dir.
    # "" means class dirs are at the root (e.g., Speech Commands).
    # "train" means split-style layout (e.g., Imagenette has train/ + val/).
    train_subdir: str = "train"
    # True when the tarball has NO top-level dir of its own. We then create
    # `raw/<extracted_dir>/` and extract into it.
    flat_archive: bool = False


IMAGENETTE2 = FastaiDataset(
    name="imagenette",
    url="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
    tgz_filename="imagenette2.tgz",
    extracted_dir="imagenette2",
    classes=(
        "n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
        "n03394916", "n03417042", "n03425413", "n03445777", "n03888257",
    ),
)

IMAGEWOOF2 = FastaiDataset(
    name="imagewoof",
    url="https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz",
    tgz_filename="imagewoof2.tgz",
    extracted_dir="imagewoof2",
    classes=(
        "n02086240", "n02087394", "n02088364", "n02089973", "n02093754",
        "n02096294", "n02099601", "n02105641", "n02111889", "n02115641",
    ),
)

SPEECH_COMMANDS_V2 = FastaiDataset(
    name="speech_commands",
    url="https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
    tgz_filename="speech_commands_v0.02.tar.gz",
    extracted_dir="speech_commands_v2",
    classes=(
        "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow",
        "backward", "follow", "forward", "learn", "visual",
    ),
    train_subdir="",       # class dirs are at the root
    flat_archive=True,     # tarball has no top-level dir
)


ALL_DATASETS = (IMAGENETTE2, IMAGEWOOF2, SPEECH_COMMANDS_V2)


def download_with_progress(url: str, dest: Path, chunk_size: int = 1024 * 1024) -> None:
    """Stream-download a URL to dest, printing progress."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    print(f"[download] {url} → {dest}", flush=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        last_print = time.time()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_print > 2.0:
                    pct = (100 * downloaded / total) if total else 0
                    mb = downloaded / 1e6
                    print(f"  {mb:.1f} MB ({pct:.1f}%)", flush=True)
                    last_print = now
    tmp.rename(dest)
    print(f"[download] done: {dest}", flush=True)


def extract_tgz(tgz: Path, dest: Path) -> None:
    """Extract a .tgz archive."""
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {tgz} → {dest}", flush=True)
    with tarfile.open(tgz, "r:gz") as tf:
        # Python 3.12+: use the safer 'data' filter
        if sys.version_info >= (3, 12):
            tf.extractall(dest, filter="data")
        else:
            tf.extractall(dest)
    print(f"[extract] done", flush=True)


def ensure_dataset(ds: FastaiDataset, data_root: Path) -> Path:
    """Ensure a fastai dataset is downloaded + extracted under data_root/raw/.

    Idempotent: returns the extracted path if marker file present; otherwise
    downloads + extracts. Returns the extracted dataset directory (containing
    train/ and val/ subdirs).
    """
    raw_dir = data_root / "raw"
    tgz_path = raw_dir / ds.tgz_filename
    extracted = raw_dir / ds.extracted_dir
    marker = raw_dir / f".{ds.name}.done"

    if marker.exists() and extracted.exists():
        print(f"[skip] {ds.name} already prepared at {extracted}", flush=True)
        return extracted

    if not tgz_path.exists():
        download_with_progress(ds.url, tgz_path)

    if not extracted.exists():
        if ds.flat_archive:
            # Tarball has no top-level dir; create the target subdir and
            # extract into it so files don't pollute raw/.
            extracted.mkdir(parents=True, exist_ok=True)
            extract_tgz(tgz_path, extracted)
        else:
            # Tarball includes its own top-level dir matching extracted_dir.
            extract_tgz(tgz_path, raw_dir)

    if not extracted.exists():
        raise RuntimeError(f"extraction did not produce {extracted}")

    marker.touch()
    return extracted
