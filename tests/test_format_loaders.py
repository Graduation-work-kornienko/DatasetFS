"""Format-matrix invariant test (G1): every storage format is built from the
SAME filtered file set, so each loader must yield batches of the right shape and
labels drawn from the SAME class set. Guards against prep/loader drift.

Skips a format whose data isn't prepared (raises FormatUnavailable) — e.g. FFCV
on macOS. The DatasetFS loader is exercised by the daemon-based suites, not here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.datasetfs_bench.loaders.base import FormatUnavailable
from benchmarks.datasetfs_bench.loaders.ffcv_loader import FFCVLoader
from benchmarks.datasetfs_bench.loaders.hdf5_loader import HDF5Loader
from benchmarks.datasetfs_bench.loaders.hf_loader import HuggingFaceLoader
from benchmarks.datasetfs_bench.loaders.imagefolder import ImageFolderLoader
from benchmarks.datasetfs_bench.loaders.lmdb_loader import LMDBLoader
from benchmarks.datasetfs_bench.loaders.synthetic import SyntheticLoader
from benchmarks.datasetfs_bench.loaders.tfrecord_loader import TFRecordLoader

IMAGE_SIZE = 64
BATCH = 16
FORMATS_ROOT = _REPO_ROOT / "data" / "formats" / "imagenette"

# (loader_cls, root_subdir or None for synthetic)
CASES = [
    (ImageFolderLoader, "imagefolder"),
    (LMDBLoader, "lmdb"),
    (HDF5Loader, "hdf5"),
    (TFRecordLoader, "tfrecord"),
    (HuggingFaceLoader, "huggingface"),
    (FFCVLoader, "ffcv"),
    (SyntheticLoader, None),
]

pytestmark = pytest.mark.skipif(
    not (FORMATS_ROOT / "imagefolder" / ".done").exists(),
    reason="imagenette not prepared; run `make data-imagenette` + extra-format prep",
)


def _label_to_idx() -> dict[str, int]:
    imagefolder = FORMATS_ROOT / "imagefolder"
    classes = sorted(p.name for p in imagefolder.iterdir() if p.is_dir())
    return {c: i for i, c in enumerate(classes)}


def _spec(root_subdir, label_to_idx):
    spec = {
        "batch_size": BATCH,
        "num_workers": 0,
        "image_size": IMAGE_SIZE,
        "label_to_idx": label_to_idx,
        "seed": 0,
    }
    if root_subdir is not None:
        spec["root"] = str(FORMATS_ROOT / root_subdir)
    else:
        spec["synthetic_samples"] = 64
    return spec


@pytest.mark.parametrize("loader_cls,root_subdir", CASES, ids=[c[0].name for c in CASES])
def test_format_yields_consistent_batch(loader_cls, root_subdir):
    label_to_idx = _label_to_idx()
    loader = loader_cls(_spec(root_subdir, label_to_idx))
    try:
        loader.setup()
    except FormatUnavailable as e:
        pytest.skip(f"{loader_cls.name} unavailable: {e}")

    dl = loader.make_loader()
    images, targets = next(iter(dl))
    loader.teardown()

    assert images.ndim == 4 and images.shape[1] == 3, f"{loader_cls.name}: bad image shape {images.shape}"
    assert images.shape[2] == IMAGE_SIZE and images.shape[3] == IMAGE_SIZE
    assert images.dtype == torch.float32
    assert targets.shape[0] == images.shape[0]
    n_classes = len(label_to_idx)
    assert int(targets.min()) >= 0 and int(targets.max()) < n_classes, (
        f"{loader_cls.name}: labels out of range [0,{n_classes})"
    )
