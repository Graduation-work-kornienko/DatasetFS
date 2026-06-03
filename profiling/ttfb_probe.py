"""TTFB phase breakdown probe (opt-04 investigation).

Splits DataSetFS time-to-first-batch into POST(/initialize_loading), iter setup,
and first-batch wait, for num_workers in {0,4}, across 2 epochs. Compares against
ImageFolder at the same num_workers to isolate the macOS DataLoader spawn cost
(spawn start method) from DatasetFS daemon priming.

Assumes a daemon is already running on localhost:51409 pointed at
data/formats/imagenette/datasetfs.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from clients.python import DatasetFS  # noqa: E402
from benchmarks.datasetfs_bench.loaders._common import (  # noqa: E402
    make_image_transform, bound_dfs_collate, imagefolder_collate,
)

IMAGEFOLDER = Path("data/formats/imagenette/imagefolder")
LABELS = {c: i for i, c in enumerate(sorted(p.name for p in IMAGEFOLDER.iterdir() if p.is_dir()))}
TF = make_image_transform(96)


def time_first_batch(make_dl, n_steady=20):
    t0 = time.perf_counter()
    dl = make_dl()
    it = iter(dl)
    t1 = time.perf_counter()       # DataLoader+iter constructed
    next(it)
    t2 = time.perf_counter()       # first batch (includes worker spawn)
    tA = time.perf_counter()
    n = 0
    for _ in range(n_steady):
        next(it)
        n += 1
    tB = time.perf_counter()
    del it, dl
    return t1 - t0, t2 - t1, (tB - tA) / max(n, 1) * 1000


def run_dfs(nw):
    print(f"\n===== DatasetFS num_workers={nw} =====")
    for epoch in range(2):
        tp = time.perf_counter()
        ds = DatasetFS(num_workers=nw, seed=0, transform=TF, daemon_url="http://localhost:51409")
        post = time.perf_counter() - tp

        def make_dl():
            return DataLoader(ds, batch_size=64, num_workers=nw,
                              collate_fn=bound_dfs_collate(LABELS),
                              persistent_workers=False, pin_memory=False)
        setup, first, steady = time_first_batch(make_dl)
        print(f"  epoch{epoch}: POST={post:.3f}s  dl_setup={setup:.3f}s  "
              f"first_batch={first:.3f}s  TTFB(post+setup+first)={post+setup+first:.3f}s  "
              f"steady={steady:.1f}ms/batch")
        del ds


def run_imagefolder(nw):
    import torchvision
    print(f"\n===== ImageFolder num_workers={nw} =====")
    ds = torchvision.datasets.ImageFolder(str(IMAGEFOLDER), transform=TF)
    for epoch in range(2):
        def make_dl():
            return DataLoader(ds, batch_size=64, num_workers=nw, shuffle=True,
                              collate_fn=imagefolder_collate,
                              persistent_workers=False, pin_memory=False)
        setup, first, steady = time_first_batch(make_dl)
        print(f"  epoch{epoch}: dl_setup={setup:.3f}s  first_batch={first:.3f}s  "
              f"TTFB={setup+first:.3f}s  steady={steady:.1f}ms/batch")


if __name__ == "__main__":
    run_dfs(0)
    run_dfs(4)
    run_imagefolder(0)
    run_imagefolder(4)
