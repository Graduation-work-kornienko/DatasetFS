"""Download Imagenette + Imagewoof from fastai.

Usage:
    python -m scripts.datasets.download                  # both datasets
    python -m scripts.datasets.download imagenette       # one dataset
    python -m scripts.datasets.download --data-root data
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.datasets._fastai import ALL_DATASETS, ensure_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="Dataset name(s) to download; default: all")
    parser.add_argument("--data-root", default="data", help="Root directory for data/ (default: data)")
    args = parser.parse_args()

    requested = set(args.names) if args.names else {ds.name for ds in ALL_DATASETS}
    unknown = requested - {ds.name for ds in ALL_DATASETS}
    if unknown:
        parser.error(f"unknown dataset(s): {unknown}")

    data_root = Path(args.data_root).resolve()
    for ds in ALL_DATASETS:
        if ds.name in requested:
            ensure_dataset(ds, data_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
