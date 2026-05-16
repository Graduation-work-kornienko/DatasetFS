"""Ground-truth utilities for correctness tests."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Iterable


def path_key(p) -> str:
    """Canonical 'class_name/filename' key for a sample, matching
    imagefolder_index/imagefolder_paths.

    We can't use just the basename: some datasets (Speech Commands V2) have
    the same filename hash across classes (same speaker, different words).
    """
    p = Path(p)
    return f"{p.parent.name}/{p.name}"


def imagefolder_index(root: Path) -> dict[str, str]:
    """Build {"class_name/filename": class_name} from an ImageFolder-style root.

    Keying by class/filename (not just filename) keeps lookups unambiguous
    even when datasets have the same basename across multiple classes.
    """
    index: dict[str, str] = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for img in class_dir.iterdir():
            if not img.is_file():
                continue
            key = f"{class_dir.name}/{img.name}"
            if key in index:
                raise ValueError(f"path collision {key}")
            index[key] = class_dir.name
    return index


def imagefolder_paths(root: Path) -> dict[str, Path]:
    """Build {"class_name/filename": absolute Path}."""
    paths: dict[str, Path] = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for img in class_dir.iterdir():
            if img.is_file():
                paths[f"{class_dir.name}/{img.name}"] = img
    return paths


def hash_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def hash_file(p: Path) -> str:
    return hash_bytes(p.read_bytes())


def hashes_for(paths: Iterable[Path]) -> dict[str, str]:
    return {p.name: hash_file(p) for p in paths}


def hash_tensor(t) -> str:
    """Stable hash of a torch.Tensor's raw bytes (after .contiguous().cpu())."""
    import torch  # local import; tests/helpers.py is also imported by non-torch tools
    tensor = t.detach().cpu().contiguous()
    return hash_bytes(tensor.numpy().tobytes())


def decode_with(transform: Callable, path: Path):
    """Open an image file and apply the given transform — same path the DatasetFS
    Python client uses internally. Returns a tensor."""
    from PIL import Image
    with Image.open(path) as im:
        return transform(im)
