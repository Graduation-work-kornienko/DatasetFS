"""Loader abstraction so the training loop is identical across formats.

Each concrete loader (DatasetFS, WebDataset, ImageFolder) wraps its native
iteration semantics into a uniform interface:
    - construct with a per-loader config dict
    - `setup()` does any expensive one-time work (start daemon, scan files)
    - `make_loader()` returns a torch DataLoader that yields (images, targets)
    - `teardown()` cleans up
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from torch.utils.data import DataLoader


class BaseLoader(ABC):
    """Wraps one storage format as a uniform PyTorch DataLoader factory.

    Subclasses set `name` (used in reports) and implement setup/make_loader/teardown.
    The loader is responsible for choosing its own DataLoader's collate function;
    the contract is only that batches yield `(images: Tensor[N,C,H,W], targets: Tensor[N])`.
    """

    name: ClassVar[str]

    def __init__(self, spec: dict[str, Any]):
        self.spec = spec
        self.batch_size: int = spec["batch_size"]
        self.num_workers: int = spec["num_workers"]
        self.image_size: int = spec.get("image_size", 224)
        self.label_to_idx: dict[str, int] = spec["label_to_idx"]
        self.seed: int | None = spec.get("seed")

    @abstractmethod
    def setup(self) -> None:
        """One-time setup before iteration (e.g., spawn daemon, build index)."""

    @abstractmethod
    def make_loader(self) -> DataLoader:
        """Return a fresh DataLoader. Called once per epoch.

        Each yielded batch must be `(images, targets)` tensors."""

    @abstractmethod
    def teardown(self) -> None:
        """Release any resources (daemon process, mmaps, file handles)."""

    def internal_metrics(self) -> dict[str, Any]:
        """Loader-specific counters (e.g., DatasetFS cache hits). Default empty.

        In Phase 3 we'll plug daemon `/metrics` data through this hook."""
        return {}
