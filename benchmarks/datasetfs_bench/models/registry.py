"""Model factory. SimpleCNN for fast smoke tests; ResNet-18/50 for real benchmark runs.

ResNet variants use torchvision's implementation with the final FC layer
swapped for `num_classes`. Pretrained weights are NOT loaded — we want the
forward+backward path representative of from-scratch training, since that's
what most thesis benchmarks care about.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18, resnet50


class SimpleCNN(nn.Module):
    """Tiny CNN — used when we want training to be data-bound, not compute-bound,
    so the loader differences dominate runtime."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def _resnet(arch: str, num_classes: int) -> nn.Module:
    if arch == "resnet18":
        model = resnet18(weights=None)
    elif arch == "resnet50":
        model = resnet50(weights=None)
    else:
        raise ValueError(f"unknown resnet arch: {arch}")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "simplecnn":
        return SimpleCNN(num_classes=num_classes)
    if name in ("resnet18", "resnet50"):
        return _resnet(name, num_classes)
    raise ValueError(f"unknown model: {name}")
