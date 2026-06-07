from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from scripts.datasets import prepare_real_universal as pru


def _write_jpeg(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path, format="JPEG")


def test_prepare_publaynet_slice_from_existing_raw(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "data/raw/publaynet/train/layout"
    _write_jpeg(raw / "doc_a.jpg", (200, 30, 30))
    _write_jpeg(raw / "doc_b.jpg", (30, 30, 200))

    pru.prepare_publaynet_slice(target_gb=0.000001)

    out = tmp_path / "data/formats/real_universal/publaynet_slice/datasetfs"
    manifest = json.loads((out / "metadata.jsonl").read_text(encoding="utf-8"))
    assert (out / ".done").exists()
    assert manifest["files"]
    for meta in manifest["files"].values():
        assert meta["meta"]["label"] == "layout"
        assert "regions" in meta["meta"]
        assert meta["size"] > 0
