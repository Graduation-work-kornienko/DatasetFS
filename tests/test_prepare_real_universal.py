from __future__ import annotations

import sys
import tarfile
import types
from pathlib import Path

from PIL import Image

from scripts.datasets import prepare_real_universal as pru
from scripts.datasets.datasetfs_writer import read_parquet_manifest


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
    manifest = read_parquet_manifest(out)
    assert (out / ".done").exists()
    assert (out / "metadata.parquet").exists()
    assert not (out / "metadata.jsonl").exists()
    assert manifest["files"]
    for meta in manifest["files"].values():
        assert meta["meta"]["label"] == "layout"
        assert "regions" in meta["meta"]
        assert meta["size"] > 0


def test_prepare_amazon_polarity_slice_writes_text_labels(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def fake_load_dataset(name, split, streaming):
        assert name == "fancyzhx/amazon_polarity"
        assert split == "train"
        assert streaming is True
        return [
            {"title": "great", "content": "works well", "label": 1},
            {"title": "bad", "content": "broke quickly", "label": 0},
        ]

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))

    pru.prepare_amazon_polarity_slice(target_gb=0.000001)

    out = tmp_path / "data/formats/real_universal/amazon_polarity_slice/datasetfs"
    manifest = read_parquet_manifest(out)
    labels = {meta["meta"]["label"] for meta in manifest["files"].values()}
    assert (out / ".done").exists()
    assert labels == {"0", "1"}
    assert all(meta["path"].endswith(".txt") for meta in manifest["files"].values())
    assert (tmp_path / "data/formats/real_universal/amazon_polarity_slice/parquet/data.parquet").exists()


def test_prepare_wikipedia_slice_writes_parquet_baseline(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def fake_load_dataset(name, config, split, streaming):
        assert name == "wikimedia/wikipedia"
        assert config == "20231101.en"
        assert split == "train"
        assert streaming is True
        return [
            {"text": "alpha encyclopedic article"},
            {"text": "zulu encyclopedic article"},
        ]

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))

    pru.prepare_wikipedia_en_slice(target_gb=0.000001)

    out = tmp_path / "data/formats/real_universal/wikipedia_en_slice/datasetfs"
    parquet = tmp_path / "data/formats/real_universal/wikipedia_en_slice/parquet/data.parquet"
    manifest = read_parquet_manifest(out)
    assert manifest["files"]
    assert parquet.exists()


def test_prepare_ucf101_subset_writes_video_labels(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    archive = tmp_path / "fixture.tar.gz"
    src = tmp_path / "src"
    (src / "ApplyEyeMakeup").mkdir(parents=True)
    (src / "BasketballDunk").mkdir(parents=True)
    (src / "ApplyEyeMakeup" / "a.avi").write_bytes(b"video-a" * 128)
    (src / "BasketballDunk" / "b.avi").write_bytes(b"video-b" * 128)
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(src / "ApplyEyeMakeup", arcname="UCF101_subset/train/ApplyEyeMakeup")
        tf.add(src / "BasketballDunk", arcname="UCF101_subset/train/BasketballDunk")

    def fake_download(_url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(archive.read_bytes())

    monkeypatch.setattr(pru, "_download_file", fake_download)
    pru.prepare_ucf101_subset(target_gb=0.000001)

    out = tmp_path / "data/formats/real_universal/ucf101_subset/datasetfs"
    manifest = read_parquet_manifest(out)
    labels = {meta["meta"]["label"] for meta in manifest["files"].values()}
    assert (out / ".done").exists()
    assert labels == {"ApplyEyeMakeup", "BasketballDunk"}
    assert all(meta["path"].endswith(".avi") for meta in manifest["files"].values())
