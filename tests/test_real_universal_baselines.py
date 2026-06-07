from __future__ import annotations

from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq

from benchmarks.datasetfs_bench.runner.real_universal import _run_baseline_training


def test_real_universal_text_parquet_baseline_runs(tmp_path):
    parquet_path = tmp_path / "text.parquet"
    table = pa.Table.from_pylist(
        [
            {"path": "a.txt", "text": "alpha alpha alpha", "label": "a"},
            {"path": "b.txt", "text": "zulu zulu zulu", "label": "b"},
            {"path": "c.txt", "text": "alpha more text", "label": "a"},
            {"path": "d.txt", "text": "zulu more text", "label": "b"},
        ]
    )
    pq.write_table(table, parquet_path)

    row = _run_baseline_training(
        {"name": "tiny_text", "task": "text_classification", "modality": "text", "batch_size": 2},
        {"format": "parquet", "path": str(parquet_path), "modality": "text"},
        SimpleNamespace(max_batches=2, batch_size=2),
    )

    assert row["status"] == "ok"
    assert row["format"] == "parquet"
    assert row["n_samples"] > 0


def test_real_universal_video_folder_baseline_runs(tmp_path):
    root = tmp_path / "videos"
    (root / "class_a").mkdir(parents=True)
    (root / "class_b").mkdir(parents=True)
    for i in range(2):
        (root / "class_a" / f"a{i}.avi").write_bytes(b"video-a" * 1024)
        (root / "class_b" / f"b{i}.avi").write_bytes(b"video-b" * 1024)

    row = _run_baseline_training(
        {"name": "tiny_video", "task": "video_action_classification", "modality": "video", "batch_size": 2},
        {"format": "folder", "path": str(root), "modality": "video"},
        SimpleNamespace(max_batches=2, batch_size=2),
    )

    assert row["status"] == "ok"
    assert row["format"] == "folder"
    assert row["n_samples"] > 0
