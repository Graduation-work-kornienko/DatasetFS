from __future__ import annotations

import subprocess
from pathlib import Path

from benchmarks.datasetfs_bench.runner.daemon_ctl import DaemonManager
from benchmarks.datasetfs_bench.runner import mutation_bench
from scripts.datasets.datasetfs_writer import DatasetFSWriter


class _FakeProc:
    pid = 12345

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0


def test_read_only_benchmark_daemon_disables_wal(tmp_path: Path, monkeypatch):
    captured = {}

    def fake_popen(argv, **_kwargs):
        captured["argv"] = argv
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr("benchmarks.datasetfs_bench.runner.daemon_ctl.wait_for_healthz", lambda *_a, **_kw: None)

    manager = DaemonManager(
        binary=Path("bin/datasetfs"),
        root_path=tmp_path / "datasetfs",
        cwd=tmp_path,
        log_path=tmp_path / "daemon.log",
    )
    manager.start()

    assert "--no-wal" in captured["argv"]
    assert "--wal-format" not in captured["argv"]


def test_mutation_benchmark_daemon_uses_binary_wal(tmp_path: Path, monkeypatch):
    captured = {}

    def fake_popen(argv, **_kwargs):
        captured["argv"] = argv
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(mutation_bench, "_cleanup_tmp_files", lambda: None)
    monkeypatch.setattr(mutation_bench, "_force_unmount", lambda _mount: None)
    monkeypatch.setattr(mutation_bench, "_wait_for_healthz", lambda *_a, **_kw: None)
    monkeypatch.setattr(mutation_bench, "_wait_for_mount", lambda _mount: None)

    daemon = mutation_bench.MountedDaemon(
        binary=Path("bin/datasetfs"),
        root=tmp_path / "datasetfs",
        mount=tmp_path / "mnt",
        log_dir=tmp_path / "logs",
    )
    daemon.start()

    assert "--no-wal" not in captured["argv"]
    assert captured["argv"][captured["argv"].index("--wal-format") + 1] == "binary"


def test_vacuum_matrix_daemon_can_use_json_wal_with_auto_vacuum(tmp_path: Path, monkeypatch):
    captured = {}

    def fake_popen(argv, **_kwargs):
        captured["argv"] = argv
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(mutation_bench, "_cleanup_tmp_files", lambda: None)
    monkeypatch.setattr(mutation_bench, "_force_unmount", lambda _mount: None)
    monkeypatch.setattr(mutation_bench, "_wait_for_healthz", lambda *_a, **_kw: None)
    monkeypatch.setattr(mutation_bench, "_wait_for_mount", lambda _mount: None)

    daemon = mutation_bench.MountedDaemon(
        binary=Path("bin/datasetfs"),
        root=tmp_path / "datasetfs",
        mount=tmp_path / "mnt",
        log_dir=tmp_path / "logs",
        wal_format="json",
        auto_vacuum=True,
        vacuum_interval="1s",
        vacuum_threshold=0.05,
    )
    daemon.start()

    assert captured["argv"][captured["argv"].index("--wal-format") + 1] == "json"
    assert "--auto-vacuum" in captured["argv"]
    assert captured["argv"][captured["argv"].index("--vacuum-threshold") + 1] == "0.05"


def test_vacuum_matrix_daemon_can_disable_wal_and_vacuum(tmp_path: Path, monkeypatch):
    captured = {}

    def fake_popen(argv, **_kwargs):
        captured["argv"] = argv
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(mutation_bench, "_cleanup_tmp_files", lambda: None)
    monkeypatch.setattr(mutation_bench, "_force_unmount", lambda _mount: None)
    monkeypatch.setattr(mutation_bench, "_wait_for_healthz", lambda *_a, **_kw: None)
    monkeypatch.setattr(mutation_bench, "_wait_for_mount", lambda _mount: None)

    daemon = mutation_bench.MountedDaemon(
        binary=Path("bin/datasetfs"),
        root=tmp_path / "datasetfs",
        mount=tmp_path / "mnt",
        log_dir=tmp_path / "logs",
        no_wal=True,
    )
    daemon.start()

    assert "--no-wal" in captured["argv"]
    assert "--auto-vacuum" not in captured["argv"]
    assert "--wal-format" not in captured["argv"]


def test_no_wal_benchmark_daemon_does_not_create_wal_file(tmp_path: Path, daemon_binary):
    root = tmp_path / "datasetfs"
    with DatasetFSWriter(root) as writer:
        writer.add("a.txt", b"alpha", {"label": "a"})

    manager = DaemonManager(
        binary=daemon_binary,
        root_path=root,
        cwd=Path.cwd(),
        log_path=tmp_path / "daemon.log",
    )
    manager.start()
    manager.stop()

    assert not (root / "wal.log").exists()
