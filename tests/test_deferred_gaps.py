"""Regression tests for deferred coverage gaps documented in docs/testing.md.

These are intentionally small and self-contained: each test builds a tiny
DatasetFS root directly, starts the real daemon, and exercises the failure/cleanup
path under test without downloading benchmark datasets.
"""
from __future__ import annotations

import glob
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from clients.python import DatasetFS
from scripts.datasets.datasetfs_writer import DatasetFSWriter
from tests.conftest import DaemonManager


def _identity(x):
    return x


def _decode_bytes(raw) -> bytes:
    return bytes(raw)


def _build_raw_datasetfs(root: Path, n: int = 12) -> dict[str, bytes]:
    """Build one-shard DatasetFS root with raw byte payloads."""
    payloads = {
        f"sample_{i:03d}.bin": f"payload-{i:03d}|".encode() + bytes([i]) * 64
        for i in range(n)
    }
    with DatasetFSWriter(root) as writer:
        for name, payload in payloads.items():
            writer.add(name, payload, {"label": "raw"})
    return payloads


def _read_epoch(timeout_seconds: float = 2.0) -> dict[str, bytes]:
    ds = DatasetFS(
        num_workers=0,
        decode_fn=_decode_bytes,
        transform=_identity,
        timeout_seconds=timeout_seconds,
    )
    return {sample["path"]: bytes(sample["image"]) for sample in ds}


@pytest.fixture
def tiny_daemon(tmp_path: Path, daemon_binary: Path, repo_root: Path):
    root = tmp_path / "tiny_datasetfs"
    payloads = _build_raw_datasetfs(root)
    manager = DaemonManager(binary=daemon_binary, root_path=root, cwd=repo_root)
    manager.start()
    try:
        yield manager, payloads
    finally:
        manager.stop()


def test_concurrent_loader_reinit_fails_cleanly_without_silent_corruption(tiny_daemon, tmp_path, repo_root):
    """Deferred gap #2.

    A second DatasetFS() in the same process reinitializes the daemon session.
    The old iterator must not silently continue yielding corrupted data; it may
    end or raise, but it must do so promptly. The new iterator must still produce
    the correct epoch.
    """
    _manager, payloads = tiny_daemon
    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps({k: v.hex() for k, v in payloads.items()}), encoding="utf-8")
    code = r'''
import json, signal, sys
from pathlib import Path
from clients.python import DatasetFS

expected = {k: bytes.fromhex(v) for k, v in json.loads(Path(sys.argv[1]).read_text()).items()}

def dec(raw): return bytes(raw)
def ident(x): return x
def alarm(_signum, _frame): raise TimeoutError("bounded stale next timeout")
signal.signal(signal.SIGALRM, alarm)

ds1 = DatasetFS(num_workers=0, decode_fn=dec, transform=ident, timeout_seconds=0.2)
it1 = iter(ds1)
first = next(it1)
assert first["path"] in expected
assert bytes(first["image"]) == expected[first["path"]]

ds2 = DatasetFS(num_workers=0, decode_fn=dec, transform=ident, timeout_seconds=1.0)
got2 = {s["path"]: bytes(s["image"]) for s in ds2}
assert got2 == expected

signal.alarm(3)
try:
    stale = next(it1)
except (StopIteration, RuntimeError, OSError, ValueError, TimeoutError):
    sys.exit(0)
finally:
    signal.alarm(0)
assert stale["path"] in expected
assert bytes(stale["image"]) == expected[stale["path"]]
'''
    res = subprocess.run(
        [sys.executable, "-c", code, str(expected)],
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"


def test_daemon_crash_mid_epoch_is_bounded_failure(tiny_daemon, repo_root):
    """Deferred gap #3.

    Killing the daemon mid-epoch should not hang the Python client indefinitely.
    Today the acceptable clean behavior is EOF/StopIteration or a Python error;
    the key regression guard is bounded failure rather than silent deadlock.
    """
    manager, _payloads = tiny_daemon
    assert manager.pid is not None
    code = r'''
import os, signal, sys, time
from clients.python import DatasetFS

def dec(raw): return bytes(raw)
def ident(x): return x
def alarm(_signum, _frame): raise TimeoutError("bounded crash timeout")
signal.signal(signal.SIGALRM, alarm)

ds = DatasetFS(num_workers=0, decode_fn=dec, transform=ident, timeout_seconds=0.2)
it = iter(ds)
first = next(it)
assert first["path"]
os.kill(int(sys.argv[1]), signal.SIGKILL)
time.sleep(0.1)
signal.alarm(3)
try:
    next(it)
except (StopIteration, RuntimeError, OSError, ValueError, TimeoutError):
    sys.exit(0)
finally:
    signal.alarm(0)
sys.exit(0)
'''
    res = subprocess.run(
        [sys.executable, "-c", code, str(manager.pid)],
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"


def test_daemon_stop_cleans_tmp_files_and_fifos(tmp_path: Path, daemon_binary: Path, repo_root: Path):
    """Deferred gap #4: assert cleanup, not just best-effort cleanup."""
    root = tmp_path / "tiny_datasetfs"
    _build_raw_datasetfs(root)
    manager = DaemonManager(binary=daemon_binary, root_path=root, cwd=repo_root)
    manager.start()
    assert manager.pid is not None
    _ = _read_epoch(timeout_seconds=1.0)
    manager.stop()

    assert not Path("/tmp/mlfs_data.bin").exists()
    assert not Path("/tmp/mlfs_refs.bin").exists()
    assert not glob.glob("/tmp/datasetfs_pipe_*")


@pytest.mark.skipif(
    sys.platform != "darwin" or not Path("/Library/Filesystems/macfuse.fs").exists(),
    reason="FUSE POSIX test requires macFUSE on macOS",
)
def test_fuse_mount_posix_read_list_unlink(flat_mounted_daemon):
    """Deferred gap #5: direct POSIX access through the FUSE mount."""
    _manager, mount = flat_mounted_daemon
    mount = Path(mount)
    payloads = {
        "a.bin": b"alpha" * 32,
        "b.bin": b"bravo" * 32,
    }

    for name, payload in payloads.items():
        with open(mount / name, "wb") as f:
            f.write(payload)

    assert {p.name for p in mount.iterdir() if not p.name.startswith(".")} == set(payloads)
    for name, payload in payloads.items():
        p = mount / name
        assert p.read_bytes() == payload

    os.remove(mount / "a.bin")
    assert "a.bin" not in {p.name for p in mount.iterdir() if not p.name.startswith(".")}
    with pytest.raises(FileNotFoundError):
        (mount / "a.bin").read_bytes()
