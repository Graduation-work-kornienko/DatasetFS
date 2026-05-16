"""Pytest fixtures for Phase 1 correctness tests.

Key fixtures:
  - `repo_root`, `data_root` (session): paths
  - `daemon_binary`, `converter_binary` (session): built Go binaries
  - `imagenette_prepared` (session): downloaded + converted Imagenette in all formats
  - `daemon` (function): a DaemonManager — call `.url` for the URL, `.restart()` between
                         epochs in tests that iterate multiple times (avoids cross-session
                         FIFO interleave). Always torn down cleanly.
"""
from __future__ import annotations

import glob
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests


# Repo root must be on sys.path BEFORE pytest imports test modules
# (so they can `from clients.python import DatasetFS` and `from tests.helpers import ...`).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return _REPO_ROOT


@pytest.fixture(scope="session")
def data_root(repo_root: Path) -> Path:
    return repo_root / "data"


@pytest.fixture(scope="session")
def daemon_binary(repo_root: Path) -> Path:
    binary = repo_root / "bin" / "fuse_daemon"
    binary.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[fixture] building daemon → {binary}", flush=True)
    subprocess.run(
        ["go", "build", "-o", str(binary), "./cmd/fuse_daemon"],
        cwd=repo_root,
        check=True,
    )
    return binary


@pytest.fixture(scope="session")
def converter_binary(repo_root: Path) -> Path:
    binary = repo_root / "bin" / "dataset_converter"
    binary.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[fixture] building converter → {binary}", flush=True)
    subprocess.run(
        ["go", "build", "-o", str(binary), "./cmd/dataset_converter"],
        cwd=repo_root,
        check=True,
    )
    return binary


def _prepare_dataset(ds_def, repo_root: Path, data_root: Path) -> dict[str, Path]:
    """Idempotent download + format prep for one dataset.

    Imagefolder + WebDataset + DatasetFS are built. HuggingFace is skipped —
    Phase 1 tests don't exercise it and it adds dependency on `datasets`.
    """
    from scripts.datasets._fastai import ensure_dataset
    from scripts.datasets.prepare_formats import (
        prepare_imagefolder,
        prepare_webdataset,
        prepare_datasetfs,
    )

    extracted = ensure_dataset(ds_def, data_root)
    formats_root = data_root / "formats" / ds_def.name

    paths = {
        "imagefolder": formats_root / "imagefolder",
        "webdataset": formats_root / "webdataset",
        "datasetfs": formats_root / "datasetfs",
    }

    prepare_imagefolder(ds_def, extracted, paths["imagefolder"])
    prepare_webdataset(ds_def, extracted, paths["webdataset"])
    # prepare_datasetfs takes the per-class root (filtered imagefolder), not
    # raw extracted — so the Go converter sees ONLY classes from ds_def.classes.
    prepare_datasetfs(ds_def, paths["imagefolder"], paths["datasetfs"], repo_root)

    return paths


@pytest.fixture(scope="session")
def imagenette_prepared(repo_root: Path, data_root: Path, converter_binary: Path) -> dict[str, Path]:
    from scripts.datasets._fastai import IMAGENETTE2
    return _prepare_dataset(IMAGENETTE2, repo_root, data_root)


@pytest.fixture(scope="session")
def imagewoof_prepared(repo_root: Path, data_root: Path, converter_binary: Path) -> dict[str, Path]:
    from scripts.datasets._fastai import IMAGEWOOF2
    return _prepare_dataset(IMAGEWOOF2, repo_root, data_root)


@pytest.fixture(scope="session")
def speech_commands_prepared(repo_root: Path, data_root: Path, converter_binary: Path) -> dict[str, Path]:
    from scripts.datasets._fastai import SPEECH_COMMANDS_V2
    return _prepare_dataset(SPEECH_COMMANDS_V2, repo_root, data_root)


def _cleanup_tmp_files() -> None:
    """Remove /tmp/ artifacts left by a previous daemon session."""
    for path in ["/tmp/mlfs_data.bin", "/tmp/mlfs_refs.bin"]:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    for fifo in glob.glob("/tmp/datasetfs_pipe_*"):
        try:
            os.remove(fifo)
        except FileNotFoundError:
            pass


def _wait_for_healthz(url: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/healthz", timeout=1)
            if r.status_code == 200:
                return
        except Exception as e:
            last_err = e
        time.sleep(0.1)
    raise RuntimeError(f"daemon /healthz did not respond within {timeout}s: {last_err}")


class DaemonManager:
    """Owns a fuse_daemon subprocess. Supports restart-in-place for tests that
    iterate multiple times, where leaving the previous session's dealers
    blocked on the same FIFO would risk cross-session interleave."""

    def __init__(self, binary: Path, root_path: Path, cwd: Path, url: str = "http://localhost:51409"):
        self.binary = binary
        self.root_path = root_path
        self.cwd = cwd
        self.url = url
        self._proc: subprocess.Popen | None = None

    def start(self) -> None:
        _cleanup_tmp_files()
        print(
            f"\n[daemon] start: {self.binary} --no-mount --root {self.root_path}",
            flush=True,
        )
        log_dir = self.cwd / "runs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = log_dir / f"daemon-{int(time.time()*1000)}.log"
        self._log_file = open(self._log_path, "w")
        self._proc = subprocess.Popen(
            [str(self.binary), "--no-mount", "--root", str(self.root_path)],
            cwd=self.cwd,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
        _wait_for_healthz(self.url, timeout=30.0)
        print(f"[daemon] ready (log: {self._log_path})", flush=True)

    def stop(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
                else:
                    self._proc.terminate()
            except ProcessLookupError:
                pass
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("[daemon] SIGTERM timed out, sending SIGKILL", flush=True)
                if hasattr(os, "killpg"):
                    try:
                        os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    self._proc.kill()
                self._proc.wait(timeout=5)
        self._proc = None
        if hasattr(self, "_log_file"):
            self._log_file.close()
        _cleanup_tmp_files()

    def restart(self) -> None:
        self.stop()
        self.start()

    @property
    def pid(self) -> int | None:
        """OS pid of the running daemon, or None if not running. Lets long
        tests sample its RSS via psutil."""
        return self._proc.pid if self._proc is not None and self._proc.poll() is None else None


@pytest.fixture
def daemon(daemon_binary: Path, imagenette_prepared: dict[str, Path], repo_root: Path):
    manager = DaemonManager(
        binary=daemon_binary,
        root_path=imagenette_prepared["datasetfs"],
        cwd=repo_root,
    )
    manager.start()
    try:
        yield manager
    finally:
        manager.stop()


@pytest.fixture
def daemon_imagewoof(daemon_binary: Path, imagewoof_prepared: dict[str, Path], repo_root: Path):
    """Daemon pointed at Imagewoof's DatasetFS shards. Used for cross-dataset
    smoke tests so we don't only validate against Imagenette."""
    manager = DaemonManager(
        binary=daemon_binary,
        root_path=imagewoof_prepared["datasetfs"],
        cwd=repo_root,
    )
    manager.start()
    try:
        yield manager
    finally:
        manager.stop()


@pytest.fixture
def daemon_speech_commands(daemon_binary: Path, speech_commands_prepared: dict[str, Path], repo_root: Path):
    """Daemon pointed at Speech Commands' DatasetFS shards. Used for audio
    correctness + training tests (different data modality than Imagenette)."""
    manager = DaemonManager(
        binary=daemon_binary,
        root_path=speech_commands_prepared["datasetfs"],
        cwd=repo_root,
    )
    manager.start()
    try:
        yield manager
    finally:
        manager.stop()
