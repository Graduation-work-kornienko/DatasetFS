"""Subprocess lifecycle for the FUSE daemon. Same shape as tests/conftest.py's
DaemonManager but here it lives with the benchmark code so it can evolve
independently (e.g., capture /metrics in Phase 3)."""
from __future__ import annotations

import glob
import os
import signal
import subprocess
import time
from pathlib import Path

import requests


def cleanup_tmp_files() -> None:
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


def wait_for_healthz(url: str, timeout: float = 30.0) -> None:
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
    """Owns one datasetfs daemon process. Single-instance per benchmark run.
    Re-init the dataset for each loader iteration via /initialize_loading;
    the daemon process itself stays up."""

    def __init__(
        self,
        binary: Path,
        root_path: Path | str,
        cwd: Path,
        log_path: Path | None = None,
        url: str = "http://localhost:51409",
        cache_dir: Path | str | None = None,
        prefetch_concurrency: int | None = None,
        remote_throttle: int | None = None,
    ):
        self.binary = binary
        self.root_path = root_path
        self.cwd = cwd
        self.url = url
        self.cache_dir = cache_dir
        self.prefetch_concurrency = prefetch_concurrency
        self.remote_throttle = remote_throttle
        self._proc: subprocess.Popen | None = None
        self._log_path = log_path
        self._log_file = None

    def start(self) -> None:
        cleanup_tmp_files()
        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(self._log_path, "w")
            stdout = self._log_file
        else:
            stdout = subprocess.DEVNULL
        print(
            f"[daemon] start {self.binary} daemon --no-mount --no-wal --root {self.root_path}",
            flush=True,
        )
        argv = [str(self.binary), "daemon", "--no-mount", "--no-wal", "--root", str(self.root_path)]
        if self.cache_dir is not None:
            argv += ["--cache-dir", str(self.cache_dir)]
        if self.prefetch_concurrency is not None:
            argv += ["--prefetch-concurrency", str(self.prefetch_concurrency)]
        if self.remote_throttle is not None and self.remote_throttle > 0:
            argv += ["--remote-throttle", str(self.remote_throttle)]
        self._proc = subprocess.Popen(
            argv,
            cwd=self.cwd,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
        wait_for_healthz(self.url, timeout=30.0)
        print("[daemon] ready", flush=True)

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
                if hasattr(os, "killpg"):
                    try:
                        os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    self._proc.kill()
                self._proc.wait(timeout=5)
        self._proc = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        cleanup_tmp_files()

    def restart(self) -> None:
        self.stop()
        self.start()

    @property
    def pid(self) -> int | None:
        """OS pid of the running daemon process, or None if not running.
        Used by SystemSampler to track daemon RSS alongside the Python tree."""
        return self._proc.pid if self._proc is not None and self._proc.poll() is None else None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
