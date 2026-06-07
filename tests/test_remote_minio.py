"""End-to-end remote-storage integration test.

Spins up a real MinIO server in Docker, uploads a DatasetFS-format dataset
(manifest + shards) into an anonymous-download bucket, points the daemon at the
bucket over HTTP (`--root http://.../bucket --cache-dir ...`), and runs a few
training steps against it. This exercises the whole remote path: prefetch →
local cache → pipeline → shared memory → PyTorch.

Skips automatically if Docker (or the `minio` SDK) is unavailable. The dataset
is a tiny synthetic ImageFolder converted by the Go converter, so the test is
self-contained and fast.
"""
from __future__ import annotations

import functools
import json
import os
import shutil
import signal
import socket
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
import requests

from tests.conftest import _cleanup_tmp_files, _wait_for_healthz

REPO_ROOT = Path(__file__).resolve().parent.parent
DAEMON_URL = "http://localhost:51409"
MINIO_IMAGE = "minio/minio:latest"
MINIO_ACCESS = "minioadmin"
MINIO_SECRET = "minioadmin"


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


def _minio_sdk_available() -> bool:
    try:
        import minio  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not (_docker_available() and _minio_sdk_available()),
    reason="requires Docker and the `minio` Python SDK",
)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _collate(items, label_to_idx):
    """Module-level collate so it's picklable for DataLoader `spawn` workers
    (macOS/py3.13 default). Local closures/lambdas fail to pickle."""
    import torch
    x = torch.stack([it["image"] for it in items])
    y = torch.tensor([label_to_idx[it["label"]] for it in items], dtype=torch.long)
    return x, y


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def datasetfs_binary() -> Path:
    """Build the single datasetfs binary (daemon | vacuum | converter)."""
    binary = REPO_ROOT / "bin" / "datasetfs"
    binary.parent.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "CGO_ENABLED": "1",
        "PKG_CONFIG_PATH": (
            "/opt/homebrew/opt/jpeg-turbo/lib/pkgconfig"
            + (":" + os.environ["PKG_CONFIG_PATH"] if "PKG_CONFIG_PATH" in os.environ else "")
        ),
    }
    subprocess.run(["go", "build", "-o", str(binary), "./cmd/datasetfs"],
                   cwd=REPO_ROOT, env=env, check=True)
    return binary


@pytest.fixture(scope="module")
def converter_binary(datasetfs_binary: Path) -> Path:
    return datasetfs_binary


@pytest.fixture(scope="module")
def daemon_binary(datasetfs_binary: Path) -> Path:
    return datasetfs_binary


@pytest.fixture(scope="module")
def datasetfs_dir(tmp_path_factory, converter_binary: Path) -> Path:
    """Build a tiny ImageFolder and convert it to DatasetFS format."""
    from PIL import Image

    base = tmp_path_factory.mktemp("remote_ds")
    imgfolder = base / "imagefolder"
    classes = ["alpha", "beta", "gamma"]
    rng = np.random.default_rng(0)
    for ci, cls in enumerate(classes):
        d = imgfolder / cls
        d.mkdir(parents=True)
        for i in range(12):
            # Class-correlated tint so a model could in principle learn it.
            arr = (rng.integers(0, 60, (32, 32, 3)) + ci * 60).astype("uint8")
            Image.fromarray(arr).save(d / f"{cls}_{i}.jpg")

    out = base / "datasetfs"
    subprocess.run(
        [str(converter_binary), "converter", "dataset-folder", "--source", str(imgfolder), "--target", str(out)],
        cwd=REPO_ROOT, check=True,
    )
    # Converter writes parquet manifest + base shards (+ empty delta shard_-1).
    assert (out / "metadata.parquet").exists(), "converter must emit a parquet manifest"
    shards = sorted(p.name for p in out.glob("shard_*") if p.name != "shard_-1")
    assert shards, "converter produced no base shards"
    return out


@pytest.fixture(scope="module")
def minio_bucket(datasetfs_dir: Path):
    """Start MinIO, create an anonymous-download bucket, upload the dataset.

    Yields (endpoint_host_port, bucket_name). Tears the container down after.
    """
    from minio import Minio

    port = _free_port()
    container = f"datasetfs-minio-{uuid.uuid4().hex[:8]}"
    endpoint = f"127.0.0.1:{port}"

    subprocess.run([
        "docker", "run", "-d", "--name", container,
        "-p", f"{port}:9000",
        "-e", f"MINIO_ROOT_USER={MINIO_ACCESS}",
        "-e", f"MINIO_ROOT_PASSWORD={MINIO_SECRET}",
        MINIO_IMAGE, "server", "/data",
    ], check=True, capture_output=True)

    try:
        # Wait for the S3 health endpoint.
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                r = requests.get(f"http://{endpoint}/minio/health/live", timeout=1)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.5)
        else:
            logs = subprocess.run(["docker", "logs", container], capture_output=True, text=True)
            raise RuntimeError(f"MinIO did not become healthy:\n{logs.stdout}\n{logs.stderr}")

        client = Minio(endpoint, access_key=MINIO_ACCESS, secret_key=MINIO_SECRET, secure=False)
        bucket = "datasetfs"
        client.make_bucket(bucket)

        # Anonymous read-only download policy so the daemon's plain HTTP GETs work.
        policy = {
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{bucket}/*"],
            }],
        }
        client.set_bucket_policy(bucket, json.dumps(policy))

        # Upload the manifest + base shards (skip the empty delta shard; the
        # daemon recreates it locally on demand).
        for f in sorted(datasetfs_dir.iterdir()):
            if f.name == "shard_-1" or not f.is_file():
                continue
            client.fput_object(bucket, f.name, str(f))

        # Sanity: an anonymous GET of a shard must succeed.
        probe = requests.get(f"http://{endpoint}/{bucket}/metadata.parquet", timeout=5)
        assert probe.status_code == 200, f"anonymous GET failed: {probe.status_code}"

        yield endpoint, bucket
    finally:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True)


class _RemoteDaemon:
    """Launches the datasetfs daemon against a remote (HTTP) root with a cache dir."""

    def __init__(self, binary: Path, root_url: str, cache_dir: Path):
        self.binary = binary
        self.root_url = root_url
        self.cache_dir = cache_dir
        self.url = DAEMON_URL
        self._proc = None

    def start(self) -> None:
        _cleanup_tmp_files()
        log_dir = REPO_ROOT / "runs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = log_dir / f"daemon-remote-{int(time.time()*1000)}.log"
        self._log_file = open(self._log_path, "w")
        self._proc = subprocess.Popen(
            [str(self.binary), "daemon", "--no-mount",
             "--root", self.root_url,
             "--cache-dir", str(self.cache_dir)],
            cwd=REPO_ROOT, stdout=self._log_file, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
        _wait_for_healthz(self.url, timeout=60.0)

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if hasattr(self, "_log_file"):
            self._log_file.close()
        _cleanup_tmp_files()


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_remote_prefetch_and_training(daemon_binary, datasetfs_dir, minio_bucket, tmp_path):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import transforms

    from clients.python import DatasetFS

    endpoint, bucket = minio_bucket
    root_url = f"http://{endpoint}/{bucket}"
    cache_dir = tmp_path / "cache"

    daemon = _RemoteDaemon(daemon_binary, root_url, cache_dir)
    daemon.start()
    try:
        # 1. Streaming-overlap fetches only the manifest at daemon startup. Shards
        # arrive in the background / on demand while training drains the epoch.
        expected = sorted(p.name for p in datasetfs_dir.glob("shard_*") if p.name != "shard_-1")
        assert (cache_dir / "metadata.parquet").exists(), "manifest not prefetched"

        # 2. Build labels and a DatasetFS loader against the running daemon.
        labels = sorted(p.name for p in (datasetfs_dir.parent / "imagefolder").iterdir() if p.is_dir())
        label_to_idx = {name: i for i, name in enumerate(labels)}

        ds = DatasetFS(num_workers=2, seed=7, transform=transforms.ToTensor())
        loader = DataLoader(ds, batch_size=8, num_workers=2,
                            collate_fn=functools.partial(_collate, label_to_idx=label_to_idx))

        # 3. Run a real training step or two; assert data flows and loss is finite.
        model = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, len(labels)),
        )
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.CrossEntropyLoss()

        seen = 0
        first_loss = None
        for x, y in loader:
            assert x.ndim == 4 and x.shape[1] == 3, f"bad image batch shape {x.shape}"
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            assert torch.isfinite(loss), "loss is not finite"
            loss.backward()
            opt.step()
            seen += x.shape[0]
            if first_loss is None:
                first_loss = loss.item()

        assert seen > 0, "no samples were delivered from the remote dataset"
        assert first_loss is not None and np.isfinite(first_loss)
        cached_shards = sorted(p.name for p in cache_dir.glob("shard_*") if p.name != "shard_-1")
        assert cached_shards == expected, f"cache {cached_shards} != dataset {expected}"
        print(f"[remote-minio] trained on {seen} samples from {root_url}, first loss={first_loss:.3f}")
    finally:
        daemon.stop()
