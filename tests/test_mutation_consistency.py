"""Feature F1 — snapshot-consistent concurrent mutation, end-to-end over FUSE.

A training run pins a manifest generation at epoch start; mutations (rm/cp over the
mount) publish a NEW generation; the running epoch keeps its pinned view until it
re-pins next epoch. This test exercises the *real* path: the daemon is FUSE-mounted
and files are added/removed by writing/removing bare-name files on the mountpoint,
which drives MutationManager.AddDeltaFile / DeleteFile.

Headline assertions:
  - the multiset of samples an epoch sees == exactly the files of its pinned
    generation (a delete mid-epoch is still fully served; an add mid-epoch is not);
  - every batch of one epoch carries the SAME generation (no torn read);
  - the next epoch reflects the committed mutations, at a higher generation.

Skips when FUSE can't be mounted (e.g. macFUSE absent), since the path under test
is the mount itself.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Fast guard before the (slower) mount attempt in the fixture.
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or not Path("/Library/Filesystems/macfuse.fs").exists(),
    reason="FUSE mutation-consistency test requires macFUSE",
)

from clients.python import DatasetFS  # noqa: E402


def _identity(x):
    """decode_fn / transform passthrough: yield the raw slot bytes unchanged so
    the test is independent of any image codec and can verify exact content."""
    return x


def _content(i: int) -> bytes:
    """Distinct, verifiable per-file payload (also catches offset/size bugs in the
    delta-serving path)."""
    return f"DATASETFS-FILE-{i:03d}|".encode() + bytes([i % 256]) * 64


def _write_on_mount(mount: Path, name: str, data: bytes) -> None:
    # open(O_CREAT|O_WRONLY|O_TRUNC) → vfs Create → Write → (close) Release →
    # MutationManager.AddDeltaFile(name, tmp). No chmod/utimes (would hit
    # unimplemented Setattr), so write the bytes directly rather than shutil.copy.
    with open(mount / name, "wb") as f:
        f.write(data)


def _read_epoch(**kwargs):
    """Drain one full epoch. Returns (paths_in_order, content_by_path, generations_set)."""
    ds = DatasetFS(
        num_workers=0,
        decode_fn=_identity,
        transform=_identity,
        timeout_seconds=20.0,
        **kwargs,
    )
    paths, contents, gens = [], {}, set()
    for s in ds:
        p = s["path"]
        paths.append(p)
        contents[p] = bytes(s["image"])
        if "dfs_generation" in s:
            gens.add(s["dfs_generation"])
    return paths, contents, gens


def test_snapshot_consistent_mutation_over_fuse(flat_mounted_daemon):
    manager, mount = flat_mounted_daemon
    mount = Path(mount)

    # ---- Populate an initial dataset entirely via FUSE writes (delta path) ----
    N = 8
    names = [f"f{i:03d}" for i in range(N)]
    payloads = {names[i]: _content(i) for i in range(N)}
    for i, name in enumerate(names):
        _write_on_mount(mount, name, payloads[name])

    P0 = set(names)
    # Sanity: the mount lists exactly what we wrote (Readdir reflects the index).
    listed = {n for n in os.listdir(mount) if not n.startswith(".")}
    assert listed == P0, f"mount listing {sorted(listed)} != written {sorted(P0)}"

    # ---- Epoch A: a clean read must serve every added (delta) file ----
    paths_a, contents_a, gens_a = _read_epoch()
    assert set(paths_a) == P0, f"epoch A served {sorted(set(paths_a))}, want {sorted(P0)}"
    assert len(paths_a) == N, f"epoch A had duplicates: {paths_a}"
    assert len(gens_a) == 1, f"generation not constant within epoch A: {gens_a}"
    for name in P0:
        assert contents_a[name] == payloads[name], f"served bytes for {name} are wrong"
    gen_a = next(iter(gens_a))

    # ---- Epoch B: mutate *during* the epoch; the pinned view must not tear ----
    deleted = "f000"
    added = "fNEW"
    payloads[added] = _content(999)

    ds = DatasetFS(num_workers=0, decode_fn=_identity, transform=_identity, timeout_seconds=20.0)
    it = iter(ds)
    collected = [next(it)]  # consume one sample → slot is loaded, snapshot is live

    # Mutations land strictly after this epoch pinned its generation.
    os.remove(mount / deleted)                     # vfs Unlink → DeleteFile
    _write_on_mount(mount, added, payloads[added])  # vfs Create/Release → AddDeltaFile

    collected.extend(it)  # drain the rest of the SAME epoch

    paths_b = [s["path"] for s in collected]
    gens_b = {s["dfs_generation"] for s in collected if "dfs_generation" in s}
    contents_b = {s["path"]: bytes(s["image"]) for s in collected}

    assert len(gens_b) == 1, f"generation not constant within epoch B (torn read!): {gens_b}"
    assert next(iter(gens_b)) == gen_a, "epoch B pinned a different gen than A despite no prior mutation"
    assert set(paths_b) == P0, (
        f"epoch B must serve its pinned generation {sorted(P0)} — deleted file still "
        f"present, added file absent — got {sorted(set(paths_b))}"
    )
    # The file deleted mid-epoch was STILL served, with correct bytes.
    assert deleted in contents_b and contents_b[deleted] == payloads[deleted]
    assert added not in contents_b, "file added mid-epoch leaked into the pinned epoch"

    # ---- Epoch C: a fresh read reflects the committed mutations at a higher gen ----
    paths_c, contents_c, gens_c = _read_epoch()
    expected_c = (P0 - {deleted}) | {added}
    assert set(paths_c) == expected_c, (
        f"epoch C served {sorted(set(paths_c))}, want {sorted(expected_c)}"
    )
    assert len(gens_c) == 1, f"generation not constant within epoch C: {gens_c}"
    assert next(iter(gens_c)) > gen_a, "mutations during/after epoch B must advance the generation"
    assert contents_c[added] == payloads[added], "newly added file served with wrong bytes"

    # ---- Hygiene: no upload temp files leaked by the FUSE write path ----
    leaked = [p for p in os.listdir("/tmp") if p.startswith("mlfs_upload_")]
    assert not leaked, f"FUSE write path leaked temp files: {leaked}"

    # Daemon survived all of it.
    assert manager.pid is not None, "daemon died during the test"
