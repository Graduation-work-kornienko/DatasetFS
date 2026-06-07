"""Yandex.Disk staging/archive helper for DatasetFS benchmarks.

The >RAM format-matrix work (PubLayNet ~40 GB across imagefolder/webdataset/
datasetfs) only *just* fits in the local free disk. To stay under budget we
offload datasets/formats not currently under test to Yandex.Disk (consumer
1 TB) and pull them back when needed.

IMPORTANT: Yandex.Disk is a STAGING/ARCHIVE store only. Benchmarks always read
from the local SSD — never over the network — otherwise we'd measure the
network instead of the storage format/page-cache behaviour. The remote-streaming
benchmark (Phase 5) uses a local MinIO, not this helper.

Auth: set the ``YADISK_TOKEN`` environment variable to a Yandex.Disk OAuth
token. A token can be minted at https://yandex.ru/dev/disk/poligon/ (the
"Polygon" gives a ready token for your own account) or via an OAuth app with
the ``cloud_api:disk.read`` / ``cloud_api:disk.write`` scopes.

CLI::

    python -m scripts.storage.ydisk push <local_path> <remote_path>
    python -m scripts.storage.ydisk pull <remote_path> <local_path>
    python -m scripts.storage.ydisk rm   <remote_path>
    python -m scripts.storage.ydisk ls   <remote_path>

Convention: archive under a ``datasetfs-bench/`` root on the Disk that mirrors
the local layout, e.g.::

    datasetfs-bench/formats/publaynet/datasetfs
    datasetfs-bench/raw/publaynet

Both files and directories are supported; directory transfers are recursive and
idempotent (a remote file is skipped when it already exists with the same size).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path, PurePosixPath

import requests

try:
    import yadisk
except ImportError:  # pragma: no cover - dependency hint
    yadisk = None


ENV_TOKEN = "YADISK_TOKEN"
PROGRESS_INTERVAL_SECONDS = 1.0
TRANSFER_CHUNK_BYTES = 8 * 1024 * 1024


def _fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"[ydisk] ERROR: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(2)


def _load_dotenv_token() -> None:
    """Best-effort: if YADISK_TOKEN isn't in the env, read it from a repo-root
    `.env` (KEY=value lines). Keeps `make` targets and ad-hoc runs from needing
    `set -a; . ./.env` every time. `.env` is gitignored."""
    if os.environ.get(ENV_TOKEN):
        return
    # repo root = three levels up from scripts/storage/ydisk.py
    dotenv = Path(__file__).resolve().parents[2] / ".env"
    if not dotenv.is_file():
        return
    for line in dotenv.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key == ENV_TOKEN and key not in os.environ:
            os.environ[key] = val.strip().strip('"').strip("'")


def get_client():
    """Return an authenticated yadisk client or fail fast with a clear message."""
    if yadisk is None:
        _fail(
            "the 'yadisk' package is not installed. "
            "Install it with: pip install yadisk  (it is in tests/requirements.txt)."
        )
    _load_dotenv_token()
    token = os.environ.get(ENV_TOKEN, "").strip()
    if not token:
        _fail(
            f"environment variable {ENV_TOKEN} is not set. "
            "Mint an OAuth token at https://yandex.ru/dev/disk/poligon/ and run: "
            f"export {ENV_TOKEN}=<token>"
        )
    # yadisk 3.x exposes Client; 2.x exposed YaDisk. Support both.
    factory = getattr(yadisk, "Client", None) or getattr(yadisk, "YaDisk", None)
    if factory is None:  # pragma: no cover - unexpected SDK layout
        _fail("unrecognised yadisk SDK layout (no Client/YaDisk class).")
    client = factory(token=token)
    try:
        ok = client.check_token()
    except Exception as exc:  # pragma: no cover - network/credential failure
        _fail(f"could not validate token against Yandex.Disk: {exc}")
    if not ok:
        _fail(f"{ENV_TOKEN} was rejected by Yandex.Disk (invalid/expired token).")
    return client


def _norm_remote(remote: str) -> str:
    """Normalise a remote path to the 'disk:/...'-friendly absolute form."""
    remote = remote.strip()
    if remote.startswith("disk:"):
        remote = remote[len("disk:"):]
    if not remote.startswith("/"):
        remote = "/" + remote
    return str(PurePosixPath(remote))


def _ensure_remote_dirs(client, remote_dir: str) -> None:
    """mkdir -p for a remote directory path (Yandex.Disk has no recursive mkdir)."""
    parts = PurePosixPath(_norm_remote(remote_dir)).parts  # ('/', 'a', 'b', ...)
    cur = ""
    for part in parts:
        if part == "/":
            continue
        cur = f"{cur}/{part}"
        if not client.exists(cur):
            try:
                client.mkdir(cur)
            except Exception as exc:  # noqa: BLE001 - tolerate races/already-exists
                if not client.exists(cur):
                    _fail(f"could not create remote dir {cur}: {exc}")


def _remote_file_size(client, remote: str) -> int | None:
    try:
        meta = client.get_meta(remote)
    except Exception:  # noqa: BLE001
        return None
    size = getattr(meta, "size", None)
    return int(size) if size is not None else None


def _fmt_bytes(n: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    v = float(n)
    for unit in units:
        if abs(v) < 1024 or unit == units[-1]:
            return f"{v:.1f} {unit}"
        v /= 1024
    return f"{v:.1f} TiB"


class _ProgressReader:
    def __init__(self, path: Path, label: str):
        self.path = path
        self.label = label
        self.total = path.stat().st_size
        self.sent = 0
        self.started = time.monotonic()
        self.last_print = self.started
        self.f = path.open("rb")

    def __len__(self):
        return self.total

    def read(self, n: int = -1) -> bytes:
        data = self.f.read(n)
        if data:
            self.sent += len(data)
            self._maybe_print(final=False)
        return data

    def close(self) -> None:
        self._maybe_print(final=True)
        self.f.close()

    def _maybe_print(self, final: bool) -> None:
        now = time.monotonic()
        if not final and now - self.last_print < PROGRESS_INTERVAL_SECONDS:
            return
        elapsed = max(now - self.started, 1e-9)
        speed = self.sent / elapsed
        pct = (100 * self.sent / self.total) if self.total else 100.0
        print(
            f"{self.label} {pct:5.1f}% "
            f"({_fmt_bytes(self.sent)}/{_fmt_bytes(self.total)}) "
            f"avg={_fmt_bytes(speed)}/s",
            flush=True,
        )
        self.last_print = now


def _progress_download(url: str, local_file: Path, label: str, total: int | None) -> None:
    started = time.monotonic()
    last_print = started
    done = 0
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        if total is None:
            total = int(r.headers.get("content-length", 0)) or None
        with local_file.open("wb") as f:
            for chunk in r.iter_content(chunk_size=TRANSFER_CHUNK_BYTES):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                now = time.monotonic()
                if now - last_print >= PROGRESS_INTERVAL_SECONDS:
                    elapsed = max(now - started, 1e-9)
                    speed = done / elapsed
                    if total:
                        pct = 100 * done / total
                        suffix = f"{pct:5.1f}% ({_fmt_bytes(done)}/{_fmt_bytes(total)})"
                    else:
                        suffix = _fmt_bytes(done)
                    print(f"{label} {suffix} avg={_fmt_bytes(speed)}/s", flush=True)
                    last_print = now
    elapsed = max(time.monotonic() - started, 1e-9)
    print(f"{label} done {_fmt_bytes(done)} avg={_fmt_bytes(done / elapsed)}/s", flush=True)


def _upload_temp_path(remote: str) -> str:
    """Return a sibling path without a throttled extension.

    Yandex.Disk REST upload links are throttled for media_type values inferred
    from the destination extension (`compressed`, `data`, `video`). Upload under
    an extensionless temporary name, then move it to the requested final name.
    See https://yadisk.readthedocs.io/en/stable/known_issues.html.
    """
    p = PurePosixPath(_norm_remote(remote))
    safe_name = p.name.replace(".", "_") + "__uploading"
    return str(p.with_name(safe_name))


def _move_remote(client, src: str, dst: str) -> None:
    if client.exists(dst):
        client.remove(dst, permanently=True)
    try:
        client.move(src, dst, overwrite=True)
    except TypeError:  # older yadisk versions may not accept overwrite
        client.move(src, dst)


def push(client, local: str, remote: str) -> None:
    """Upload a local file or directory tree to Yandex.Disk (idempotent)."""
    local_path = Path(local)
    if not local_path.exists():
        _fail(f"local path does not exist: {local_path}")
    remote = _norm_remote(remote)

    if local_path.is_file():
        _ensure_remote_dirs(client, str(PurePosixPath(remote).parent))
        _push_file(client, local_path, remote)
        return

    # Directory: walk and mirror under `remote`.
    files = sorted(p for p in local_path.rglob("*") if p.is_file())
    _ensure_remote_dirs(client, remote)
    complete = f"{remote}/.complete"
    if client.exists(complete):
        client.remove(complete, permanently=True)
    total = len(files)
    for i, f in enumerate(files, 1):
        rel = f.relative_to(local_path).as_posix()
        dst = f"{remote}/{rel}"
        _ensure_remote_dirs(client, str(PurePosixPath(dst).parent))
        _push_file(client, f, dst, prefix=f"[{i}/{total}] ")
    marker = local_path / ".ydisk_complete.json"
    marker.write_text(
        json.dumps({"files": total, "bytes": sum(f.stat().st_size for f in files)}, sort_keys=True),
        encoding="utf-8",
    )
    try:
        _push_file(client, marker, complete, prefix=f"[{total + 1}/{total + 1}] ")
    finally:
        marker.unlink(missing_ok=True)


def _push_file(client, local_file: Path, remote: str, prefix: str = "") -> None:
    local_size = local_file.stat().st_size
    if client.exists(remote) and _remote_file_size(client, remote) == local_size:
        print(f"{prefix}[skip] {remote} (same size {local_size})", flush=True)
        return
    tmp_remote = _upload_temp_path(remote)
    if client.exists(tmp_remote):
        client.remove(tmp_remote, permanently=True)
    print(
        f"{prefix}[push] {local_file} -> {remote} ({_fmt_bytes(local_size)}; "
        f"temp={tmp_remote})",
        flush=True,
    )
    link = client.get_upload_link(tmp_remote, overwrite=True)
    reader = _ProgressReader(local_file, f"{prefix}[upload]")
    try:
        resp = requests.put(link, data=reader, timeout=120)
        resp.raise_for_status()
    finally:
        reader.close()
    _move_remote(client, tmp_remote, remote)


def pull(client, remote: str, local: str) -> None:
    """Download a remote file or directory tree to a local path (idempotent)."""
    remote = _norm_remote(remote)
    if not client.exists(remote):
        _fail(f"remote path does not exist: {remote}")
    local_path = Path(local)

    if not _remote_is_dir(client, remote):
        local_path.parent.mkdir(parents=True, exist_ok=True)
        _pull_file(client, remote, local_path)
        return

    entries = list(_walk_remote(client, remote))
    total = len(entries)
    for i, rfile in enumerate(entries, 1):
        rel = PurePosixPath(rfile).relative_to(remote).as_posix()
        dst = local_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        _pull_file(client, rfile, dst, prefix=f"[{i}/{total}] ")


def _pull_file(client, remote: str, local_file: Path, prefix: str = "") -> None:
    rsize = _remote_file_size(client, remote)
    if local_file.exists() and rsize is not None and local_file.stat().st_size == rsize:
        print(f"{prefix}[skip] {local_file} (same size {rsize})", flush=True)
        return
    print(f"{prefix}[pull] {remote} -> {local_file} ({_fmt_bytes(rsize or 0)})", flush=True)
    link = client.get_download_link(remote)
    tmp = local_file.with_suffix(local_file.suffix + ".partial")
    tmp.unlink(missing_ok=True)
    _progress_download(link, tmp, f"{prefix}[download]", rsize)
    tmp.replace(local_file)


def _remote_is_dir(client, remote: str) -> bool:
    try:
        meta = client.get_meta(remote)
    except Exception:  # noqa: BLE001
        return False
    return getattr(meta, "type", None) == "dir"


def _walk_remote(client, remote: str):
    """Yield every file path under a remote directory (recursive, sorted)."""
    stack = [remote]
    while stack:
        cur = stack.pop()
        for item in client.listdir(cur):
            path = f"{cur}/{item.name}"
            if getattr(item, "type", None) == "dir":
                stack.append(path)
            else:
                yield path


def rm(client, remote: str) -> None:
    remote = _norm_remote(remote)
    if not client.exists(remote):
        print(f"[ydisk] [skip-rm] {remote} (absent)", flush=True)
        return
    print(f"[ydisk] [rm] {remote}", flush=True)
    client.remove(remote, permanently=True)


def ls(client, remote: str) -> None:
    remote = _norm_remote(remote)
    if not client.exists(remote):
        _fail(f"remote path does not exist: {remote}")
    if not _remote_is_dir(client, remote):
        size = _remote_file_size(client, remote)
        print(f"{remote}\t{size} B")
        return
    for item in client.listdir(remote):
        kind = "d" if getattr(item, "type", None) == "dir" else "-"
        size = getattr(item, "size", "") or ""
        print(f"{kind} {item.name}\t{size}")


def inventory(client, remote: str) -> None:
    """Print recursive file count and total bytes for a remote path."""
    remote = _norm_remote(remote)
    if not client.exists(remote):
        _fail(f"remote path does not exist: {remote}")
    if not _remote_is_dir(client, remote):
        size = _remote_file_size(client, remote) or 0
        print(f"files=1 bytes={size} size={_fmt_bytes(size)} path={remote}")
        return

    files = 0
    total = 0
    by_suffix: dict[str, tuple[int, int]] = {}
    for rfile in _walk_remote(client, remote):
        files += 1
        size = _remote_file_size(client, rfile) or 0
        total += size
        suffix = PurePosixPath(rfile).suffix or "<none>"
        count, subtotal = by_suffix.get(suffix, (0, 0))
        by_suffix[suffix] = (count + 1, subtotal + size)
    print(f"files={files} bytes={total} size={_fmt_bytes(total)} path={remote}")
    for suffix, (count, subtotal) in sorted(by_suffix.items()):
        print(f"suffix={suffix} files={count} bytes={subtotal} size={_fmt_bytes(subtotal)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Yandex.Disk staging helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_push = sub.add_parser("push", help="upload local file/dir to Yandex.Disk")
    p_push.add_argument("local")
    p_push.add_argument("remote")

    p_pull = sub.add_parser("pull", help="download remote file/dir from Yandex.Disk")
    p_pull.add_argument("remote")
    p_pull.add_argument("local")

    p_rm = sub.add_parser("rm", help="remove a remote path (permanently)")
    p_rm.add_argument("remote")

    p_ls = sub.add_parser("ls", help="list a remote path")
    p_ls.add_argument("remote")

    p_inv = sub.add_parser("inventory", help="recursive remote file count and total bytes")
    p_inv.add_argument("remote")

    args = parser.parse_args(argv)
    client = get_client()

    if args.cmd == "push":
        push(client, args.local, args.remote)
    elif args.cmd == "pull":
        pull(client, args.remote, args.local)
    elif args.cmd == "rm":
        rm(client, args.remote)
    elif args.cmd == "ls":
        ls(client, args.remote)
    elif args.cmd == "inventory":
        inventory(client, args.remote)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
