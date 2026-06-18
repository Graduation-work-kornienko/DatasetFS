"""Run DatasetFS universality probes on real prepared datasets.

The YAML config is intentionally explicit about large dataset preparation. This
runner does not download multi-GB data implicitly; it validates what is present,
runs lightweight training probes, and writes missing.csv for absent datasets.
"""
from __future__ import annotations

import argparse
import csv
import functools
import io
import os
import signal
import subprocess
import time
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset
import torchvision.transforms as T

from benchmarks.datasetfs_bench.metrics import daemon as daemon_metrics
from benchmarks.datasetfs_bench.metrics.system import SystemSampler
from clients.python import DatasetFS
from scripts.datasets.datasetfs_writer import read_parquet_manifest


DAEMON_URL = "http://localhost:51409"


def _cleanup_tmp_files() -> None:
    for path in ["/tmp/mlfs_data.bin", "/tmp/mlfs_refs.bin"]:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    for fifo in Path("/tmp").glob("datasetfs_pipe_*"):
        try:
            fifo.unlink()
        except FileNotFoundError:
            pass


def _wait_for_healthz(timeout_s: float = 30.0) -> None:
    import requests
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{DAEMON_URL}/healthz", timeout=1)
            if r.status_code == 200:
                return
        except Exception as e:
            last = e
        time.sleep(0.1)
    raise RuntimeError(f"daemon did not become healthy: {last}")


class Daemon:
    def __init__(self, binary: Path, root: Path, log_dir: Path):
        self.binary = binary
        self.root = root
        self.log_dir = log_dir
        self.proc = None
        self.log_file = None
        self.log_path = None

    def start(self) -> None:
        _cleanup_tmp_files()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"daemon-real-universal-{int(time.time()*1000)}.log"
        self.log_file = open(self.log_path, "w")
        self.proc = subprocess.Popen(
            [str(self.binary), "daemon", "--no-mount", "--no-wal", "--root", str(self.root)],
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
        _wait_for_healthz()

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                else:
                    self.proc.terminate()
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                else:
                    self.proc.kill()
                self.proc.wait(timeout=5)
        if self.log_file is not None:
            self.log_file.close()
        _cleanup_tmp_files()


class VectorProbe(nn.Module):
    def __init__(self, dim: int, classes: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, classes))

    def forward(self, x):
        return self.net(x)


class ImageProbe(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, classes),
        )

    def forward(self, x):
        return self.net(x)


class FusionProbe(nn.Module):
    def __init__(self, classes: int, tab_dim: int = 8):
        super().__init__()
        self.image = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.tab = nn.Sequential(nn.Linear(tab_dim, 16), nn.ReLU())
        self.head = nn.Linear(32, classes)

    def forward(self, image, tab):
        return self.head(torch.cat([self.image(image), self.tab(tab)], dim=1))


def _decode_text(raw) -> torch.Tensor:
    data = bytes(raw).lower()
    hist = torch.zeros(128, dtype=torch.float32)
    for b in data[:8192]:
        hist[b % 128] += 1.0
    return hist / max(1.0, hist.sum())


def _decode_audio(raw) -> torch.Tensor:
    try:
        import soundfile as sf
        data, _sr = sf.read(io.BytesIO(bytes(raw)), dtype="float32", always_2d=False)
        samples = torch.as_tensor(data, dtype=torch.float32).flatten()
    except Exception:
        try:
            with wave.open(io.BytesIO(bytes(raw)), "rb") as wf:
                frames = wf.readframes(wf.getnframes())
            samples = torch.from_numpy(np.frombuffer(frames, dtype="<i2").astype("float32")) / 32768.0
        except Exception:
            return torch.zeros(33, dtype=torch.float32)
    if samples.numel() == 0:
        return torch.zeros(33, dtype=torch.float32)
    chunks = torch.chunk(samples, 32)
    energy = torch.stack([c.abs().mean() for c in chunks])
    zc = (samples[:-1] * samples[1:] < 0).float().mean().view(1) if samples.numel() > 1 else torch.zeros(1)
    return torch.cat([energy, zc])


def _decode_video(raw) -> torch.Tensor:
    data = bytes(raw)
    try:
        import imageio.v3 as iio
        frames = iio.imread(io.BytesIO(data), index=None)
        arr = np.asarray(frames)
        if arr.size:
            arr = arr.reshape((-1,) + arr.shape[-3:])[:16].astype("float32") / 255.0
            means = torch.as_tensor(arr.mean(axis=(1, 2)), dtype=torch.float32).flatten()
            stds = torch.as_tensor(arr.std(axis=(1, 2)), dtype=torch.float32).flatten()
            feat = torch.cat([means, stds])[:64]
            return torch.nn.functional.pad(feat, (0, max(0, 64 - feat.numel())))
    except Exception:
        pass
    hist = torch.zeros(64, dtype=torch.float32)
    for b in data[:65536]:
        hist[b % 64] += 1.0
    return hist / max(1.0, hist.sum())


_IMG_TF = T.Compose([T.Resize((160, 160)), T.ToTensor()])


def _decode_image(raw):
    try:
        return Image.open(io.BytesIO(bytes(raw))).convert("RGB")
    except Exception:
        return Image.new("RGB", (160, 160))


def _identity(x):
    return x


def _label(item: dict) -> str:
    if "label" in item:
        return str(item["label"])
    if "caption" in item:
        return str(len(str(item["caption"])) % 4)
    if "text" in item:
        return str(len(str(item["text"])) % 4)
    return str(hash(item.get("path", "")) % 4)


class ParquetTextDataset(IterableDataset):
    def __init__(self, path: Path):
        self.path = path

    def __iter__(self):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(self.path)
        for batch in pf.iter_batches(columns=["text", "label", "path"], batch_size=1024):
            data = batch.to_pydict()
            for text, label, path in zip(data["text"], data["label"], data["path"]):
                yield {"image": _decode_text(str(text).encode("utf-8")), "label": str(label), "path": str(path)}


class VideoFolderDataset(IterableDataset):
    def __init__(self, root: Path):
        self.root = root

    def __iter__(self):
        exts = {".avi", ".mp4", ".mov", ".mkv", ".webm"}
        for path in sorted(p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in exts):
            label = path.parent.name if path.parent != self.root else path.stem.split("_")[1]
            yield {"image": _decode_video(path.read_bytes()), "label": label, "path": str(path)}


def _vector_collate(items, label_to_idx: dict[str, int]):
    x = torch.stack([it["image"] for it in items])
    y = torch.tensor([label_to_idx[_label(it)] for it in items], dtype=torch.long)
    return x, y


def _image_collate(items, label_to_idx: dict[str, int]):
    x = torch.stack([it["image"] for it in items])
    y = torch.tensor([label_to_idx[_label(it)] for it in items], dtype=torch.long)
    return x, y


def _fusion_collate(items, label_to_idx: dict[str, int]):
    tabs = []
    for it in items:
        if "tab" in it:
            v = torch.as_tensor(it["tab"], dtype=torch.float32)
        elif "bbox" in it:
            flat = torch.as_tensor(it["bbox"], dtype=torch.float32).flatten()
            v = torch.nn.functional.pad(flat[:8], (0, max(0, 8 - flat.numel())))
        else:
            text = str(it.get("caption") or it.get("text") or "")
            v = torch.tensor([len(text), len(set(text)), text.count(" "), len(it.get("path", "")), 0, 0, 0, 0], dtype=torch.float32)
        if v.numel() < 8:
            v = torch.nn.functional.pad(v, (0, 8 - v.numel()))
        tabs.append(v[:8])
    inputs = {"image": torch.stack([it["image"] for it in items]), "tab": torch.stack(tabs)}
    y = torch.tensor([label_to_idx[_label(it)] for it in items], dtype=torch.long)
    return inputs, y


def _collect_labels(root: Path, modality: str) -> list[str]:
    manifest = read_parquet_manifest(root)
    labels = []
    for info in manifest.get("files", {}).values():
        if info.get("deleted"):
            continue
        labels.append(_label(info.get("meta") or {"path": info.get("path", "")}))
    labels.extend([str(i) for i in range(4)])
    uniq = sorted(set(labels))
    return uniq or ["0", "1"]


def _collect_labels_from_iterable(ds, limit: int = 512) -> list[str]:
    labels = []
    for i, item in enumerate(ds):
        labels.append(_label(item))
        if i + 1 >= limit:
            break
    uniq = sorted(set(labels))
    return uniq or ["0", "1"]


def _build_baseline_loader(format_cfg: dict, modality: str, batch_size: int):
    fmt = format_cfg["format"]
    path = Path(format_cfg["path"])
    if fmt == "parquet" and modality == "text":
        ds = ParquetTextDataset(path)
        labels = _collect_labels_from_iterable(ParquetTextDataset(path))
        loader = DataLoader(ds, batch_size=batch_size, num_workers=0, collate_fn=functools.partial(_vector_collate, label_to_idx={l: i for i, l in enumerate(labels)}))
        return loader, VectorProbe(128, len(labels)), len(labels)
    if fmt == "folder" and modality == "video":
        ds = VideoFolderDataset(path)
        labels = _collect_labels_from_iterable(VideoFolderDataset(path))
        loader = DataLoader(ds, batch_size=batch_size, num_workers=0, collate_fn=functools.partial(_vector_collate, label_to_idx={l: i for i, l in enumerate(labels)}))
        return loader, VectorProbe(64, len(labels)), len(labels)
    raise ValueError(f"unsupported baseline format={fmt!r} modality={modality!r}")


def _run_training(ds_cfg: dict, args, out_dir: Path) -> tuple[dict, list[dict]]:
    root = Path(ds_cfg["datasetfs"])
    modality = ds_cfg["modality"]
    profile = _dataset_profile(root)
    daemon = Daemon(args.binary, root, out_dir / "logs")
    daemon.start()
    try:
        labels = _collect_labels(root, modality)
        label_to_idx = {l: i for i, l in enumerate(labels)}
        batch_size = int(ds_cfg.get("batch_size", args.batch_size))
        max_batches = int(args.max_batches)

        if modality in ("image",):
            ds = DatasetFS(num_workers=0, decode_fn=_decode_image, transform=_IMG_TF, timeout_seconds=10)
            loader = DataLoader(ds, batch_size=batch_size, num_workers=0, collate_fn=functools.partial(_image_collate, label_to_idx=label_to_idx))
            model = ImageProbe(len(label_to_idx))
        elif modality in ("image_text", "image_regions"):
            ds = DatasetFS(num_workers=0, decode_fn=_decode_image, transform=_IMG_TF, timeout_seconds=10)
            loader = DataLoader(ds, batch_size=batch_size, num_workers=0, collate_fn=functools.partial(_fusion_collate, label_to_idx=label_to_idx))
            model = FusionProbe(len(label_to_idx))
        elif modality in ("audio", "audio_text"):
            ds = DatasetFS(num_workers=0, decode_fn=_decode_audio, transform=_identity, timeout_seconds=10)
            loader = DataLoader(ds, batch_size=batch_size, num_workers=0, collate_fn=functools.partial(_vector_collate, label_to_idx=label_to_idx))
            model = VectorProbe(33, len(label_to_idx))
        elif modality == "text":
            ds = DatasetFS(num_workers=0, decode_fn=_decode_text, transform=_identity, timeout_seconds=10)
            loader = DataLoader(ds, batch_size=batch_size, num_workers=0, collate_fn=functools.partial(_vector_collate, label_to_idx=label_to_idx))
            model = VectorProbe(128, len(label_to_idx))
        elif modality == "video":
            ds = DatasetFS(num_workers=0, decode_fn=_decode_video, transform=_identity, timeout_seconds=15)
            loader = DataLoader(ds, batch_size=batch_size, num_workers=0, collate_fn=functools.partial(_vector_collate, label_to_idx=label_to_idx))
            model = VectorProbe(64, len(label_to_idx))
        else:
            raise ValueError(f"unsupported modality {modality!r}")

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        n_batches = 0
        n_samples = 0
        losses = []
        sampler = SystemSampler(
            interval_s=0.2,
            track_pids=[os.getpid()] + ([daemon.proc.pid] if daemon.proc and daemon.proc.poll() is None else []),
            track_labels={"python": os.getpid(), **({"daemon": daemon.proc.pid} if daemon.proc and daemon.proc.poll() is None else {})},
        )
        daemon_sampler = daemon_metrics.DaemonSampler(
            interval_s=0.5,
            context={"name": ds_cfg["name"], "modality": modality},
        )
        daemon_before = daemon_metrics.snapshot()
        t0 = time.perf_counter()
        sampler.start()
        daemon_sampler.start()
        try:
            for n_batches, (inputs, targets) in enumerate(loader, start=1):
                if n_batches > max_batches:
                    break
                opt.zero_grad()
                out = model(**inputs) if isinstance(inputs, dict) else model(inputs)
                loss = loss_fn(out, targets)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
                n_samples += int(targets.shape[0])
        finally:
            daemon_sampler.stop()
            sampler.stop()
        daemon_after = daemon_metrics.snapshot()
        wall = time.perf_counter() - t0
        if not losses:
            raise RuntimeError("no batches produced")
        row = {
            "name": ds_cfg["name"],
            "format": "datasetfs",
            "task": ds_cfg["task"],
            "modality": modality,
            "status": "ok",
            **profile,
            "n_batches": min(n_batches, max_batches),
            "n_samples": n_samples,
            "wall_s": wall,
            "samples_per_s": n_samples / wall if wall > 0 else 0,
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "classes_seen": len(label_to_idx),
            "daemon_log": str(daemon.log_path),
        }
        row.update({f"sys_{k}": v for k, v in sampler.summary().items()})
        row.update(daemon_metrics.cell_summary(daemon_before, daemon_after))
        return row, daemon_sampler.samples
    finally:
        daemon.stop()


def _run_baseline_training(ds_cfg: dict, format_cfg: dict, args) -> dict:
    modality = format_cfg.get("modality", ds_cfg["modality"])
    batch_size = int(format_cfg.get("batch_size", ds_cfg.get("batch_size", args.batch_size)))
    max_batches = int(args.max_batches)
    loader, model, classes_seen = _build_baseline_loader(format_cfg, modality, batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    sampler = SystemSampler(interval_s=0.2, track_pids=[os.getpid()], track_labels={"python": os.getpid()})
    losses = []
    n_batches = 0
    n_samples = 0
    t0 = time.perf_counter()
    sampler.start()
    try:
        for n_batches, (inputs, targets) in enumerate(loader, start=1):
            if n_batches > max_batches:
                break
            opt.zero_grad()
            out = model(**inputs) if isinstance(inputs, dict) else model(inputs)
            loss = loss_fn(out, targets)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            n_samples += int(targets.shape[0])
    finally:
        sampler.stop()
    wall = time.perf_counter() - t0
    if not losses:
        raise RuntimeError("no batches produced")
    row = {
        "name": ds_cfg["name"],
        "format": format_cfg["format"],
        "task": ds_cfg["task"],
        "modality": modality,
        "status": "ok",
        "n_batches": min(n_batches, max_batches),
        "n_samples": n_samples,
        "wall_s": wall,
        "samples_per_s": n_samples / wall if wall > 0 else 0,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "classes_seen": classes_seen,
    }
    row.update({f"sys_{k}": v for k, v in sampler.summary().items()})
    return row


def _dataset_profile(root: Path) -> dict:
    manifest_path = root / "metadata.parquet"
    if not manifest_path.exists():
        return {}
    try:
        manifest = read_parquet_manifest(root)
    except Exception:
        return {}
    files = manifest.get("files", {})
    live = [m for m in files.values() if not m.get("deleted")]
    sizes = [int(m.get("size", 0) or 0) for m in live]
    shards = manifest.get("shards_meta", {})
    shard_sizes = [int(s.get("total_size", 0) or 0) for s in shards.values()]
    total = sum(sizes)
    return {
        "dataset_object_count": len(live),
        "dataset_total_bytes": total,
        "dataset_avg_object_bytes": (total / len(sizes)) if sizes else 0,
        "dataset_max_object_bytes": max(sizes) if sizes else 0,
        "dataset_shard_count": len(shards),
        "dataset_avg_shard_bytes": (sum(shard_sizes) / len(shard_sizes)) if shard_sizes else 0,
        "dataset_max_shard_bytes": max(shard_sizes) if shard_sizes else 0,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("benchmarks/datasetfs_bench/configs/real_universal_datasets.yaml"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--binary", type=Path, default=Path("bin/datasetfs"))
    p.add_argument("--max-batches", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--require-all", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    daemon_timeseries_rows = []
    missing = []
    for ds_cfg in cfg["datasets"]:
        root = Path(ds_cfg["datasetfs"])
        if root.exists():
            print(f"[run] {ds_cfg['name']} ({ds_cfg['modality']})", flush=True)
            try:
                row, daemon_samples = _run_training(ds_cfg, args, args.output)
                rows.append(row)
                daemon_timeseries_rows.extend(daemon_samples)
                daemon_metrics.write_rows_union(args.output / "daemon_timeseries.csv", daemon_timeseries_rows)
            except Exception as e:
                rows.append({"name": ds_cfg["name"], "format": "datasetfs", "modality": ds_cfg["modality"], "status": "error", "error": repr(e)})
                if args.require_all:
                    raise
        else:
            missing.append({
                "name": ds_cfg["name"],
                "format": "datasetfs",
                "modality": ds_cfg["modality"],
                "datasetfs": str(root),
                "prepare": ds_cfg.get("prepare", ""),
            })
            print(f"[missing] {ds_cfg['name']}: {root}", flush=True)

        for format_cfg in ds_cfg.get("formats", []):
            fmt_path = Path(format_cfg["path"])
            fmt_name = format_cfg["format"]
            modality = format_cfg.get("modality", ds_cfg["modality"])
            if not fmt_path.exists():
                missing.append({
                    "name": ds_cfg["name"],
                    "format": fmt_name,
                    "modality": modality,
                    "datasetfs": str(fmt_path),
                    "prepare": ds_cfg.get("prepare", ""),
                })
                print(f"[missing] {ds_cfg['name']}/{fmt_name}: {fmt_path}", flush=True)
                continue
            print(f"[run] {ds_cfg['name']}/{fmt_name} ({modality})", flush=True)
            try:
                rows.append(_run_baseline_training(ds_cfg, format_cfg, args))
            except Exception as e:
                rows.append({"name": ds_cfg["name"], "format": fmt_name, "modality": modality, "status": "error", "error": repr(e)})
                if args.require_all:
                    raise
    _write_csv(args.output / "summary.csv", rows)
    _write_csv(args.output / "missing.csv", missing)
    if missing and args.require_all:
        raise SystemExit(f"missing {len(missing)} datasets; see {args.output / 'missing.csv'}")
    print(f"[real-universal] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
