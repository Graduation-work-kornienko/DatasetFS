"""Create an augmented Speech Commands ImageFolder tree.

Unlike ``replicate_speech_commands.py``, this script writes physically distinct
WAV files. Replicas keep the ImageFolder layout but apply deterministic, small
audio perturbations so the resulting dataset is a large collection of distinct
logical objects rather than hardlink clones.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf


VALID_SUFFIXES = {".wav"}


def _samples(source: Path) -> list[tuple[Path, str]]:
    rows: list[tuple[Path, str]] = []
    for class_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and not path.name.startswith("._") and path.suffix.lower() in VALID_SUFFIXES:
                rows.append((path, class_dir.name))
    if not rows:
        raise SystemExit(f"no .wav samples found under {source}")
    return rows


def _augment(audio: np.ndarray, replica: int, sample_idx: int) -> tuple[np.ndarray, str]:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]

    rng = np.random.default_rng((replica + 1) * 1_000_003 + sample_idx)
    ops: list[str] = []

    gain = 10 ** (rng.uniform(-4.0, 4.0) / 20.0)
    x = x * gain
    ops.append(f"gain={gain:.4f}")

    if replica % 2 == 1 and len(x) > 8:
        shift = int(rng.integers(1, max(2, min(len(x), 1600))))
        x = np.roll(x, shift, axis=0)
        ops.append(f"roll={shift}")

    if replica % 3 == 1:
        noise_std = float(rng.uniform(0.0005, 0.0030))
        x = x + rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)
        ops.append(f"noise={noise_std:.5f}")

    if replica % 5 == 2 and len(x) > 32:
        width = int(rng.integers(max(8, len(x) // 80), max(9, len(x) // 20)))
        start = int(rng.integers(0, max(1, len(x) - width)))
        x[start:start + width] *= float(rng.uniform(0.05, 0.35))
        ops.append(f"mask={start}:{start + width}")

    if replica % 7 == 3:
        x = -x
        ops.append("polarity=-1")

    x = np.clip(x, -0.999, 0.999)
    if audio.ndim == 1:
        x = x[:, 0]
    return x.astype(np.float32, copy=False), ";".join(ops)


def augment_audio(audio: np.ndarray, replica: int, sample_idx: int) -> tuple[np.ndarray, str]:
    """Public wrapper used by streaming format preparers."""
    return _augment(audio, replica, sample_idx)


def replicate(source: Path, output: Path, replicas: int, overwrite: bool) -> None:
    source = source.resolve()
    if not source.exists():
        raise SystemExit(f"source does not exist: {source}")
    if output.exists():
        if not overwrite:
            raise SystemExit(f"output already exists: {output}; pass --overwrite to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    base = _samples(source)
    total_files = len(base) * replicas
    total_payload = sum(path.stat().st_size for path, _ in base) * replicas
    print(
        f"[replicate-aug] source={len(base)} files replicas={replicas} "
        f"target={total_files} files logical_payload={total_payload / (1024**3):.2f} GiB",
        flush=True,
    )

    manifest_path = output / "replication_manifest.csv"
    done = 0
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["logical_path", "source_path", "label", "replica", "size_bytes", "augmentation"],
        )
        writer.writeheader()
        for replica in range(replicas):
            for sample_idx, (src, label) in enumerate(base):
                data, sr = sf.read(src, dtype="float32", always_2d=False)
                augmented, aug_desc = _augment(data, replica, sample_idx)
                rel_name = f"rep{replica:04d}__{src.name}"
                dst_dir = output / label
                dst_dir.mkdir(exist_ok=True)
                dst = dst_dir / rel_name
                sf.write(dst, augmented, sr, subtype="PCM_16")
                writer.writerow({
                    "logical_path": str(dst.relative_to(output)),
                    "source_path": str(src),
                    "label": label,
                    "replica": replica,
                    "size_bytes": dst.stat().st_size,
                    "augmentation": aug_desc,
                })
                done += 1
                if done % 25_000 == 0:
                    print(f"[replicate-aug] wrote {done}/{total_files}", flush=True)

    (output / ".done").touch()
    (output / "README.txt").write_text(
        "Augmented Speech Commands ImageFolder tree for object-count scaling.\n"
        f"source={source}\n"
        f"replicas={replicas}\n"
        f"files={total_files}\n"
        f"logical_payload_bytes={total_payload}\n",
        encoding="utf-8",
    )
    print(f"[replicate-aug] done -> {output}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/formats/speech_commands/imagefolder"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicas", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.replicas < 1:
        raise SystemExit("--replicas must be >= 1")
    replicate(args.source, args.output, args.replicas, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
