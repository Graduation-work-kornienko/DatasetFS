"""Stream augmented Speech Commands replicas directly into DatasetFS."""
from __future__ import annotations

import argparse
import io
import shutil
from pathlib import Path

import soundfile as sf

from scripts.datasets.datasetfs_writer import DatasetFSWriter
from scripts.datasets.replicate_speech_commands_augmented import _samples, augment_audio


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/formats/speech_commands/imagefolder"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicas", type=int, required=True)
    parser.add_argument("--max-objects", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists: {args.output}; pass --overwrite")
        shutil.rmtree(args.output)

    base = _samples(args.source.resolve())
    target = len(base) * args.replicas
    if args.max_objects is not None:
        target = min(target, args.max_objects)
    print(
        f"[prepare-aug-datasetfs] base={len(base)} replicas={args.replicas} target={target}",
        flush=True,
    )

    written = 0
    with DatasetFSWriter(args.output, shard_target_bytes=96 * 1024 * 1024) as writer:
        for replica in range(args.replicas):
            for sample_idx, (src, label) in enumerate(base):
                data, sr = sf.read(src, dtype="float32", always_2d=False)
                augmented, aug_desc = augment_audio(data, replica, sample_idx)
                buf = io.BytesIO()
                sf.write(buf, augmented, sr, format="WAV", subtype="PCM_16")
                writer.add(
                    f"{label}/rep{replica:04d}__{src.name}",
                    buf.getvalue(),
                    {"label": label, "replica": replica, "augmentation": aug_desc},
                )
                written += 1
                if written % 25_000 == 0:
                    print(f"[prepare-aug-datasetfs] wrote {written}/{target}", flush=True)
                if written >= target:
                    print(f"[prepare-aug-datasetfs] done -> {args.output}", flush=True)
                    return 0
    print(f"[prepare-aug-datasetfs] done -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
