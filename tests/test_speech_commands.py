"""Correctness tests on Speech Commands V2 (audio classification, 35 classes).

Validates DatasetFS on a fundamentally different modality (WAV audio, not images).
Replicates the P0/P1 image tests AND adds audio-specific assertions:
  - WAV decodes successfully (uncorrupted bytes)
  - Sample rate is 16 kHz (Speech Commands invariant)
  - Waveform shape is (1, ~16000) — 1 second mono
  - Not all-zero / not all-clipped (sanity)
"""
from __future__ import annotations

import functools
import io
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader, get_worker_info

from clients.python import DatasetFS

from tests.helpers import imagefolder_index, path_key


pytestmark = pytest.mark.timeout(1800)


# ---- Decoders / collates (module-level for picklability on macOS spawn) ----
# Using `soundfile` directly instead of torchaudio.load because PyTorch ≥2.9
# routes torchaudio.load through `torchcodec`, which has no macOS wheels.


def _decode_wav(raw_bytes: bytes):
    """Decode WAV bytes → (waveform_tensor [C, N], sample_rate)."""
    data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=True)
    # soundfile returns (N, C) when always_2d=True; transpose to (C, N) for
    # torchaudio-style layout.
    waveform = torch.from_numpy(data.T.copy())
    return (waveform, sr)


def _identity(x):
    return x


def _audio_tagging_collate(items):
    """Captures worker_id + path + label + sample_rate + first-sample tensor stats."""
    info = get_worker_info()
    wid = info.id if info is not None else 0
    return {
        "worker_id": wid,
        "paths": [it["path"] for it in items],
        "labels": [it.get("label") for it in items],
        "sample_rates": [it["image"][1] for it in items],
        "shapes": [tuple(it["image"][0].shape) for it in items],
        "abs_max": [float(it["image"][0].abs().max()) for it in items],
    }


def _read_all_audio(num_workers: int):
    """Drain one epoch. Returns list of (wid, path, label, sample_rate, shape, abs_max)."""
    ds = DatasetFS(
        num_workers=num_workers,
        decode_fn=_decode_wav,
        transform=_identity,
    )
    loader = DataLoader(
        ds,
        batch_size=64,
        num_workers=num_workers,
        collate_fn=_audio_tagging_collate,
    )
    samples = []
    for batch in loader:
        wid = batch["worker_id"]
        for path, label, sr, shape, amax in zip(
            batch["paths"], batch["labels"], batch["sample_rates"], batch["shapes"], batch["abs_max"]
        ):
            samples.append((wid, path, label, sr, shape, amax))
    return samples


# ---- Replicated correctness tests (P0/P1) on audio ----


def test_audio_completeness_workers4(daemon_speech_commands, speech_commands_prepared):
    """All WAV files in source must come through DatasetFS exactly once."""
    truth = imagefolder_index(speech_commands_prepared["imagefolder"])
    samples = _read_all_audio(num_workers=4)
    got = [path_key(p) for _, p, _, _, _, _ in samples]

    assert set(got) == set(truth.keys()), (
        f"missing {len(set(truth.keys()) - set(got))}, "
        f"extra {len(set(got) - set(truth.keys()))}"
    )


def test_audio_no_duplicates(daemon_speech_commands, speech_commands_prepared):
    samples = _read_all_audio(num_workers=4)
    got = [path_key(p) for _, p, _, _, _, _ in samples]
    counts = Counter(got)
    dups = {k: c for k, c in counts.items() if c > 1}
    assert not dups, f"duplicates: {list(dups.items())[:5]}"


def test_audio_multiworker_disjoint(daemon_speech_commands, speech_commands_prepared):
    samples = _read_all_audio(num_workers=4)
    by_worker: dict[int, set[str]] = {}
    for wid, path, _, _, _, _ in samples:
        by_worker.setdefault(wid, set()).add(path_key(path))

    assert len(by_worker) == 4, f"expected 4 workers, got {sorted(by_worker)}"
    worker_ids = sorted(by_worker)
    for i in range(len(worker_ids)):
        for j in range(i + 1, len(worker_ids)):
            inter = by_worker[worker_ids[i]] & by_worker[worker_ids[j]]
            assert not inter, (
                f"workers {worker_ids[i]} and {worker_ids[j]} share {len(inter)} files"
            )


def test_audio_labels_match_per_sample(daemon_speech_commands, speech_commands_prepared):
    truth = imagefolder_index(speech_commands_prepared["imagefolder"])
    samples = _read_all_audio(num_workers=4)

    mismatches = []
    for _, path, label, _, _, _ in samples:
        name = path_key(path)
        expected = truth.get(name)
        if expected is None or label is None or label != expected:
            mismatches.append((name, expected, label))

    assert not mismatches, f"{len(mismatches)} label mismatches; first 5: {mismatches[:5]}"


# ---- Audio-specific tests ----


def test_audio_sample_rate_is_16khz(daemon_speech_commands):
    """Speech Commands V2 invariant: every WAV is sampled at 16 kHz. Verifies
    that bytes are uncorrupted (a wrong byte range would yield bad WAV header
    → torchaudio either fails or returns wrong sample rate)."""
    samples = _read_all_audio(num_workers=4)
    wrong_sr = [(path_key(p), sr) for _, p, _, sr, _, _ in samples if sr != 16000]
    assert not wrong_sr, (
        f"{len(wrong_sr)} files with sample rate != 16000; first 5: {wrong_sr[:5]}"
    )


def test_audio_waveform_shape_plausible(daemon_speech_commands):
    """Mono channel, length in plausible range. Speech Commands clips are
    nominally 1 second @ 16 kHz = 16000 samples, but some are shorter
    (truncated recordings) — allow a generous range."""
    samples = _read_all_audio(num_workers=4)

    bad_shape = []
    bad_length = []
    for _, path, _, _, shape, _ in samples:
        if len(shape) != 2 or shape[0] != 1:
            bad_shape.append((path_key(path), shape))
            continue
        n = shape[1]
        # 1ms to 2s — generous bounds. Real corruption would be either
        # 0-length or huge multi-second buffers.
        if n < 16 or n > 32000:
            bad_length.append((path_key(path), n))

    assert not bad_shape, f"unexpected shapes (not [1, N]): {bad_shape[:5]}"
    assert not bad_length, f"implausible audio lengths: {bad_length[:5]}"


def test_audio_not_all_silent(daemon_speech_commands):
    """Real speech recordings should have non-trivial amplitude. If bytes were
    corrupted to all-zero, every clip would silently 'decode' to silence."""
    samples = _read_all_audio(num_workers=4)
    silent = [
        (path_key(p), amax)
        for _, p, _, _, _, amax in samples
        if amax < 1e-4  # below typical noise floor for 16-bit PCM
    ]
    # Tolerate up to 0.5% silent clips (occasional bad recordings exist)
    silent_fraction = len(silent) / len(samples) if samples else 1
    assert silent_fraction < 0.005, (
        f"{len(silent)}/{len(samples)} clips effectively silent (max abs < 1e-4); "
        f"first 5: {silent[:5]}"
    )


def test_audio_bytes_match_random_sample(daemon_speech_commands, speech_commands_prepared):
    """Hash decoded waveforms from DFS, compare against hash of decoding the
    same files directly from the source. Same as image byte-level test but for WAV."""
    from tests.helpers import imagefolder_paths, hash_tensor

    truth_paths = imagefolder_paths(speech_commands_prepared["imagefolder"])

    # Collect DFS waveform hashes (workers=0 keeps it simple)
    ds = DatasetFS(num_workers=0, decode_fn=_decode_wav, transform=_identity)

    def _collate_pair(items):
        return [(it["path"], it["image"]) for it in items]

    loader = DataLoader(ds, batch_size=64, num_workers=0, collate_fn=_collate_pair)
    dfs_hashes: dict[str, str] = {}
    for batch in loader:
        for path, (waveform, sr) in batch:
            name = path_key(path)
            assert name not in dfs_hashes, f"dup in DFS output: {name}"
            dfs_hashes[name] = hash_tensor(waveform)

    rng = random.Random(0)
    sample_names = rng.sample(list(truth_paths.keys()), k=min(100, len(truth_paths)))

    mismatches = []
    for name in sample_names:
        if name not in dfs_hashes:
            mismatches.append(f"{name}: not in DFS output")
            continue
        expected_data, _ = sf.read(str(truth_paths[name]), dtype="float32", always_2d=True)
        expected_wf = torch.from_numpy(expected_data.T.copy())
        expected = hash_tensor(expected_wf)
        if dfs_hashes[name] != expected:
            mismatches.append(
                f"{name}: DFS={dfs_hashes[name][:12]}.. expected={expected[:12]}.."
            )

    assert not mismatches, (
        f"{len(mismatches)}/{len(sample_names)} byte mismatches; first 3: {mismatches[:3]}"
    )
