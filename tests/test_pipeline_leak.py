"""Regression test for the opt-03 slot-leak fix.

Before opt 03, the Python client decremented a slot's refcount only on the
*yield* path — every `continue` for a skipped sample (decode returned None,
size mismatch, transform raised) bypassed the decrement. A slot containing any
skipped sample therefore never reached refcount 0, so the Go planner never
recycled it. With more shards than shared-memory slots (Speech Commands has 37
shards vs 9 slots, so a single worker must recycle), enough skips would exhaust
the slots and stall the epoch — it would then end only via the 30 s idle
timeout, having served a small fraction of the data.

opt 03 counts EVERY item toward its slot (skipped or not) and decrements once
per slot per frame, so slots always recycle. This test drives a decode_fn that
skips half the samples and asserts the epoch still traverses the whole dataset
(≈ half the items survive), which is only possible if slots recycle.

A no-op decode_fn is used (no real audio decode) so the drain runs at transport
speed — the test is about slot lifecycle, not decoding.
"""
from __future__ import annotations

import pytest

from clients.python import DatasetFS

pytestmark = pytest.mark.timeout(120)


def _identity(x):
    return x


def test_skipped_samples_do_not_leak_slots(daemon_speech_commands):
    """With more shards than slots, skipping half the samples must NOT stall the
    epoch: every slot still recycles, so ≈ half the dataset survives."""
    url = daemon_speech_commands.url

    def drain(skip_fn):
        # Fresh session each drain (re-init the daemon's loading pipelines).
        ds = DatasetFS(num_workers=0, seed=0, decode_fn=skip_fn, transform=_identity,
                       timeout_seconds=20.0, daemon_url=url)
        return sum(1 for _ in ds)

    # Baseline: keep everything → full dataset size.
    full = drain(lambda buf: 1)
    assert full > 9 * 64, f"expected a multi-slot dataset, got only {full} samples"

    # Skip every other sample. A per-call counter is fine here (num_workers=0,
    # single process). If skipped items leaked slots, the epoch would stall after
    # ~9 slots and `kept` would be a small fraction, not ≈ full/2.
    state = {"i": 0}

    def skip_half(buf):
        state["i"] += 1
        return None if state["i"] % 2 == 0 else 1

    kept = drain(skip_half)

    # Survivors should be ≈ half of the full dataset. A leak would strand most
    # slots and collapse this far below half.
    assert 0.4 * full <= kept <= 0.6 * full, (
        f"kept={kept} not ≈ half of full={full} — slots likely leaked on skip"
    )
