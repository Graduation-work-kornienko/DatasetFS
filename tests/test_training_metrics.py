from __future__ import annotations

from benchmarks.datasetfs_bench.metrics.training import EpochStats


def test_epoch_stats_includes_stage_timing_breakdown():
    stats = EpochStats(
        epoch=0,
        n_batches=2,
        n_samples=16,
        wall_seconds=1.0,
        time_to_first_batch=0.1,
        fetch_latency_seconds=[0.2, 0.2],
        compute_seconds=[0.3, 0.3],
        zero_grad_seconds=[0.01, 0.01],
        forward_backward_seconds=[0.25, 0.25],
        optimizer_step_seconds=[0.04, 0.04],
        steady_n_samples=16,
        steady_wall_seconds=1.0,
    )

    summary = stats.summary()

    assert summary["batch_wait_total_s"] == 0.4
    assert summary["compute_total_s"] == 0.6
    assert summary["forward_backward_total_s"] == 0.5
    assert round(summary["batch_wait_fraction"], 6) == 0.4
    assert round(summary["forward_backward_fraction"], 6) == 0.5
    assert round(summary["optimizer_fraction"], 6) == 0.08
