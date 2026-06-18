"""Generate a compact Markdown report for one benchmark run directory."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def _read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, "") or default)
    except ValueError:
        return default


def _fmt(value: float, digits: int = 2) -> str:
    if value != value:
        return ""
    return f"{value:.{digits}f}"


def _existing_artifacts(run_dir: Path) -> list[str]:
    names = [
        "throughput_bar.png",
        "training_stage_breakdown.png",
        "wait_compute_breakdown.png",
        "format_matrix.png",
        "sweep_throughput.png",
        "sweep_stall.png",
        "mutation_benchmark.png",
        "mutation_format_compare.png",
        "mutation_endurance.png",
        "mutation_endurance_timeline.png",
        "pipeline_memory.png",
        "real_universal.png",
        "remote_streaming.png",
        "system_timeseries.png",
        "daemon_timeseries.png",
        "latency_table.md",
        "summary.csv",
        "memory_timeseries.csv",
        "sweep_summary.csv",
        "daemon_timeseries.csv",
        "system_timeseries.csv",
        "missing.csv",
    ]
    return [name for name in names if (run_dir / name).exists()]


def _host_info(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "host_info.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _single_run_table(rows: list[dict[str, str]]) -> list[str]:
    if rows and "mode" in rows[0] and "mutation_rate_s" in rows[0]:
        return _mutation_table(rows)
    if rows and rows[0].get("scenario") == "format_mutation":
        return _format_mutation_table(rows)
    if rows and rows[0].get("scenario") == "pipeline_memory":
        return _pipeline_memory_table(rows)
    if rows and "vacuum_scenario" in rows[0]:
        return _vacuum_matrix_table(rows)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("warmup", "").lower() == "true":
            continue
        name = row.get("loader") or row.get("name") or "run"
        if row.get("format"):
            name = f"{name}/{row['format']}"
        grouped[name].append(row)
    lines = [
        "| loader/dataset | rows | samples/s mean | samples/s std | TTFB mean, s | steady wait mean, % | fwd/bwd mean, % | optimizer mean, % | CPU mean, % | daemon RSS max, MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in sorted(grouped):
        vals = grouped[name]
        sps_key = "steady_samples_per_second" if any(r.get("steady_samples_per_second") for r in vals) else "samples_per_second"
        sps = [_f(r, sps_key) for r in vals if r.get(sps_key)]
        ttfb = [_f(r, "time_to_first_batch") for r in vals if r.get("time_to_first_batch")]
        stall_key = "steady_batch_wait_fraction" if any(r.get("steady_batch_wait_fraction") for r in vals) else "batch_wait_fraction"
        stall = [_f(r, stall_key, _f(r, "stall_fraction")) * 100.0 for r in vals if r.get(stall_key) or r.get("stall_fraction")]
        fwd_bwd = [_f(r, "forward_backward_fraction") * 100.0 for r in vals if r.get("forward_backward_fraction")]
        opt = [_f(r, "optimizer_fraction") * 100.0 for r in vals if r.get("optimizer_fraction")]
        cpu = [_f(r, "sys_cpu_pct_mean") for r in vals if r.get("sys_cpu_pct_mean")]
        daemon_rss = [_f(r, "sys_daemon_rss_max_bytes") / (1024 * 1024) for r in vals if r.get("sys_daemon_rss_max_bytes")]
        lines.append(
            f"| {name} | {len(vals)} | {_fmt(mean(sps) if sps else 0)} | "
            f"{_fmt(stdev(sps) if len(sps) > 1 else 0)} | {_fmt(mean(ttfb) if ttfb else 0, 3)} | "
            f"{_fmt(mean(stall) if stall else 0, 2)} | {_fmt(mean(fwd_bwd) if fwd_bwd else 0, 2)} | "
            f"{_fmt(mean(opt) if opt else 0, 2)} | {_fmt(mean(cpu) if cpu else 0, 1)} | "
            f"{_fmt(max(daemon_rss) if daemon_rss else 0, 1)} |"
        )
    return lines


def _vacuum_matrix_table(rows: list[dict[str, str]]) -> list[str]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("vacuum_scenario", "scenario")].append(row)
    lines = [
        "| scenario | WAL | auto-vacuum | rows | samples/s mean | CPU mean, % | RSS max, MiB | disk free min, GiB | disk used delta, MiB | disk write, MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario, vals in sorted(grouped.items()):
        sps = [_f(r, "samples_per_second") for r in vals if r.get("samples_per_second")]
        cpu = [_f(r, "cpu_pct_mean") for r in vals if r.get("cpu_pct_mean")]
        rss = [_f(r, "tracked_rss_max_bytes") / (1024 * 1024) for r in vals if r.get("tracked_rss_max_bytes")]
        free = [_f(r, "disk_free_min_bytes") / (1024 ** 3) for r in vals if r.get("disk_free_min_bytes")]
        used_delta = [_f(r, "disk_used_delta_bytes") / (1024 * 1024) for r in vals if r.get("disk_used_delta_bytes")]
        writes = [_f(r, "disk_write_bytes") / (1024 * 1024) for r in vals if r.get("disk_write_bytes")]
        lines.append(
            f"| {scenario} | {vals[0].get('wal_format', '')} | {vals[0].get('auto_vacuum', '')} | {len(vals)} | "
            f"{_fmt(mean(sps) if sps else 0)} | {_fmt(mean(cpu) if cpu else 0, 1)} | "
            f"{_fmt(max(rss) if rss else 0, 1)} | {_fmt(min(free) if free else 0, 2)} | "
            f"{_fmt(mean(used_delta) if used_delta else 0, 1)} | {_fmt(mean(writes) if writes else 0, 1)} |"
        )
    return lines


def _format_mutation_table(rows: list[dict[str, str]]) -> list[str]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("format", ""), row.get("changed_files", ""))].append(row)
    lines = [
        "| format | changed files | rows | mean op, ms | elapsed, s | failed ops | bytes written, MiB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (fmt, changed), vals in sorted(grouped.items(), key=lambda item: (item[0][0], float(item[0][1] or 0))):
        op_ms = [_f(r, "mean_operation_ms") for r in vals if r.get("mean_operation_ms")]
        elapsed = [_f(r, "elapsed_s") for r in vals if r.get("elapsed_s")]
        failed = sum(_f(r, "operations_failed") for r in vals)
        written = sum(_f(r, "bytes_written") for r in vals) / (1024 * 1024)
        lines.append(
            f"| {fmt} | {changed} | {len(vals)} | {_fmt(mean(op_ms) if op_ms else 0, 3)} | "
            f"{_fmt(mean(elapsed) if elapsed else 0, 3)} | {_fmt(failed, 0)} | {_fmt(written, 2)} |"
        )
    return lines


def _pipeline_memory_table(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "| mode | cycles | warmup | files | replacements/cycle | RSS min, MiB | RSS max, MiB | RSS growth, MiB | RSS slope, MiB/cycle | drain mean, s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: r.get("mode", "")):
        lines.append(
            f"| {row.get('mode', '')} | {_fmt(_f(row, 'cycles'), 0)} | {_fmt(_f(row, 'warmup_cycles'), 0)} | "
            f"{_fmt(_f(row, 'files'), 0)} | {_fmt(_f(row, 'replacements_per_cycle'), 0)} | "
            f"{_fmt(_f(row, 'rss_min_mib'), 2)} | {_fmt(_f(row, 'rss_max_mib'), 2)} | "
            f"{_fmt(_f(row, 'rss_growth_mib'), 2)} | {_fmt(_f(row, 'rss_slope_mib_per_cycle'), 4)} | "
            f"{_fmt(_f(row, 'drain_elapsed_mean_s'), 4)} |"
        )
    return lines


def _mutation_table(rows: list[dict[str, str]]) -> list[str]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("mode", ""), row.get("mutation_rate_s", ""))].append(row)
    lines = [
        "| mode | mutations/s | rows | samples/s mean | violations | mutations ok/fail | latency mean, ms | CPU mean, % | tracked RSS max, MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (mode, rate), vals in sorted(grouped.items(), key=lambda item: (item[0][0], float(item[0][1] or 0))):
        sps = [_f(r, "samples_per_s") for r in vals if r.get("samples_per_s")]
        violations = sum(_f(r, "consistency_violations") for r in vals)
        succeeded = sum(_f(r, "mutations_succeeded") for r in vals)
        failed = sum(_f(r, "mutations_failed") for r in vals)
        latency = [_f(r, "mutation_latency_mean_ms") for r in vals if r.get("mutation_latency_mean_ms")]
        cpu = [_f(r, "cpu_pct_mean") for r in vals if r.get("cpu_pct_mean")]
        rss = [_f(r, "tracked_rss_max_bytes") / (1024 * 1024) for r in vals if r.get("tracked_rss_max_bytes")]
        lines.append(
            f"| {mode} | {rate} | {len(vals)} | {_fmt(mean(sps) if sps else 0)} | "
            f"{_fmt(violations, 0)} | {_fmt(succeeded, 0)}/{_fmt(failed, 0)} | "
            f"{_fmt(mean(latency) if latency else 0, 2)} | {_fmt(mean(cpu) if cpu else 0, 1)} | "
            f"{_fmt(max(rss) if rss else 0, 1)} |"
        )
    return lines


def _sweep_table(rows: list[dict[str, str]]) -> list[str]:
    axis_cols = [k for k in rows[0] if k.startswith("axis_")] if rows else []
    axis_col = axis_cols[0] if axis_cols else "axis"
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("warmup", "").lower() == "true":
            continue
        loader = row.get("loader", "loader")
        axis_val = row.get(axis_col, "")
        metric = row.get("steady_samples_per_second") or row.get("samples_per_second")
        if metric:
            grouped[(loader, axis_val)].append(float(metric))
    lines = [
        f"| loader | {axis_col.removeprefix('axis_')} | samples/s mean | samples/s std |",
        "|---|---:|---:|---:|",
    ]
    for (loader, axis_val), vals in sorted(grouped.items()):
        lines.append(f"| {loader} | {axis_val} | {_fmt(mean(vals))} | {_fmt(stdev(vals) if len(vals) > 1 else 0)} |")
    return lines


def _daemon_pipeline_stage_table(rows: list[dict[str, str]]) -> list[str]:
    stage_keys = [
        ("storage_read", "storage read"),
        ("metadata_build", "metadata build"),
        ("decode", "decode"),
        ("shm_write", "SHM write"),
        ("frame_encode", "frame encode"),
        ("pipe_write", "pipe write"),
    ]
    if not any(f"daemon_{key}_latency_p50" in row for row in rows for key, _ in stage_keys):
        return []
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("warmup", "").lower() == "true":
            continue
        name = row.get("loader") or row.get("name") or row.get("mode") or "run"
        if row.get("format"):
            name = f"{name}/{row['format']}"
        grouped[name].append(row)
    lines = [
        "| loader/dataset | stage | p50 mean, ms | p95 mean, ms | samples |",
        "|---|---|---:|---:|---:|",
    ]
    for name in sorted(grouped):
        vals = grouped[name]
        for key, label in stage_keys:
            p50 = [_f(r, f"daemon_{key}_latency_p50") * 1000.0 for r in vals if r.get(f"daemon_{key}_latency_p50")]
            p95 = [_f(r, f"daemon_{key}_latency_p95") * 1000.0 for r in vals if r.get(f"daemon_{key}_latency_p95")]
            count = sum(_f(r, f"daemon_{key}_latency_count") for r in vals if r.get(f"daemon_{key}_latency_count"))
            if p50 or p95 or count:
                lines.append(f"| {name} | {label} | {_fmt(mean(p50) if p50 else 0, 3)} | {_fmt(mean(p95) if p95 else 0, 3)} | {_fmt(count, 0)} |")
    return lines if len(lines) > 2 else []


def _missing_table(run_dir: Path) -> list[str]:
    path = run_dir / "missing.csv"
    if not path.exists():
        return []
    rows = _read_rows(path)
    if not rows:
        return []
    lines = ["| missing dataset | format | modality | prepare command |", "|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row.get('name', '')} | {row.get('format', '')} | {row.get('modality', '')} | `{row.get('prepare', '')}` |")
    return lines


def generate_report(run_dir: Path, out_path: Path | None = None) -> Path:
    out = out_path or (run_dir / "REPORT.md")
    lines: list[str] = [f"# Benchmark Report: {run_dir.name}", ""]

    host = _host_info(run_dir)
    if host:
        lines.extend(["## Host", ""])
        for key in ("platform", "python", "cpu", "cpu_count", "memory_total_bytes"):
            if key in host:
                lines.append(f"- `{key}`: `{host[key]}`")
        lines.append("")

    summary_path = run_dir / "summary.csv"
    sweep_path = run_dir / "sweep_summary.csv"
    if summary_path.exists():
        rows = _read_rows(summary_path)
        if rows:
            lines.extend(["## Summary", ""])
            lines.extend(_single_run_table(rows))
            lines.append("")
            daemon_pipeline = _daemon_pipeline_stage_table(rows)
            if daemon_pipeline:
                lines.extend(["## Daemon Pipeline Stages", ""])
                lines.extend(daemon_pipeline)
                lines.append("")
    if sweep_path.exists():
        rows = _read_rows(sweep_path)
        if rows:
            lines.extend(["## Sweep Summary", ""])
            lines.extend(_sweep_table(rows))
            lines.append("")

    missing = _missing_table(run_dir)
    if missing:
        lines.extend(["## Missing Datasets", ""])
        lines.extend(missing)
        lines.append("")

    artifacts = _existing_artifacts(run_dir)
    if artifacts:
        lines.extend(["## Artifacts", ""])
        for name in artifacts:
            lines.append(f"- [{name}]({name})")
        lines.append("")

    lines.extend(["## Notes", "", "- Warmup rows are excluded from aggregate tables.", "- Report is generated from local CSV/PNG artifacts only.", ""])
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] wrote {out}", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    generate_report(args.run_dir, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
