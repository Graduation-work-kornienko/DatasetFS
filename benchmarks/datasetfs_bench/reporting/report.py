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
        "format_matrix.png",
        "sweep_throughput.png",
        "sweep_stall.png",
        "mutation_benchmark.png",
        "mutation_endurance.png",
        "mutation_endurance_timeline.png",
        "real_universal.png",
        "remote_streaming.png",
        "system_timeseries.png",
        "daemon_timeseries.png",
        "latency_table.md",
        "summary.csv",
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
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("warmup", "").lower() == "true":
            continue
        grouped[row.get("loader") or row.get("name") or "run"].append(row)
    lines = [
        "| loader/dataset | rows | samples/s mean | samples/s std | TTFB mean, s | stall mean, % | CPU mean, % | daemon RSS max, MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in sorted(grouped):
        vals = grouped[name]
        sps_key = "steady_samples_per_second" if any(r.get("steady_samples_per_second") for r in vals) else "samples_per_second"
        sps = [_f(r, sps_key) for r in vals if r.get(sps_key)]
        ttfb = [_f(r, "time_to_first_batch") for r in vals if r.get("time_to_first_batch")]
        stall = [_f(r, "stall_fraction") * 100.0 for r in vals if r.get("stall_fraction")]
        cpu = [_f(r, "sys_cpu_pct_mean") for r in vals if r.get("sys_cpu_pct_mean")]
        daemon_rss = [_f(r, "sys_daemon_rss_max_bytes") / (1024 * 1024) for r in vals if r.get("sys_daemon_rss_max_bytes")]
        lines.append(
            f"| {name} | {len(vals)} | {_fmt(mean(sps) if sps else 0)} | "
            f"{_fmt(stdev(sps) if len(sps) > 1 else 0)} | {_fmt(mean(ttfb) if ttfb else 0, 3)} | "
            f"{_fmt(mean(stall) if stall else 0, 2)} | {_fmt(mean(cpu) if cpu else 0, 1)} | "
            f"{_fmt(max(daemon_rss) if daemon_rss else 0, 1)} |"
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


def _missing_table(run_dir: Path) -> list[str]:
    path = run_dir / "missing.csv"
    if not path.exists():
        return []
    rows = _read_rows(path)
    if not rows:
        return []
    lines = ["| missing dataset | modality | prepare command |", "|---|---|---|"]
    for row in rows:
        lines.append(f"| {row.get('name', '')} | {row.get('modality', '')} | `{row.get('prepare', '')}` |")
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
