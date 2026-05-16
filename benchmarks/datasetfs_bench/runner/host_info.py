"""Capture host fingerprint per run — critical for cross-host aggregation later.
Without this, comparing numbers from different machines is impossible."""
from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path


def gather() -> dict:
    info: dict = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
        "cpu_count_logical": os.cpu_count(),
        "hostname": platform.node(),
    }

    # Best-effort RAM detection
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            info["ram_bytes"] = int(out)
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        info["ram_bytes"] = int(line.split()[1]) * 1024
                        break
    except Exception:
        pass

    # Best-effort git SHA
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
        info["git_sha"] = sha
    except Exception:
        pass

    return info


def write(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        json.dump(gather(), f, indent=2)
