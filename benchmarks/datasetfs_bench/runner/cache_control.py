"""Drop OS page cache before a cold-cache benchmark cell.

macOS: `sudo purge` (requires sudo; we use `-n` for non-interactive)
Linux: write 3 to /proc/sys/vm/drop_caches (also requires root)

If sudo isn't configured passwordlessly, this fails gracefully — we log a
warning and let the run continue (the result will be labeled `cache_state=warm`
so plots can filter accordingly).

For repeatable benchmarks, set up passwordless sudo for these specific
commands once (e.g., via /etc/sudoers.d/datasetfs-bench).
"""
from __future__ import annotations

import shutil
import subprocess
import sys


def can_drop_caches() -> bool:
    """Quick check whether we'll be able to drop caches without prompting."""
    if sys.platform == "darwin":
        return shutil.which("purge") is not None and _sudo_ok(["purge"])
    return _sudo_ok(["sh", "-c", "true"])


def _sudo_ok(cmd: list[str]) -> bool:
    """Return True iff `sudo -n` can execute the command without a password
    prompt. We probe with a harmless dry-run."""
    probe = ["sudo", "-n"] + cmd
    try:
        r = subprocess.run(probe, capture_output=True, timeout=2)
        return r.returncode == 0
    except Exception:
        return False


def drop_page_cache() -> bool:
    """Best-effort page cache drop. Returns True on success, False otherwise."""
    if sys.platform == "darwin":
        cmd = ["sudo", "-n", "purge"]
    else:
        cmd = ["sudo", "-n", "sh", "-c", "sync && echo 3 > /proc/sys/vm/drop_caches"]
    try:
        r = subprocess.run(cmd, check=False, capture_output=True, timeout=60)
        if r.returncode != 0:
            print(
                f"[cache] could not drop page cache "
                f"(exit={r.returncode}): {r.stderr.decode().strip()[:200]}",
                flush=True,
            )
            return False
        print("[cache] dropped page cache", flush=True)
        return True
    except Exception as e:
        print(f"[cache] page cache drop failed: {e}", flush=True)
        return False
