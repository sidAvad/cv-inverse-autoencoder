"""Shared provenance logging for all plotting scripts."""

import hashlib
import subprocess
import sys
from datetime import datetime
from pathlib import Path

LOG_PATH = Path(__file__).parent / "provenance.log"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip())
    except Exception:
        return False


def _script_hash() -> str:
    script = Path(sys.argv[0]).resolve()
    return hashlib.sha256(script.read_bytes()).hexdigest()[:10]


def log(output_path: str) -> None:
    cmd        = " ".join(sys.argv)
    commit     = _git_commit()
    dirty      = " (dirty)" if _git_dirty() else ""
    script_sha = _script_hash()
    entry = (
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  "
        f"git={commit}{dirty}  script_sha={script_sha}  "
        f"{cmd}  →  {output_path}\n"
    )
    with open(LOG_PATH, "a") as f:
        f.write(entry)
    print(f"Provenance logged → {LOG_PATH}")
