"""Shared provenance logging for all plotting scripts."""

import sys
from datetime import datetime
from pathlib import Path

LOG_PATH = Path(__file__).parent / "provenance.log"


def log(output_path: str) -> None:
    cmd = " ".join(sys.argv)
    entry = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {cmd}  →  {output_path}\n"
    with open(LOG_PATH, "a") as f:
        f.write(entry)
    print(f"Provenance logged → {LOG_PATH}")
