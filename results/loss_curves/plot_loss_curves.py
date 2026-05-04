"""
Plot train/val loss curves from a training log file.

Usage:
    python results/loss_curves/plot_loss_curves.py
    python results/loss_curves/plot_loss_curves.py --log-file logs/train_20260424_130543.log
    python results/loss_curves/plot_loss_curves.py --run-name epoch05_comparison
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parents[1]))
from _provenance import log

LOG_DIR  = Path(__file__).parents[2] / "logs"
OUT_DIR  = Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log-file", type=Path, default=None,
                   help="Path to training log. Defaults to most recent in logs/.")
    p.add_argument("--run-name", type=str, default=None,
                   help="If set, saves as loss_curves_<run-name>.svg instead of overwriting default.")
    return p.parse_args()


def main():
    args = parse_args()

    log_file = args.log_file or sorted(LOG_DIR.glob("train_*.log"))[-1]
    print(f"Parsing {log_file}")

    epochs, train_loss, val_loss = [], [], []
    for line in log_file.read_text().splitlines():
        m = re.match(r"^(\d+)\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+([\d.]+)", line)
        if m:
            epochs.append(int(m.group(1)))
            train_loss.append(float(m.group(2)))
            val_loss.append(float(m.group(3)))
    print(f"Epochs parsed: {len(epochs)}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_loss, label="Train")
    ax.plot(epochs, val_loss,   label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Total loss")
    ax.legend()
    plt.tight_layout()

    stem = f"loss_curves_{args.run_name}" if args.run_name else "loss_curves"
    out  = OUT_DIR / f"{stem}.svg"
    fig.savefig(out, format="svg")
    print(f"Saved {out}")
    log(str(out))


if __name__ == "__main__":
    main()
