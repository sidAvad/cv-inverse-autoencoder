"""
Plot GT vs predicted waveforms for a single simulation.

Usage:
    python results/surrogate_perf/plot_waveforms.py
    python results/surrogate_perf/plot_waveforms.py --checkpoint checkpoints/checkpoint_epoch05.pt --sim-idx 42 --split test
    python results/surrogate_perf/plot_waveforms.py --run-name epoch05_sim42
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parents[1]))
from dataset import CVDataset
from model import CVSurrogate
from _provenance import log

CHECKPOINT_DIR = ROOT / "checkpoints"
STATS_PATH     = ROOT / "norm_stats.json"
DATA_DIR_TRAIN = Path("/media/8TBNVME/data/neh10/hdf5/cv8/simset_10M_cv8Eed_20260314/train")
DATA_DIR_TEST  = Path("/media/8TBNVME/data/neh10/hdf5/cv8/simset_10M_cv8Eed_20260314/test")
OUT_DIR        = Path(__file__).parent

WAVE_NAMES_CONT = ["Pap","Pas","Pla","Plv","Pra","Prv","Pvp","Pvs",
                   "Qap","Qas","Qla","Qlv","Qra","Qrv","Qvp","Qvs",
                   "Vap","Vas","Vla","Vlv","Vra","Vrv","Vvp","Vvs"]
VALVE_NAMES     = ["AV", "MV", "PV", "TV"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Checkpoint .pt file. Defaults to latest in checkpoints/.")
    p.add_argument("--sim-idx", type=int, default=0,
                   help="Index of simulation to plot (default: 0).")
    p.add_argument("--split", choices=["train", "test"], default="test",
                   help="Dataset split to draw from (default: test).")
    p.add_argument("--run-name", type=str, default=None,
                   help="If set, appends to output filenames instead of overwriting defaults.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.checkpoint or sorted(CHECKPOINT_DIR.glob("checkpoint_epoch*.pt"))[-1]
    print(f"Loading {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = CVSurrogate().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with open(STATS_PATH) as f:
        stats = json.load(f)

    data_dir = DATA_DIR_TEST if args.split == "test" else DATA_DIR_TRAIN
    manifest_path = data_dir.parent / f"manifest_{args.split}.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    idx     = args.sim_idx
    ds      = CVDataset(data_dir, [manifest["index"][idx]], stats)
    params, waves_cont, waves_valve = ds[0]
    ds.close()

    with torch.no_grad():
        pred_cont, pred_valve = model(params.unsqueeze(0).to(device))

    pred_cont  = pred_cont.squeeze(0).cpu().numpy()
    pred_valve = pred_valve.squeeze(0).cpu().numpy()
    gt_cont    = waves_cont.numpy()
    gt_valve   = waves_valve.numpy()
    t          = np.arange(201)

    suffix = f"_{args.run_name}" if args.run_name else ""

    # Continuous waveforms
    fig, axes = plt.subplots(len(WAVE_NAMES_CONT), 1, figsize=(10, 2.5 * len(WAVE_NAMES_CONT)), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, gt_cont[i],   label="GT",   color="steelblue")
        ax.plot(t, pred_cont[i], label="Pred", color="tomato", linestyle="--")
        ax.set_ylabel(WAVE_NAMES_CONT[i])
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time step")
    fig.suptitle(f"{args.split.capitalize()} sim {idx} — continuous waveforms (epoch {ckpt['epoch']})")
    plt.tight_layout()
    out_cont = OUT_DIR / f"waveforms_cont{suffix}.svg"
    fig.savefig(out_cont, format="svg")
    print(f"Saved {out_cont}")
    log(str(out_cont))

    # Valve signals
    fig, axes = plt.subplots(4, 1, figsize=(10, 6), sharex=True)
    for i, (ax, name) in enumerate(zip(axes, VALVE_NAMES)):
        ax.step(t, gt_valve[i],                       label="GT",   color="steelblue", where="post")
        ax.step(t, (pred_valve[i] > 0).astype(float), label="Pred", color="tomato",    where="post", linestyle="--")
        ax.set_ylabel(name)
        ax.set_ylim(-0.1, 1.1)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time step")
    fig.suptitle(f"{args.split.capitalize()} sim {idx} — valve signals (epoch {ckpt['epoch']})")
    plt.tight_layout()
    out_valve = OUT_DIR / f"waveforms_valve{suffix}.svg"
    fig.savefig(out_valve, format="svg")
    print(f"Saved {out_valve}")
    log(str(out_valve))


if __name__ == "__main__":
    main()
