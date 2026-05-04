"""
Plot per-channel MAE and MAPE for the surrogate model on the test set.

Usage:
    python results/surrogate_perf/plot_per_channel_mae.py
    python results/surrogate_perf/plot_per_channel_mae.py --checkpoint checkpoints/checkpoint_epoch05.pt --n-sims 2048
    python results/surrogate_perf/plot_per_channel_mae.py --run-name epoch05_comparison
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parents[1]))
from dataset import CVDataset
from model import CVSurrogate, CVLoss
from _provenance import log

CHECKPOINT_DIR  = ROOT / "checkpoints"
STATS_PATH      = ROOT / "norm_stats.json"
DATA_DIR_TEST   = Path("/media/8TBNVME/data/neh10/hdf5/cv8/simset_10M_cv8Eed_20260314/test")
MANIFEST_TEST   = DATA_DIR_TEST.parent / "manifest_test.json"
OUT_DIR         = Path(__file__).parent

WAVE_NAMES_CONT = ["Pap","Pas","Pla","Plv","Pra","Prv","Pvp","Pvs",
                   "Qap","Qas","Qla","Qlv","Qra","Qrv","Qvp","Qvs",
                   "Vap","Vas","Vla","Vlv","Vra","Vrv","Vvp","Vvs"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Checkpoint .pt file. Defaults to latest in checkpoints/.")
    p.add_argument("--n-sims", type=int, default=1024,
                   help="Number of test sims to evaluate (default: 1024).")
    p.add_argument("--run-name", type=str, default=None,
                   help="If set, saves as per_channel_mae_<run-name>.svg instead of overwriting default.")
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
    print(f"Epoch {ckpt['epoch']}  val_loss={ckpt['val_loss']:.6f}")

    with open(STATS_PATH) as f:
        stats = json.load(f)
    with open(MANIFEST_TEST) as f:
        test_manifest = json.load(f)

    test_index  = test_manifest["index"][:args.n_sims]
    test_ds     = CVDataset(DATA_DIR_TEST, test_index, stats)
    test_loader = DataLoader(test_ds, batch_size=256, num_workers=0)

    all_pred, all_gt = [], []
    with torch.no_grad():
        for params_b, waves_cont_b, waves_valve_b in test_loader:
            pred_cont_b, _ = model(params_b.to(device))
            all_pred.append(pred_cont_b.cpu())
            all_gt.append(waves_cont_b)

    test_ds.close()

    all_pred = torch.cat(all_pred).numpy()  # (N, 24, 201)
    all_gt   = torch.cat(all_gt).numpy()

    wave_means = np.array([stats["waves"][n]["mean"] for n in WAVE_NAMES_CONT])
    wave_stds  = np.array([stats["waves"][n]["std"]  for n in WAVE_NAMES_CONT])
    pred_phys  = all_pred * wave_stds[None,:,None] + wave_means[None,:,None]
    gt_phys    = all_gt   * wave_stds[None,:,None] + wave_means[None,:,None]

    abs_err     = np.abs(pred_phys - gt_phys)
    pct_err     = abs_err / (np.abs(gt_phys) + 1e-6) * 100
    sim_ch_mae  = abs_err.mean(axis=2)   # (N, 24)
    sim_ch_mape = pct_err.mean(axis=2)
    ch_mae      = sim_ch_mae.mean(axis=0)
    ch_mape     = sim_ch_mape.mean(axis=0)

    print(f"\n{'Channel':<8} {'MAE':>10} {'MAPE (%)':>10}")
    print("-" * 32)
    for i, name in enumerate(WAVE_NAMES_CONT):
        print(f"{name:<8} {ch_mae[i]:>10.4f} {ch_mape[i]:>10.2f}")

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    x = np.arange(len(WAVE_NAMES_CONT))

    axes[0, 0].boxplot(sim_ch_mae, tick_labels=WAVE_NAMES_CONT, patch_artist=True,
                       boxprops=dict(facecolor="steelblue", alpha=0.6))
    axes[0, 0].set_title("MAE distribution per channel")
    axes[0, 0].set_ylabel("MAE (physical units)")
    axes[0, 0].tick_params(axis="x", rotation=45)

    axes[0, 1].boxplot(sim_ch_mape, tick_labels=WAVE_NAMES_CONT, patch_artist=True,
                       boxprops=dict(facecolor="tomato", alpha=0.6))
    axes[0, 1].set_title("MAPE distribution per channel")
    axes[0, 1].set_ylabel("MAPE (%)")
    axes[0, 1].tick_params(axis="x", rotation=45)

    axes[1, 0].bar(x, ch_mae, color="steelblue", alpha=0.8)
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(WAVE_NAMES_CONT, rotation=45, ha="right")
    axes[1, 0].set_title("Mean MAE per channel"); axes[1, 0].set_ylabel("MAE (physical units)")

    axes[1, 1].bar(x, ch_mape, color="tomato", alpha=0.8)
    axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(WAVE_NAMES_CONT, rotation=45, ha="right")
    axes[1, 1].set_title("Mean MAPE per channel"); axes[1, 1].set_ylabel("MAPE (%)")

    title = f"Surrogate performance — epoch {ckpt['epoch']}, n={args.n_sims}"
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()

    stem = f"per_channel_mae_{args.run_name}" if args.run_name else "per_channel_mae"
    out  = OUT_DIR / f"{stem}.svg"
    fig.savefig(out, format="svg")
    print(f"Saved {out}")
    log(str(out))


if __name__ == "__main__":
    main()
