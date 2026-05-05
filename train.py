"""
Train CVSurrogate on cardiovascular simulation data.

Prerequisites:
    - Run compute_stats.py first to produce norm_stats.json.
    - manifest_train.json must exist one level above DATA_DIR.

Dry-run (smoke-test, finishes in seconds):
    python train.py --dry-run
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset import CVDataset
from model import CVSurrogate, CVLoss

# ─── Config ──────────────────────────────────────────────────────────────────

DATA_DIR      = Path("/media/8TBNVME/data/neh10/hdf5/cv8/simset_10M_cv8Eed_20260314/train")
MANIFEST_PATH = DATA_DIR.parent / "manifest_train.json"
STATS_PATH    = Path("norm_stats.json")

N_SIMS       = 1_000_000  # entries drawn from manifest
N_VAL        = 100_000    # first N_VAL entries -> validation
N_TRAIN      = N_SIMS - N_VAL

BATCH_SIZE   = 512
LR           = 3e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP    = 1.0
N_EPOCHS     = 10
NUM_WORKERS  = 16
DRY_RUN_SIMS = 128


# ─── Logging ─────────────────────────────────────────────────────────────────

class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
    def flush(self):
        for s in self._streams:
            s.flush()

LOG_DIR        = Path("logs")
CHECKPOINT_DIR = Path("checkpoints")

def setup_logging(dry_run: bool) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    tag = "dry_run" if dry_run else datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"train_{tag}.log"
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)
    print(f"Logging to {log_path}")


# ─── Epoch runner ────────────────────────────────────────────────────────────

def run_epoch(model, loader, loss_fn, device, optimizer=None, scheduler=None):
    """Run one full pass over loader.

    Pass optimizer + scheduler for a training epoch; omit both for validation.
    Returns (mean_total, mean_cont, mean_valve) averaged over batches.
    """
    training = optimizer is not None
    model.train(training)

    total_acc = cont_acc = valve_acc = 0.0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for params, waves_cont, waves_valve in loader:
            params      = params.to(device, non_blocking=True)
            waves_cont  = waves_cont.to(device, non_blocking=True)
            waves_valve = waves_valve.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            pred_cont, pred_valve = model(params)
            loss, l_cont, l_valve = loss_fn(
                pred_cont, pred_valve, waves_cont, waves_valve
            )

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()

            total_acc += loss.item()
            cont_acc  += l_cont.item()
            valve_acc += l_valve.item()

    n = len(loader)
    return total_acc / n, cont_acc / n, valve_acc / n


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Smoke-test: tiny dataset, 1 epoch, 0 workers")
    parser.add_argument("--resume", type=Path, default=None, metavar="CHECKPOINT",
                        help="Resume training from this checkpoint .pt file.")
    parser.add_argument("--n-epochs", type=int, default=None,
                        help="Total epochs to train (default: N_EPOCHS). "
                             "When resuming, counts from epoch 1 so set this to "
                             "the desired final epoch number.")
    args = parser.parse_args()
    setup_logging(args.dry_run)

    # ── Prerequisites ──────────────────────────────────────────────────
    if not STATS_PATH.exists():
        raise FileNotFoundError(
            f"Run compute_stats.py first — {STATS_PATH} not found."
        )

    # ── Load normalisation stats ───────────────────────────────────────
    with open(STATS_PATH) as f:
        stats = json.load(f)
    print(f"Loaded stats from {STATS_PATH}")

    # ── Load manifest and slice first N_SIMS entries ───────────────────
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    n_sims = DRY_RUN_SIMS if args.dry_run else N_SIMS
    index  = manifest["index"][:n_sims]
    print(f"Manifest: {len(index):,} entries selected")

    # Sequential split: first N_VAL -> val, remainder -> train.
    # DataLoader shuffle handles randomisation within the train split.
    n_val       = len(index) // 2 if args.dry_run else N_VAL
    val_index   = index[:n_val]
    train_index = index[n_val:]
    print(f"  train: {len(train_index):,}   val: {len(val_index):,}")

    # ── Datasets and loaders ───────────────────────────────────────────
    train_ds = CVDataset(DATA_DIR, train_index, stats)
    val_ds   = CVDataset(DATA_DIR, val_index,   stats)

    loader_kwargs = dict(
        batch_size         = BATCH_SIZE,
        num_workers        = 0 if args.dry_run else NUM_WORKERS,
        pin_memory         = True,
        persistent_workers = False if args.dry_run else True,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    # ── Model, loss, optimiser, scheduler ─────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model   = CVSurrogate().to(device)
    loss_fn = CVLoss().to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    n_epochs    = 1 if args.dry_run else (args.n_epochs or N_EPOCHS)
    start_epoch = 1
    optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler   = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=LR / 100)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {args.resume} (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f})")

    # ── Training loop ──────────────────────────────────────────────────
    hdr = (f"{'Epoch':<6} {'Train':>10} {'T-cont':>10} {'T-valve':>10}"
           f"  {'Val':>10} {'V-cont':>10} {'V-valve':>10}")
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for epoch in range(start_epoch, start_epoch + n_epochs):
        t_total, t_cont, t_valve = run_epoch(
            model, train_loader, loss_fn, device, optimizer, scheduler
        )
        v_total, v_cont, v_valve = run_epoch(
            model, val_loader, loss_fn, device
        )

        print(
            f"{epoch:<6} {t_total:>10.6f} {t_cont:>10.6f} {t_valve:>10.6f}"
            f"  {v_total:>10.6f} {v_cont:>10.6f} {v_valve:>10.6f}"
        )

        CHECKPOINT_DIR.mkdir(exist_ok=True)
        torch.save(
            {
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss":        v_total,
            },
            CHECKPOINT_DIR / f"checkpoint_epoch{epoch:03d}.pt",
        )

    train_ds.close()
    val_ds.close()


if __name__ == "__main__":
    main()
