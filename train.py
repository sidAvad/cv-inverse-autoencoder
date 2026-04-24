"""
Train CVSurrogate on cardiovascular simulation data.

Prerequisites:
    - Run compute_stats.py first to produce norm_stats.json.
    - manifest_train.json must exist one level above DATA_DIR.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
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
    index = manifest["index"][:N_SIMS]
    print(f"Manifest: {len(index):,} entries selected")

    # Sequential split: first N_VAL -> val, remainder -> train.
    # DataLoader shuffle handles randomisation within the train split.
    val_index   = index[:N_VAL]
    train_index = index[N_VAL:]
    print(f"  train: {len(train_index):,}   val: {len(val_index):,}")

    # ── Datasets and loaders ───────────────────────────────────────────
    train_ds = CVDataset(DATA_DIR, train_index, stats)
    val_ds   = CVDataset(DATA_DIR, val_index,   stats)

    loader_kwargs = dict(
        batch_size        = BATCH_SIZE,
        num_workers       = NUM_WORKERS,
        pin_memory        = True,
        persistent_workers= True,
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

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = OneCycleLR(
        optimizer,
        max_lr           = LR,
        steps_per_epoch  = len(train_loader),
        epochs           = N_EPOCHS,
    )

    # ── Training loop ──────────────────────────────────────────────────
    hdr = (f"{'Epoch':<6} {'Train':>10} {'T-cont':>10} {'T-valve':>10}"
           f"  {'Val':>10} {'V-cont':>10} {'V-valve':>10}")
    print(f"\n{hdr}")
    print("-" * len(hdr))

    for epoch in range(1, N_EPOCHS + 1):
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

        torch.save(
            {
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss":        v_total,
            },
            f"checkpoint_epoch{epoch:02d}.pt",
        )

    train_ds.close()
    val_ds.close()


if __name__ == "__main__":
    main()
