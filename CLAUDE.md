# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A PyTorch MLP surrogate that replaces a cardiovascular simulation. Given 25 scalar physiological parameters, it predicts 28 waveforms (24 continuous + 4 binary valve signals), each 201 time steps long.

## Workflow

**Step 1 — compute normalisation stats (once per dataset):**
```bash
python compute_stats.py
```
Reads `1.h5` from `DATA_DIR`, outputs `norm_stats.json` 

**Step 2 — train:**
```bash
python train.py
```
Saves `checkpoint_epochNN.pt` after each epoch.

## Data layout

- HDF5 files are at `DATA_DIR = /media/8TBNVME/data/neh10/hdf5/cv8/simset_10M_cv8Eed_20260314/train/`
- `manifest_train.json` lives one level above `DATA_DIR`; its `"index"` list has entries `{"id", "file", "group"}` pointing into numbered HDF5 files
- Each HDF5 group (`sim_NNNNNN`) holds `parameters/<key>` scalars and `waves/<key>` arrays of length 201

## Architecture

**`CVSurrogate`** (`model.py`):
- Trunk: 6 × Linear(1024) + SiLU
- Continuous head: Linear → reshape `(B, 24, 201)` — MSE loss
- Valve head: Linear → reshape `(B, 4, 201)` raw logits — BCEWithLogitsLoss

**`CVLoss`**: returns `(total, loss_cont, loss_valve)` with equal weighting.

**`CVDataset`** (`dataset.py`): lazy-opens HDF5 handles per worker process (safe for `fork`-based `DataLoader` multiprocessing). Parameters are z-scored; continuous waveforms are z-scored per channel across sims × time; valve signals are raw `{0, 1}`.

## Git conventions

- Never add `Co-Authored-By: Claude` or any AI authorship trailer to commit messages.

## Key constants

| Symbol | Value | Meaning |
|---|---|---|
| `N_PARAMS` | 25 | Input dimension |
| `N_WAVES_CONT` | 24 | Continuous output channels |
| `N_WAVES_VALVE` | 4 | Binary valve channels (av, mv, pv, tv) |
| `T` | 201 | Time steps per waveform |
| `HIDDEN` | 1024 | MLP hidden size |
| `N_LAYERS` | 6 | MLP depth |

