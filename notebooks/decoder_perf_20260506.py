import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Decoder performance — baseline (2026-05-06)

    Per-channel MAE, MAPE comparison (standard vs range-normalised), and waveform inspection for 3 test examples.
    """)
    return


@app.cell
def _():
    import json
    import sys
    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from torch.utils.data import DataLoader

    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))

    from dataset import CVDataset
    from model import CVSurrogate

    return CVDataset, CVSurrogate, DataLoader, Path, ROOT, json, np, plt, torch


@app.cell
def _(Path, ROOT, torch):
    CHECKPOINT  = ROOT / 'checkpoints/baseline/checkpoint_epoch10.pt'
    STATS_PATH  = ROOT / 'norm_stats.json'
    DATA_DIR    = Path('/media/8TBNVME/data/neh10/hdf5/cv8/simset_10M_cv8Eed_20260314/test')
    MANIFEST    = DATA_DIR.parent / 'manifest_test.json'
    N_SIMS      = 256
    SIM_INDICES = [0, 1, 2]

    WAVE_NAMES_CONT = [
        'Pap','Pas','Pla','Plv','Pra','Prv','Pvp','Pvs',
        'Qap','Qas','Qla','Qlv','Qra','Qrv','Qvp','Qvs',
        'Vap','Vas','Vla','Vlv','Vra','Vrv','Vvp','Vvs',
    ]
    VALVE_NAMES = ['AV', 'MV', 'PV', 'TV']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return (
        CHECKPOINT,
        DATA_DIR,
        MANIFEST,
        N_SIMS,
        SIM_INDICES,
        STATS_PATH,
        VALVE_NAMES,
        WAVE_NAMES_CONT,
        device,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load model
    """)
    return


@app.cell
def _(CHECKPOINT, CVSurrogate, MANIFEST, STATS_PATH, device, json, torch):
    ckpt  = torch.load(CHECKPOINT, map_location=device)
    model = CVSurrogate().to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Epoch {ckpt['epoch']}  val_loss={ckpt['val_loss']:.6f}")

    with open(STATS_PATH) as f:
        stats = json.load(f)
    with open(MANIFEST) as f:
        manifest = json.load(f)
    return ckpt, manifest, model, stats


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run inference on test set
    """)
    return


@app.cell
def _(
    CVDataset,
    DATA_DIR,
    DataLoader,
    N_SIMS,
    WAVE_NAMES_CONT,
    device,
    manifest,
    model,
    np,
    stats,
    torch,
):
    ds     = CVDataset(DATA_DIR, manifest['index'][:N_SIMS], stats)
    loader = DataLoader(ds, batch_size=256, num_workers=0)

    all_pred, all_gt = [], []
    with torch.no_grad():
        for params_b, waves_cont_b, _ in loader:
            pred_cont_b, _ = model(params_b.to(device))
            all_pred.append(pred_cont_b.cpu())
            all_gt.append(waves_cont_b)
    ds.close()

    all_pred = torch.cat(all_pred).numpy()   # (N, 24, 201)
    all_gt   = torch.cat(all_gt).numpy()

    wave_means = np.array([stats['waves'][n]['mean'] for n in WAVE_NAMES_CONT])
    wave_stds  = np.array([stats['waves'][n]['std']  for n in WAVE_NAMES_CONT])
    pred_phys  = all_pred * wave_stds[None,:,None] + wave_means[None,:,None]
    gt_phys    = all_gt   * wave_stds[None,:,None] + wave_means[None,:,None]
    print(f'Evaluated {N_SIMS} test simulations')
    return gt_phys, pred_phys


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compute metrics
    """)
    return


@app.cell
def _(WAVE_NAMES_CONT, gt_phys, np, pred_phys):
    abs_err = np.abs(pred_phys - gt_phys)  # (N, 24, 201)
    std_pct_err = abs_err / (np.abs(gt_phys) + 1e-06) * 100
    # Standard MAPE — inflated for Q channels that pass through zero
    sim_ch_mape = std_pct_err.mean(axis=2)
    ch_mape = sim_ch_mape.mean(axis=0)  # (N, 24)
    sig_range = np.maximum(gt_phys.max(axis=2) - gt_phys.min(axis=2), 1e-06)  # (24,)
    rng_pct_err = abs_err / sig_range[:, :, None] * 100
    # Range-normalised error — safe for zero-crossing channels
    sim_ch_rne = rng_pct_err.mean(axis=2)  # (N, 24)
    ch_rne = sim_ch_rne.mean(axis=0)
    sim_ch_mae = abs_err.mean(axis=2)  # (N, 24)
    ch_mae = sim_ch_mae.mean(axis=0)  # (24,)
    print(f'{'Channel':<8} {'MAE':>10} {'MAPE (%)':>10} {'RangeErr (%)':>14}')
    # MAE
    print('-' * 46)  # (N, 24)
    for i, _name in enumerate(WAVE_NAMES_CONT):  # (24,)
        print(f'{_name:<8} {ch_mae[i]:>10.4f} {ch_mape[i]:>10.2f} {ch_rne[i]:>14.2f}')
    return ch_mae, ch_mape, ch_rne, sim_ch_mae


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Per-channel MAE
    """)
    return


@app.cell
def _(N_SIMS, WAVE_NAMES_CONT, ch_mae, ckpt, np, plt, sim_ch_mae):
    x = np.arange(len(WAVE_NAMES_CONT))
    _fig, _axes = plt.subplots(1, 2, figsize=(18, 5))
    _axes[0].boxplot(sim_ch_mae, tick_labels=WAVE_NAMES_CONT, patch_artist=True, boxprops=dict(facecolor='steelblue', alpha=0.6))
    _axes[0].set_title('MAE distribution per channel')
    _axes[0].set_ylabel('MAE (physical units)')
    _axes[0].tick_params(axis='x', rotation=45)
    _axes[1].bar(x, ch_mae, color='steelblue', alpha=0.8)
    _axes[1].set_xticks(x)
    _axes[1].set_xticklabels(WAVE_NAMES_CONT, rotation=45, ha='right')
    _axes[1].set_title('Mean MAE per channel')
    _axes[1].set_ylabel('MAE (physical units)')
    _fig.suptitle(f'Baseline — MAE  (epoch {ckpt['epoch']}, n={N_SIMS})', fontsize=13)
    plt.tight_layout()
    plt.show()
    return (x,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MAPE comparison: standard vs range-normalised

    Standard MAPE divides by `|gt(t)|` at each timestep — blows up for Q channels that cross zero.
    Range-normalised error divides by `max(gt) − min(gt)` per sim per channel — interpretable as fraction of full swing.
    """)
    return


@app.cell
def _(N_SIMS, WAVE_NAMES_CONT, ch_mape, ch_rne, ckpt, plt, x):
    _fig, _axes = plt.subplots(1, 2, figsize=(18, 5))
    _axes[0].bar(x, ch_mape, color='tomato', alpha=0.8)
    _axes[0].set_xticks(x)
    _axes[0].set_xticklabels(WAVE_NAMES_CONT, rotation=45, ha='right')
    _axes[0].set_title('Standard MAPE — inflated for zero-crossing Q/V channels')
    _axes[0].set_ylabel('MAPE (%)')
    _axes[1].bar(x, ch_rne, color='darkorange', alpha=0.8)
    _axes[1].set_xticks(x)
    _axes[1].set_xticklabels(WAVE_NAMES_CONT, rotation=45, ha='right')
    _axes[1].set_title('Range-normalised error — safe for zero-crossing channels [updated]')
    _axes[1].set_ylabel('Range-norm. error (%)')
    _fig.suptitle(f'Baseline — MAPE comparison  (epoch {ckpt['epoch']}, n={N_SIMS})', fontsize=13)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Waveforms — 3 test examples
    """)
    return


@app.cell
def _(
    CVDataset,
    DATA_DIR,
    SIM_INDICES,
    device,
    manifest,
    model,
    np,
    stats,
    torch,
):
    ds3     = CVDataset(DATA_DIR, [manifest['index'][i] for i in SIM_INDICES], stats)
    samples = [ds3[j] for j in range(len(SIM_INDICES))]
    ds3.close()

    pred_conts, pred_valves, gt_conts, gt_valves = [], [], [], []
    for params, waves_cont, waves_valve in samples:
        with torch.no_grad():
            pc, pv = model(params.unsqueeze(0).to(device))
        pred_conts.append(pc.squeeze(0).cpu().numpy())
        pred_valves.append(pv.squeeze(0).cpu().numpy())
        gt_conts.append(waves_cont.numpy())
        gt_valves.append(waves_valve.numpy())

    t = np.arange(201)
    return gt_conts, gt_valves, pred_conts, pred_valves, t


@app.cell
def _(SIM_INDICES, WAVE_NAMES_CONT, ckpt, gt_conts, plt, pred_conts, t):
    # Continuous waveforms: rows = channels, cols = examples
    ncols = len(SIM_INDICES)
    nrows = len(WAVE_NAMES_CONT)
    _fig, _axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 2.5 * nrows), sharex=True)
    for _col, _idx in enumerate(SIM_INDICES):
        for _row, _name in enumerate(WAVE_NAMES_CONT):
            _ax = _axes[_row, _col]
            _ax.plot(t, gt_conts[_col][_row], color='steelblue', label='GT')
            _ax.plot(t, pred_conts[_col][_row], color='tomato', linestyle='--', label='Pred')
            if _col == 0:
                _ax.set_ylabel(_name, fontsize=8)
            if _row == 0:
                _ax.set_title(f'Test sim {_idx}', fontsize=9)
            if _row == nrows - 1:
                _ax.set_xlabel('Time step', fontsize=8)
            if _col == ncols - 1 and _row == 0:
                _ax.legend(loc='upper right', fontsize=7)
    _fig.suptitle(f'Baseline — continuous waveforms (epoch {ckpt['epoch']})', fontsize=12)
    plt.tight_layout()
    plt.show()
    return (ncols,)


@app.cell
def _(SIM_INDICES, VALVE_NAMES, ckpt, gt_valves, ncols, plt, pred_valves, t):
    # Valve signals: rows = valves, cols = examples
    _fig, _axes = plt.subplots(4, ncols, figsize=(6 * ncols, 8), sharex=True)
    for _col, _idx in enumerate(SIM_INDICES):
        for _row, _name in enumerate(VALVE_NAMES):
            _ax = _axes[_row, _col]
            _ax.step(t, gt_valves[_col][_row], color='steelblue', where='post', label='GT')
            _ax.step(t, (pred_valves[_col][_row] > 0).astype(float), color='tomato', where='post', linestyle='--', label='Pred')
            _ax.set_ylim(-0.1, 1.1)
            if _col == 0:
                _ax.set_ylabel(_name, fontsize=8)
            if _row == 0:
                _ax.set_title(f'Test sim {_idx}', fontsize=9)
            if _row == 3:
                _ax.set_xlabel('Time step', fontsize=8)
            if _col == ncols - 1 and _row == 0:
                _ax.legend(loc='upper right', fontsize=7)
    _fig.suptitle(f'Baseline — valve signals (epoch {ckpt['epoch']})', fontsize=12)
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
