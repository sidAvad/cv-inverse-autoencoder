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
    # Autoencoder performance — baseline (2026-05-07)

    Waveforms → encoder → predicted params → frozen decoder → reconstructed waveforms.

    Per-channel MAE, Median AE, MAPE, and range-normalised error; waveform inspection for 3 test examples.
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
    from encoder import CVEncoder

    return (
        CVDataset,
        CVEncoder,
        CVSurrogate,
        DataLoader,
        Path,
        ROOT,
        json,
        np,
        plt,
        torch,
    )


@app.cell
def _(Path, ROOT, torch):
    ENCODER_CKPT = ROOT / 'checkpoints/autoenc-baseline/checkpoint_017.pt'
    STATS_PATH   = ROOT / 'norm_stats.json'
    DATA_DIR     = Path('/media/8TBNVME/data/neh10/hdf5/cv8/simset_10M_cv8Eed_20260314/test')
    MANIFEST     = DATA_DIR.parent / 'manifest_test.json'
    N_SIMS       = 256
    SIM_INDICES  = [0, 1, 2]

    WAVE_NAMES_CONT = [
        'Pap','Pas','Pla','Plv','Pra','Prv','Pvp','Pvs',
        'Qap','Qas','Qla','Qlv','Qra','Qrv','Qvp','Qvs',
        'Vap','Vas','Vla','Vlv','Vra','Vrv','Vvp','Vvs',
    ]
    VALVE_NAMES = ['AV', 'MV', 'PV', 'TV']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return (
        DATA_DIR,
        ENCODER_CKPT,
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
    ## Load models
    """)
    return


@app.cell
def _(
    CVEncoder,
    CVSurrogate,
    ENCODER_CKPT,
    MANIFEST,
    ROOT,
    STATS_PATH,
    device,
    json,
    torch,
):
    enc_ckpt = torch.load(ENCODER_CKPT, map_location=device)

    encoder = CVEncoder().to(device)
    encoder.load_state_dict(enc_ckpt['model_state'])
    encoder.eval()

    decoder_path = ROOT / enc_ckpt['decoder_ckpt']
    dec_ckpt = torch.load(decoder_path, map_location=device)
    decoder = CVSurrogate().to(device)
    decoder.load_state_dict(dec_ckpt['model_state'])
    decoder.eval()

    print(f"Encoder  epoch {enc_ckpt['epoch']}  val_loss={enc_ckpt['val_loss']:.6f}")
    print(f"Decoder  epoch {dec_ckpt['epoch']}  val_loss={dec_ckpt['val_loss']:.6f}  (frozen)")

    with open(STATS_PATH) as f:
        stats = json.load(f)
    with open(MANIFEST) as f:
        manifest = json.load(f)
    return decoder, enc_ckpt, encoder, manifest, stats


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
    decoder,
    device,
    encoder,
    manifest,
    np,
    stats,
    torch,
):
    ds     = CVDataset(DATA_DIR, manifest['index'][:N_SIMS], stats)
    loader = DataLoader(ds, batch_size=256, num_workers=0)

    all_pred, all_gt = [], []
    with torch.no_grad():
        for _, waves_cont_b, waves_valve_b in loader:
            _pred_params = encoder(waves_cont_b.to(device), waves_valve_b.to(device))
            pred_cont_b, _ = decoder(_pred_params)
            all_pred.append(pred_cont_b.cpu())
            all_gt.append(waves_cont_b)
    ds.close()

    all_pred = torch.cat(all_pred).numpy()   # (N, 24, 201)  — z-scored
    all_gt   = torch.cat(all_gt).numpy()

    wave_means = np.array([stats['waves'][n]['mean'] for n in WAVE_NAMES_CONT])
    wave_stds  = np.array([stats['waves'][n]['std']  for n in WAVE_NAMES_CONT])
    pred_phys  = all_pred * wave_stds[None, :, None] + wave_means[None, :, None]
    gt_phys    = all_gt   * wave_stds[None, :, None] + wave_means[None, :, None]
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
    abs_err = np.abs(pred_phys - gt_phys)          # (N, 24, 201)

    # MAE per channel
    sim_ch_mae    = abs_err.mean(axis=2)            # (N, 24)
    ch_mae        = sim_ch_mae.mean(axis=0)         # (24,)

    # Median AE per channel (median over sims × time)
    ch_med_ae     = np.median(abs_err.reshape(abs_err.shape[0], abs_err.shape[1], -1), axis=(0, 2))  # (24,)

    # Standard MAPE — inflated for Q/V channels that cross zero
    std_pct_err   = abs_err / (np.abs(gt_phys) + 1e-6) * 100
    sim_ch_mape   = std_pct_err.mean(axis=2)        # (N, 24)
    ch_mape       = sim_ch_mape.mean(axis=0)        # (24,)

    # Range-normalised error — safe for zero-crossing channels
    sig_range     = np.maximum(gt_phys.max(axis=2) - gt_phys.min(axis=2), 1e-6)   # (N, 24)
    rng_pct_err   = abs_err / sig_range[:, :, None] * 100
    sim_ch_rne    = rng_pct_err.mean(axis=2)        # (N, 24)
    ch_rne        = sim_ch_rne.mean(axis=0)         # (24,)

    print(f"{'Channel':<8} {'MAE':>10} {'MedAE':>10} {'MAPE (%)':>10} {'RangeErr (%)':>14}")
    print('-' * 56)
    for i, _name in enumerate(WAVE_NAMES_CONT):
        print(f'{_name:<8} {ch_mae[i]:>10.4f} {ch_med_ae[i]:>10.4f} {ch_mape[i]:>10.2f} {ch_rne[i]:>14.2f}')
    return ch_mae, ch_mape, ch_med_ae, sim_ch_mae, sim_ch_rne


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Per-channel MAE and Median AE
    """)
    return


@app.cell
def _(
    N_SIMS,
    WAVE_NAMES_CONT,
    ch_mae,
    ch_med_ae,
    enc_ckpt,
    np,
    plt,
    sim_ch_mae,
):
    x = np.arange(len(WAVE_NAMES_CONT))
    _fig, _axes = plt.subplots(1, 3, figsize=(22, 5))

    _axes[0].boxplot(
        sim_ch_mae,
        tick_labels=WAVE_NAMES_CONT,
        patch_artist=True,
        boxprops=dict(facecolor='steelblue', alpha=0.6),
    )
    _axes[0].set_title('MAE distribution per channel')
    _axes[0].set_ylabel('MAE (physical units)')
    _axes[0].tick_params(axis='x', rotation=45)

    _axes[1].bar(x, ch_mae, color='steelblue', alpha=0.8, label='Mean AE')
    _axes[1].bar(x, ch_med_ae, color='darkcyan', alpha=0.8, label='Median AE')
    _axes[1].set_xticks(x)
    _axes[1].set_xticklabels(WAVE_NAMES_CONT, rotation=45, ha='right')
    _axes[1].set_title('Mean vs Median AE per channel')
    _axes[1].set_ylabel('AE (physical units)')
    _axes[1].legend()

    _axes[2].bar(x, ch_mae - ch_med_ae, color='mediumpurple', alpha=0.8)
    _axes[2].set_xticks(x)
    _axes[2].set_xticklabels(WAVE_NAMES_CONT, rotation=45, ha='right')
    _axes[2].set_title('Mean AE − Median AE  (skew indicator)')
    _axes[2].set_ylabel('MAE − MedAE (physical units)')
    _axes[2].axhline(0, color='black', linewidth=0.8)

    _fig.suptitle(f'Autoenc-baseline — AE metrics  (epoch {enc_ckpt["epoch"]}, n={N_SIMS})', fontsize=13)
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
def _(N_SIMS, WAVE_NAMES_CONT, ch_mape, enc_ckpt, plt, sim_ch_rne, x):
    _fig, _axes = plt.subplots(1, 2, figsize=(18, 5))

    _axes[0].bar(x, ch_mape, color='tomato', alpha=0.8)
    _axes[0].set_xticks(x)
    _axes[0].set_xticklabels(WAVE_NAMES_CONT, rotation=45, ha='right')
    _axes[0].set_title('Standard MAPE — inflated for zero-crossing Q/V channels')
    _axes[0].set_ylabel('MAPE (%)')

    _axes[1].boxplot(
        sim_ch_rne,
        tick_labels=WAVE_NAMES_CONT,
        patch_artist=True,
        boxprops=dict(facecolor='darkorange', alpha=0.6),
    )
    _axes[1].set_title('Range-normalised error distribution')
    _axes[1].set_ylabel('Range-norm. error (%)')
    _axes[1].tick_params(axis='x', rotation=45)

    _fig.suptitle(f'Autoenc-baseline — MAPE comparison  (epoch {enc_ckpt["epoch"]}, n={N_SIMS})', fontsize=13)
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
    decoder,
    device,
    encoder,
    manifest,
    np,
    stats,
    torch,
):
    ds3     = CVDataset(DATA_DIR, [manifest['index'][i] for i in SIM_INDICES], stats)
    samples = [ds3[j] for j in range(len(SIM_INDICES))]
    ds3.close()

    pred_conts, pred_valves, gt_conts, gt_valves = [], [], [], []
    for _, waves_cont, waves_valve in samples:
        with torch.no_grad():
            _pred_params = encoder(
                waves_cont.unsqueeze(0).to(device),
                waves_valve.unsqueeze(0).to(device),
            )
            pc, pv = decoder(_pred_params)
        pred_conts.append(pc.squeeze(0).cpu().numpy())
        pred_valves.append(pv.squeeze(0).cpu().numpy())
        gt_conts.append(waves_cont.numpy())
        gt_valves.append(waves_valve.numpy())

    t = np.arange(201)
    return gt_conts, gt_valves, pred_conts, pred_valves, t


@app.cell
def _(SIM_INDICES, WAVE_NAMES_CONT, enc_ckpt, gt_conts, plt, pred_conts, t):
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
    _fig.suptitle(f'Autoenc-baseline — continuous waveforms (epoch {enc_ckpt["epoch"]})', fontsize=12)
    plt.tight_layout()
    plt.show()
    return (ncols,)


@app.cell
def _(
    SIM_INDICES,
    VALVE_NAMES,
    enc_ckpt,
    gt_valves,
    ncols,
    plt,
    pred_valves,
    t,
):
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
    _fig.suptitle(f'Autoenc-baseline — valve signals (epoch {enc_ckpt["epoch"]})', fontsize=12)
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
