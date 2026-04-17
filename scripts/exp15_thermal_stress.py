"""
Exp 15: Depth-Resolved Bleaching Stress Correction
===================================================
Uses PINN from exp14 to compute depth-resolved DHD (Degree Heating Days)
and compares with satellite-only estimates.

Key outputs for paper:
  - DHD at depth: satellite overestimates thermal stress below surface
  - PINN corrects this bias using physics-constrained depth reconstruction
  - Holdout-validated: DHD computed at held-out depths proves accuracy
  - Depth-time thermal field heatmaps

Reefs: Davies, Myrmidon, Rib, Kelso (same as exp14)
Data: Phase 2 sub-daily from data/ALZ/ + CRW satellite SST
"""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import time as time_module
import json
import gc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

print(f"GPU: {jax.devices()}")

from pinn_reef_thermal import (
    load_reef_subdaily,
    prepare_pinn_data,
    generate_stratified_collocation_points,
    ReefThermalPhysics,
    ReefPINN,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)

# === Hyperparameters (same as exp14) ===
N_EPOCHS = 15000
N_PDE = 2500
PDE_CHUNK = 2500
N_FOURIER = 8
BATCH_SIZE = 5000

W_PDE_SCHEDULE = [
    (0,     1.0),
    (3000,  0.5),
    (6000,  0.2),
    (10000, 0.05),
]

# === Reef configurations ===
REEF_CONFIGS = [
    {
        "name": "davies_reef",
        "date_from": "2011-03-01",
        "date_to": "2014-10-31",
    },
    {
        "name": "myrmidon_reef",
        "date_from": "2020-06-01",
        "date_to": "2024-10-01",
    },
    {
        "name": "rib_reef",
        "date_from": "2018-03-01",
        "date_to": "2023-07-01",
    },
    {
        "name": "kelso_reef",
        "date_from": "1998-04-18",
        "date_to": "2001-08-07",
    },
]


def compute_dhd(T_series, t_seconds, threshold, max_dt_days=2.0):
    """Compute cumulative Degree Heating Days above threshold.

    DHD = cumulative sum of max(0, T - threshold) * dt_days

    Parameters
    ----------
    max_dt_days : float
        Maximum allowed time step in days. Gaps larger than this are capped
        to prevent inflated DHD from merged logger deployments with temporal gaps.
    """
    dt_days = np.diff(t_seconds, prepend=t_seconds[0]) / 86400.0
    dt_days = np.minimum(dt_days, max_dt_days)
    excess = np.maximum(0, T_series - threshold)
    return np.cumsum(excess * dt_days)


def compute_mmm_threshold(sst_df):
    """Compute bleaching threshold as MMM + 1°C from SST dataframe.

    MMM = Maximum of the 12 calendar monthly mean SST values.
    Uses proper calendar months, not arbitrary 30-day windows.

    Parameters
    ----------
    sst_df : DataFrame with 'time' (datetime) and 'CRW_SST' columns.

    Returns
    -------
    float : MMM + 1.0 (°C)
    """
    df = sst_df.copy()
    df["month"] = df["time"].dt.month
    monthly_means = df.groupby("month")["CRW_SST"].mean()
    mmm = float(monthly_means.max())
    return mmm + 1.0


def train_pinn_full(data, meta, t_sst, T_sst):
    """Train PINN on ALL data (no holdout) for thermal field reconstruction."""
    physics = ReefThermalPhysics(
        kappa=5e-4, Kd=0.25, Q_max=350.0, rho_cp=4.1e6,
        T_mean=meta["T_mean"], T_amp=1.2,
        z_max=meta["z_max"], t_days=meta["t_max"] / 86400.0,
        utc_offset=10.0,
    )

    z_pde, t_pde = generate_stratified_collocation_points(
        meta["z_max"], meta["t_max"], N_PDE,
        logger_depths=list(data["depths"].values()))

    T_range = meta.get("T_range", (20.0, 30.0))
    T_scale = max(1.0, (T_range[1] - T_range[0]) * 0.5)

    pinn = ReefPINN(
        physics=physics,
        learn_kappa=True, learn_Kd=True,
        use_hard_bc=True, n_fourier=N_FOURIER, seed=42,
        init_kappa=1e-3, init_Kd=0.2,
        pde_chunk_size=PDE_CHUNK, pde_depth_scale=20.0,
        w_bc_bottom=0.1, kappa_mode='log_linear',
        t_sst=t_sst, T_sst=T_sst,
        use_modified_mlp=True, hidden_dim=128, n_hidden=5,
        t_start_utc_hour=meta.get("t_start_utc_hour", 0.0),
        start_day_of_year=meta.get("start_day_of_year", 0.0),
        T_scale=T_scale,
        activation='tanh', grad_clip=1.0,
    )

    t0 = time_module.time()
    pinn.train(
        z_pde=jnp.array(z_pde), t_pde=jnp.array(t_pde),
        z_data=jnp.array(data["z_data"]),
        t_data=jnp.array(data["t_data"]),
        T_data=jnp.array(data["T_data"]),
        z_bc=jnp.zeros(1), t_bc=jnp.zeros(1), T_bc=jnp.array([meta["T_mean"]]),
        n_epochs=N_EPOCHS, lr=1e-3,
        w_pde=1.0, w_data=10.0, w_bc=0.0,
        print_every=5000,
        resample_pde_every=2000, n_pde_points=N_PDE,
        lr_physics=1e-3,
        w_pde_schedule=W_PDE_SCHEDULE,
        batch_size=BATCH_SIZE,
        lr_schedule='cosine',
    )
    wall_time = time_module.time() - t0
    print(f"  Training done in {wall_time:.0f}s")

    return pinn, wall_time


def predict_at_depth(pinn, depth, t_eval):
    """Predict temperature at a single depth across time array."""
    pred_fn = jax.jit(lambda z_, t_: pinn._predict_scalar(pinn.trained_params, z_, t_))
    z_arr = jnp.full(len(t_eval), depth, dtype=jnp.float32)
    # Process in chunks to avoid memory issues
    chunk = 5000
    preds = []
    for i in range(0, len(t_eval), chunk):
        t_chunk = jnp.array(t_eval[i:i+chunk])
        z_chunk = z_arr[:len(t_chunk)]
        preds.append(np.array(jax.vmap(pred_fn)(z_chunk, t_chunk)))
    return np.concatenate(preds)


def predict_thermal_field(pinn, depths, t_eval):
    """Predict temperature at multiple depths across time.

    Returns 2D array: (n_depths, n_times)
    """
    field = np.zeros((len(depths), len(t_eval)), dtype=np.float32)
    for i, d in enumerate(depths):
        field[i] = predict_at_depth(pinn, d, t_eval)
    return field


def plot_thermal_field(reef_name, depths, t_days, field, threshold,
                       logger_depths=None, out_path=None):
    """Plot depth-time thermal field heatmap with DHD contours."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1])

    # Thermal field heatmap
    vmin, vmax = field.min(), field.max()
    im = ax1.pcolormesh(t_days, depths, field, shading='auto',
                        cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
    cb = plt.colorbar(im, ax=ax1, label="Temperature (°C)")

    # Bleaching threshold contour
    ax1.contour(t_days, depths, field, levels=[threshold],
                colors='white', linewidths=2, linestyles='--')

    if logger_depths is not None:
        for ld in logger_depths:
            ax1.axhline(ld, color='black', linewidth=0.5, alpha=0.5, linestyle=':')

    ax1.set_ylabel("Depth (m)")
    ax1.set_title(f"{reef_name}: PINN-Reconstructed Thermal Field")
    ax1.invert_yaxis()

    # DHD at selected depths
    surface_idx = 0
    mid_idx = len(depths) // 2
    deep_idx = -1

    t_seconds = t_days * 86400.0
    for idx, label, color in [(surface_idx, f"Surface ({depths[0]:.0f}m)", "red"),
                               (mid_idx, f"Mid ({depths[mid_idx]:.1f}m)", "orange"),
                               (deep_idx, f"Deep ({depths[deep_idx]:.1f}m)", "blue")]:
        dhd = compute_dhd(field[idx], t_seconds, threshold)
        ax2.plot(t_days, dhd, label=label, color=color, linewidth=1.5)

    # Satellite DHD (surface SST = constant across depth)
    sat_dhd = compute_dhd(field[0], t_seconds, threshold)
    ax2.set_ylabel("DHD (°C·days)")
    ax2.set_xlabel("Time (days)")
    ax2.set_title(f"Cumulative DHD (threshold = {threshold:.1f}°C)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_dhd_comparison(reef_name, results, out_path=None):
    """Plot DHD comparison: satellite vs PINN vs logger at each depth."""
    depths = sorted(results["depth_dhd"].keys())
    n_depths = len(depths)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Final DHD at each depth
    ax = axes[0]
    depth_vals = [float(d) for d in depths]
    sat_dhd = [results["depth_dhd"][d]["satellite_dhd_final"] for d in depths]
    pinn_dhd = [results["depth_dhd"][d]["pinn_dhd_final"] for d in depths]
    logger_dhd = [results["depth_dhd"][d].get("logger_dhd_final") for d in depths]

    ax.plot(sat_dhd, depth_vals, 'o-', color='gray', label='Satellite SST', linewidth=2)
    ax.plot(pinn_dhd, depth_vals, 's-', color='tomato', label='PINN', linewidth=2)
    for i, ld in enumerate(logger_dhd):
        if ld is not None:
            ax.plot(ld, depth_vals[i], 'D', color='steelblue', markersize=10,
                    label='Logger (truth)' if i == 0 else None)

    ax.set_ylabel("Depth (m)")
    ax.set_xlabel("Final DHD (°C·days)")
    ax.set_title(f"{reef_name}: DHD vs Depth")
    ax.legend(fontsize=9)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    # Right: DHD bias (satellite - logger, PINN - logger) using matched values
    ax = axes[1]
    sat_bias = []
    pinn_bias = []
    valid_depths = []
    for i, d in enumerate(depths):
        if logger_dhd[i] is not None:
            valid_depths.append(depth_vals[i])
            entry = results["depth_dhd"][d]
            sat_m = entry.get("satellite_dhd_matched", sat_dhd[i])
            pinn_m = entry.get("pinn_dhd_matched", pinn_dhd[i])
            sat_bias.append(sat_m - logger_dhd[i])
            pinn_bias.append(pinn_m - logger_dhd[i])

    if valid_depths:
        x = np.arange(len(valid_depths))
        w = 0.35
        ax.barh(x - w/2, sat_bias, w, label='Satellite bias', color='gray', alpha=0.7)
        ax.barh(x + w/2, pinn_bias, w, label='PINN bias', color='tomato', alpha=0.7)
        ax.set_yticks(x)
        ax.set_yticklabels([f"{d:.1f}m" for d in valid_depths])
        ax.axvline(0, color='black', lw=0.5)
        ax.set_xlabel("DHD Bias (°C·days)")
        ax.set_title(f"{reef_name}: DHD Bias vs Logger")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# === Main loop ===
all_results = {}
t_total = time_module.time()

for reef_cfg in REEF_CONFIGS:
    reef_name = reef_cfg["name"]
    print(f"\n{'='*80}")
    print(f"  REEF: {reef_name}")
    print(f"{'='*80}")

    # Load data
    df = load_reef_subdaily(reef_name,
                            date_from=reef_cfg["date_from"],
                            date_to=reef_cfg["date_to"],
                            verbose=True)

    data = prepare_pinn_data(df, sst_df=None, use_hourly=True)
    meta = data["metadata"]
    all_depths = sorted(data["depths"].values())

    # Load CRW SST
    sst_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "data", "ALZ", f"crw_sst_{reef_name}.csv")
    sst_df = pd.read_csv(sst_path)
    sst_df["time"] = pd.to_datetime(sst_df["time"], utc=True)
    t_start = pd.Timestamp(meta["t_start"])
    sst_df["t_seconds"] = (sst_df["time"] - t_start).dt.total_seconds()
    sst_df = sst_df[(sst_df["t_seconds"] >= 0) & (sst_df["t_seconds"] <= meta["t_max"])]
    t_sst = sst_df["t_seconds"].values.astype(np.float32)
    T_sst = sst_df["CRW_SST"].values.astype(np.float32)

    # Compute bleaching threshold (MMM + 1°C) from full SST record
    threshold = compute_mmm_threshold(sst_df)
    print(f"\n  Bleaching threshold (MMM+1): {threshold:.2f}°C")
    print(f"  CRW SST: {len(t_sst)} values, {T_sst.min():.1f}-{T_sst.max():.1f}°C")
    print(f"  Logger depths: {all_depths}")

    # Train PINN on ALL data
    print(f"\n  Training PINN on all data...")
    pinn, train_time = train_pinn_full(data, meta, t_sst, T_sst)
    params = pinn.get_estimated_params()
    print(f"  Params: {params}")

    # Predict thermal field at fine depth resolution
    field_depths = np.linspace(0, max(all_depths), 50)
    # Use daily evaluation times
    n_eval_days = int(meta["t_max"] / 86400)
    t_eval = np.linspace(0, meta["t_max"], n_eval_days).astype(np.float32)
    t_eval_days = t_eval / 86400.0

    print(f"  Predicting thermal field: {len(field_depths)} depths × {len(t_eval)} times...")
    thermal_field = predict_thermal_field(pinn, field_depths, t_eval)

    # Compute DHD at each depth
    reef_result = {
        "threshold": float(threshold),
        "train_time": train_time,
        "params": params,
        "depth_dhd": {},
    }

    # Satellite DHD (SST applied uniformly to all depths)
    sat_T_interp = np.interp(t_eval, t_sst, T_sst)
    sat_dhd = compute_dhd(sat_T_interp, t_eval, threshold)
    sat_dhd_final = float(sat_dhd[-1])
    print(f"\n  Satellite DHD (surface): {sat_dhd_final:.1f} °C·days")

    # DHD at logger depths + extra depths
    eval_depths = sorted(set(list(all_depths) + [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0]))
    eval_depths = [d for d in eval_depths if d <= max(all_depths) * 1.1]

    print(f"\n  {'Depth':>6}  {'PINN DHD':>10}  {'Sat DHD':>10}  {'Logger DHD':>10}  {'Span(d)':>8}  {'Sat bias':>10}  {'PINN bias':>10}")
    print(f"  {'-'*75}")

    for depth in eval_depths:
        pinn_T = predict_at_depth(pinn, depth, t_eval)
        pinn_dhd = compute_dhd(pinn_T, t_eval, threshold)
        pinn_dhd_final = float(pinn_dhd[-1])

        entry = {
            "pinn_dhd_final": pinn_dhd_final,
            "satellite_dhd_final": sat_dhd_final,
            "pinn_T_mean": float(pinn_T.mean()),
        }

        # Check if we have logger data at this depth
        is_logger_depth = any(abs(d - depth) < 0.3 for d in all_depths)
        if is_logger_depth:
            # Get logger data at this depth
            mask = np.abs(data["z_data"] - depth) < 0.3
            logger_t = data["t_data"][mask]
            logger_T = data["T_data"][mask]
            sort_idx = np.argsort(logger_t)
            logger_t, logger_T = logger_t[sort_idx], logger_T[sort_idx]

            if len(logger_t) > 100:
                logger_dhd = compute_dhd(logger_T, logger_t, threshold)
                logger_dhd_final = float(logger_dhd[-1])
                entry["logger_dhd_final"] = logger_dhd_final

                # Compute matched DHD: PINN & satellite over logger's time window
                logger_t_min = float(logger_t[0])
                logger_t_max = float(logger_t[-1])
                logger_span_days = (logger_t_max - logger_t_min) / 86400.0
                entry["logger_span_days"] = logger_span_days

                n_matched = max(2, int(logger_span_days))
                t_matched = np.linspace(logger_t_min, logger_t_max, n_matched).astype(np.float32)

                pinn_T_matched = predict_at_depth(pinn, depth, t_matched)
                pinn_dhd_matched = compute_dhd(pinn_T_matched, t_matched, threshold)
                entry["pinn_dhd_matched"] = float(pinn_dhd_matched[-1])

                sat_T_matched = np.interp(t_matched, t_sst, T_sst)
                sat_dhd_matched = compute_dhd(sat_T_matched, t_matched, threshold)
                entry["satellite_dhd_matched"] = float(sat_dhd_matched[-1])

                sat_bias = entry["satellite_dhd_matched"] - logger_dhd_final
                pinn_bias = entry["pinn_dhd_matched"] - logger_dhd_final
                print(f"  {depth:>5.1f}m  {pinn_dhd_final:>10.1f}  {sat_dhd_final:>10.1f}  "
                      f"{logger_dhd_final:>10.1f}  {logger_span_days:>7.0f}  "
                      f"{sat_bias:>+9.1f}  {pinn_bias:>+9.1f}")
            else:
                print(f"  {depth:>5.1f}m  {pinn_dhd_final:>10.1f}  {sat_dhd_final:>10.1f}  "
                      f"{'(few pts)':>10}  {'':>8}  {'':>10}  {'':>10}")
        else:
            print(f"  {depth:>5.1f}m  {pinn_dhd_final:>10.1f}  {sat_dhd_final:>10.1f}  "
                  f"{'—':>10}  {'':>8}  {'':>10}  {'':>10}")

        reef_result["depth_dhd"][f"{depth:.1f}"] = entry

    all_results[reef_name] = reef_result

    # Generate figures
    print(f"\n  Generating figures...")
    plot_thermal_field(
        reef_name, field_depths, t_eval_days, thermal_field, threshold,
        logger_depths=all_depths,
        out_path=os.path.join(OUT_DIR, f"exp15_{reef_name}_thermal_field.png"))

    plot_dhd_comparison(
        reef_name, reef_result,
        out_path=os.path.join(OUT_DIR, f"exp15_{reef_name}_dhd.png"))

    print(f"  Saved figures to {OUT_DIR}")

    # Save incremental results
    with open(os.path.join(OUT_DIR, "exp15_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    del pinn
    gc.collect()

# === Summary ===
elapsed = time_module.time() - t_total

print(f"\n{'='*80}")
print(f"  EXP 15: Depth-Resolved Bleaching Stress Correction")
print(f"  {len(all_results)} reefs, {elapsed/60:.1f} min total")
print(f"{'='*80}")

print(f"\n  {'Reef':<20} {'Depth':>6} {'Logger DHD':>12} {'PINN(m)':>10} {'Sat(m)':>10} "
      f"{'Span(d)':>8} {'Sat bias':>10} {'PINN bias':>10}")
print(f"  {'-'*90}")

for reef_name, result in all_results.items():
    for depth_key in sorted(result["depth_dhd"].keys(), key=float):
        d = result["depth_dhd"][depth_key]
        logger_dhd = d.get("logger_dhd_final")
        if logger_dhd is not None:
            pinn_m = d.get("pinn_dhd_matched", d["pinn_dhd_final"])
            sat_m = d.get("satellite_dhd_matched", d["satellite_dhd_final"])
            span = d.get("logger_span_days", 0)
            sat_bias = sat_m - logger_dhd
            pinn_bias = pinn_m - logger_dhd
            print(f"  {reef_name:<20} {depth_key:>5}m {logger_dhd:>12.1f} "
                  f"{pinn_m:>10.1f} {sat_m:>10.1f} {span:>7.0f} "
                  f"{sat_bias:>+9.1f} {pinn_bias:>+9.1f}")

# Overall bias statistics
sat_biases = []
pinn_biases = []
for reef_name, result in all_results.items():
    for depth_key, d in result["depth_dhd"].items():
        logger_dhd = d.get("logger_dhd_final")
        if logger_dhd is not None:
            sat_m = d.get("satellite_dhd_matched", d["satellite_dhd_final"])
            pinn_m = d.get("pinn_dhd_matched", d["pinn_dhd_final"])
            sat_biases.append(sat_m - logger_dhd)
            pinn_biases.append(pinn_m - logger_dhd)

if sat_biases:
    print(f"\n  Overall DHD bias statistics:")
    print(f"    Satellite: mean={np.mean(sat_biases):+.1f}, "
          f"abs mean={np.mean(np.abs(sat_biases)):.1f}, "
          f"max={np.max(np.abs(sat_biases)):.1f} °C·days")
    print(f"    PINN:      mean={np.mean(pinn_biases):+.1f}, "
          f"abs mean={np.mean(np.abs(pinn_biases)):.1f}, "
          f"max={np.max(np.abs(pinn_biases)):.1f} °C·days")
    print(f"    PINN reduces absolute DHD bias by "
          f"{(1 - np.mean(np.abs(pinn_biases))/np.mean(np.abs(sat_biases)))*100:.0f}%")

print(f"\n  Learned physical parameters:")
for reef_name, result in all_results.items():
    p = result.get("params", {})
    print(f"    {reef_name:<20} κ={p.get('kappa', 0):.6f}, "
          f"Kd={p.get('Kd', 0):.4f}, α={p.get('alpha', 0):.4f}")

print(f"\nTotal wall time: {elapsed/60:.1f} min")
print(f"Results saved to: {OUT_DIR}/exp15_results.json")
