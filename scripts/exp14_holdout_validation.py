"""
Exp 14: Enhanced PINN — All Changelog Fixes on 4 Reefs
=======================================================
Applies ALL bug fixes + enhancements from CHANGELOG.md:
  - Modified MLP (Wang et al. 2021) with 5 hidden layers, tanh activation
  - Learnable Kd (light attenuation)
  - Gradient clipping (global norm = 1.0)
  - Descending PDE schedule: 1.0 → 0.05 (physics-first, then data-dominated)
  - Adaptive T_scale from data range
  - z_max * 1.2 (tighter domain)
  - PDE chunk padding mask, evenly-spaced bottom BC

Reefs:
  - Davies Reef 2011-2014 (20 depths)
    Holdouts: 5.0m, 9.1m, 18.5m; sparsity: 17, 10, 5, 3, 2
  - Myrmidon Reef 2020-2024 (6 depths)
    Holdouts: 4.7m, 7.3m, 14.7m; sparsity: 5, 3, 2
  - Rib Reef 2018-2023 (3 depths)
    Holdouts: 1.0m, 6.0m, 9.0m; sparsity: 2
  - Kelso Reef 1998-2001 (3 depths: 2.0, 7.0, 19.0)
    Holdouts: 2.0m, 7.0m, 19.0m; sparsity: 2
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

print(f"GPU: {jax.devices()}")

from pinn_reef_thermal import (
    load_reef_subdaily,
    prepare_pinn_data,
    generate_stratified_collocation_points,
    ReefThermalPhysics,
    ReefPINN,
    run_all_baselines,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)

# === Hyperparameters (all changelog enhancements) ===
N_EPOCHS = 15000
N_PDE = 2500
PDE_CHUNK = 2500   # single chunk — avoids JIT loop unrolling OOM
N_FOURIER = 8
BATCH_SIZE = 5000

# Descending PDE schedule: physics-first, then data-dominated
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
        "holdout_depths": [5.0, 9.1, 18.5],
        "sparsity_levels": [17, 10, 5, 3, 2],
    },
    {
        "name": "myrmidon_reef",
        "date_from": "2020-06-01",
        "date_to": "2024-10-01",
        "holdout_depths": [4.7, 7.3, 14.7],
        "sparsity_levels": [5, 3, 2],
    },
    {
        "name": "rib_reef",
        "date_from": "2018-03-01",
        "date_to": "2023-07-01",
        "holdout_depths": [1.0, 6.0, 9.0],
        "sparsity_levels": [2],
    },
    {
        "name": "kelso_reef",
        "date_from": "1998-04-18",
        "date_to": "2001-08-07",
        "holdout_depths": [2.0, 7.0, 19.0],
        "sparsity_levels": [2],
    },
]


def select_sparse_depths(all_depths, n_keep, holdout_depth, tol=0.3):
    available = sorted([d for d in all_depths if abs(d - holdout_depth) >= tol])
    if n_keep >= len(available):
        return available
    indices = np.round(np.linspace(0, len(available) - 1, n_keep)).astype(int)
    return [available[i] for i in indices]


def filter_data_to_depths(data, keep_depths, tol=0.3):
    z_d = data["z_data"]
    mask = np.zeros(len(z_d), dtype=bool)
    kept_keys = {}
    for k, v in data["depths"].items():
        if any(abs(v - d) < tol for d in keep_depths):
            mask |= (np.abs(z_d - v) < tol)
            kept_keys[k] = v
    return {
        "z_data": z_d[mask], "t_data": data["t_data"][mask],
        "T_data": data["T_data"][mask],
        "z_bc": data["z_bc"], "t_bc": data["t_bc"], "T_bc": data["T_bc"],
        "depths": kept_keys, "metadata": data["metadata"],
    }


def make_holdout(data, holdout_depth, tol=0.3):
    z_d = data["z_data"]
    mask = np.abs(z_d - holdout_depth) < tol
    return {"z": z_d[mask], "t": data["t_data"][mask], "T": data["T_data"][mask]}


def train_pinn_enhanced(train_data, holdout_data, meta, t_sst, T_sst):
    """Train PINN with ALL changelog enhancements."""
    physics = ReefThermalPhysics(
        kappa=5e-4, Kd=0.25, Q_max=350.0, rho_cp=4.1e6,
        T_mean=meta["T_mean"], T_amp=1.2,
        z_max=meta["z_max"], t_days=meta["t_max"] / 86400.0,
        utc_offset=10.0,
    )

    z_pde, t_pde = generate_stratified_collocation_points(
        meta["z_max"], meta["t_max"], N_PDE,
        logger_depths=list(train_data["depths"].values()))

    # Adaptive T_scale from data range
    T_range = meta.get("T_range", (20.0, 30.0))
    T_scale = max(1.0, (T_range[1] - T_range[0]) * 0.5)

    pinn = ReefPINN(
        physics=physics,
        learn_kappa=True,
        learn_Kd=True,           # Enhancement #7
        use_hard_bc=True,
        n_fourier=N_FOURIER,
        seed=42,
        init_kappa=1e-3,
        init_Kd=0.2,
        pde_chunk_size=PDE_CHUNK,
        pde_depth_scale=20.0,
        w_bc_bottom=0.1,
        kappa_mode='log_linear',
        t_sst=t_sst,
        T_sst=T_sst,
        use_modified_mlp=True,   # Enhancement #4
        hidden_dim=128,
        n_hidden=5,              # Enhancement #4 (was 4)
        t_start_utc_hour=meta.get("t_start_utc_hour", 0.0),
        start_day_of_year=meta.get("start_day_of_year", 0.0),
        T_scale=T_scale,         # Enhancement #8: adaptive
        activation='tanh',       # tanh outperforms swish (A/B tested)
        grad_clip=1.0,           # Enhancement #3
    )

    t0 = time_module.time()
    pinn.train(
        z_pde=jnp.array(z_pde), t_pde=jnp.array(t_pde),
        z_data=jnp.array(train_data["z_data"]),
        t_data=jnp.array(train_data["t_data"]),
        T_data=jnp.array(train_data["T_data"]),
        z_bc=jnp.zeros(1), t_bc=jnp.zeros(1), T_bc=jnp.array([meta["T_mean"]]),
        n_epochs=N_EPOCHS, lr=1e-3,
        w_pde=1.0, w_data=10.0, w_bc=0.0,
        print_every=5000,
        resample_pde_every=2000, n_pde_points=N_PDE,
        lr_physics=1e-3,
        w_pde_schedule=W_PDE_SCHEDULE,  # Descending schedule
        batch_size=BATCH_SIZE,
        lr_schedule='cosine',
    )
    wall_time = time_module.time() - t0

    # Evaluate
    pred_fn = jax.jit(lambda z_, t_: pinn._predict_scalar(pinn.trained_params, z_, t_))
    n_eval = min(5000, len(holdout_data["z"]))
    eval_idx = np.linspace(0, len(holdout_data["z"])-1, n_eval, dtype=int)
    T_pred = np.array(jax.vmap(pred_fn)(
        jnp.array(holdout_data["z"][eval_idx]),
        jnp.array(holdout_data["t"][eval_idx])
    ))
    T_true = holdout_data["T"][eval_idx]

    rmse = float(np.sqrt(np.mean((T_pred - T_true) ** 2)))
    mae = float(np.mean(np.abs(T_pred - T_true)))
    params = pinn.get_estimated_params()

    del pinn
    gc.collect()

    return rmse, mae, params, wall_time


# === Main loop ===
all_results = []
t_total = time_module.time()

for reef_cfg in REEF_CONFIGS:
    reef_name = reef_cfg["name"]
    print(f"\n{'='*80}")
    print(f"  REEF: {reef_name}")
    print(f"  Period: {reef_cfg['date_from']} to {reef_cfg['date_to']}")
    print(f"{'='*80}")

    # Load data
    df = load_reef_subdaily(reef_name,
                            date_from=reef_cfg["date_from"],
                            date_to=reef_cfg["date_to"],
                            verbose=True)

    # Build base data (SST from real CRW satellite, loaded separately below)
    data = prepare_pinn_data(df, sst_df=None, use_hourly=True)
    meta = data["metadata"]
    all_depths = sorted(data["depths"].values())
    print(f"\nAll depths ({len(all_depths)}): {all_depths}")
    print(f"Total hourly points: {len(data['z_data']):,}")
    print(f"T range: {meta['T_range'][0]:.1f}-{meta['T_range'][1]:.1f}°C")

    T_scale = max(1.0, (meta["T_range"][1] - meta["T_range"][0]) * 0.5)
    print(f"Adaptive T_scale: {T_scale:.2f}")

    # Load real CRW satellite SST (pre-downloaded to data/ALZ/)
    sst_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "data", "ALZ", f"crw_sst_{reef_name}.csv")
    sst_df = pd.read_csv(sst_path)
    sst_df["time"] = pd.to_datetime(sst_df["time"], utc=True)
    t_start = pd.Timestamp(meta["t_start"])
    sst_df["t_seconds"] = (sst_df["time"] - t_start).dt.total_seconds()
    sst_df = sst_df[(sst_df["t_seconds"] >= 0) & (sst_df["t_seconds"] <= meta["t_max"])]
    t_sst = sst_df["t_seconds"].values.astype(np.float32)
    T_sst = sst_df["CRW_SST"].values.astype(np.float32)
    print(f"  CRW satellite SST: {len(t_sst)} daily values, "
          f"{T_sst.min():.1f}-{T_sst.max():.1f}°C")

    for holdout_depth in reef_cfg["holdout_depths"]:
        holdout_data = make_holdout(data, holdout_depth)
        n_holdout = len(holdout_data["z"])
        if n_holdout < 50:
            print(f"\n  Skipping holdout {holdout_depth}m ({n_holdout} pts)")
            continue

        print(f"\n{'#'*80}")
        print(f"  {reef_name} — HOLDOUT: {holdout_depth}m ({n_holdout:,} pts)")
        print(f"{'#'*80}")

        for n_depths in reef_cfg["sparsity_levels"]:
            keep_depths = select_sparse_depths(all_depths, n_depths, holdout_depth)
            actual_n = len(keep_depths)
            if actual_n < 2:
                continue

            print(f"\n  === {actual_n} training depths ===")
            print(f"  Depths: {[f'{d:.1f}' for d in keep_depths]}")

            train_data = filter_data_to_depths(data, keep_depths)
            z_d = train_data["z_data"]
            mask = np.abs(z_d - holdout_depth) >= 0.3
            train_data = {**train_data,
                "z_data": z_d[mask], "t_data": train_data["t_data"][mask],
                "T_data": train_data["T_data"][mask],
            }
            n_train = len(train_data["z_data"])
            print(f"  Train points: {n_train:,}")

            row = {
                "reef": reef_name,
                "holdout_depth": holdout_depth,
                "n_training_depths": actual_n,
                "training_depths": keep_depths,
                "n_train": n_train, "n_holdout": n_holdout,
            }

            # Baselines
            n_eval = min(5000, n_holdout)
            eval_idx = np.linspace(0, n_holdout-1, n_eval, dtype=int)
            T_true_eval = holdout_data["T"][eval_idx]

            baselines = run_all_baselines(train_data, holdout_data)
            for name, T_pred in baselines.items():
                if name.endswith("_std"):
                    continue
                rmse = float(np.sqrt(np.mean((T_pred[eval_idx] - T_true_eval) ** 2)))
                row[f"{name}_rmse"] = rmse
                print(f"  {name:<12} RMSE: {rmse:.4f}°C")

            # Enhanced PINN
            print(f"  Training Enhanced PINN...")
            try:
                rmse, mae, params, wt = train_pinn_enhanced(
                    train_data, holdout_data, meta, t_sst, T_sst)
                row["PINN_rmse"] = rmse
                row["PINN_mae"] = mae
                row["PINN_params"] = params
                row["PINN_time"] = wt
                print(f"  PINN         RMSE: {rmse:.4f}°C, MAE: {mae:.4f}°C ({wt:.0f}s)")
                print(f"  Params: {params}")
            except Exception as e:
                print(f"  PINN FAILED: {e}")
                import traceback
                traceback.print_exc()
                row["PINN_rmse"] = None

            # Best method
            method_rmses = {}
            for k, v in row.items():
                if k.endswith("_rmse") and v is not None:
                    method_rmses[k.replace("_rmse", "")] = v
            if method_rmses:
                best = min(method_rmses, key=method_rmses.get)
                row["best"] = best
                print(f"  ** Best: {best} ({method_rmses[best]:.4f}°C)")

            all_results.append(row)

            # Save incremental results after each experiment
            results_path = os.path.join(OUT_DIR, "exp14_results.json")
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

elapsed = time_module.time() - t_total

# Final save
results_path = os.path.join(OUT_DIR, "exp14_results.json")
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2, default=str)

# === Summary ===
methods = ["GP", "IDW", "NN", "RF", "Satellite", "PINN"]

print(f"\n{'='*100}")
print(f"  EXP 14: Enhanced PINN — All Changelog Fixes")
print(f"  Modified MLP (5 layers, tanh), learn_Kd, grad_clip=1.0, descending PDE schedule")
print(f"  {len(all_results)} experiments across {len(REEF_CONFIGS)} reefs, {elapsed/60:.1f} min total")
print(f"{'='*100}")

for reef_cfg in REEF_CONFIGS:
    reef_name = reef_cfg["name"]
    reef_results = [r for r in all_results if r["reef"] == reef_name]
    if not reef_results:
        continue

    print(f"\n  === {reef_name} ===")
    for holdout_depth in reef_cfg["holdout_depths"]:
        subset = [r for r in reef_results if r["holdout_depth"] == holdout_depth]
        if not subset:
            continue
        print(f"\n  Holdout: {holdout_depth}m")
        print(f"  {'N_depths':>8}  ", end="")
        for m_ in methods:
            print(f"{m_:>10}  ", end="")
        print("Best")
        print("  " + "─" * 85)
        for r in sorted(subset, key=lambda x: -x["n_training_depths"]):
            n = r["n_training_depths"]
            print(f"  {n:>8}  ", end="")
            for m_ in methods:
                v = r.get(f"{m_}_rmse")
                print(f"{v:>10.4f}  " if v is not None else "       -    ", end="")
            print(r.get("best", "-"))

# Cross-reef summary
print(f"\n  === Cross-Reef Summary ===")
print(f"  {'Reef':<20} {'Holdout':>8} {'N':>3}  ", end="")
for m_ in methods:
    print(f"{m_:>10}  ", end="")
print("Best")
print("  " + "─" * 100)
for r in all_results:
    print(f"  {r['reef']:<20} {r['holdout_depth']:>6.1f}m {r['n_training_depths']:>3}  ", end="")
    for m_ in methods:
        v = r.get(f"{m_}_rmse")
        print(f"{v:>10.4f}  " if v is not None else "       -    ", end="")
    print(r.get("best", "-"))

# PINN rank analysis
print(f"\n  === PINN Rank Analysis ===")
for reef_cfg in REEF_CONFIGS:
    reef_name = reef_cfg["name"]
    reef_results = [r for r in all_results if r["reef"] == reef_name]
    if not reef_results:
        continue
    ranks = []
    for r in reef_results:
        rmses = {m_: r[f"{m_}_rmse"] for m_ in methods if r.get(f"{m_}_rmse") is not None}
        sorted_m = sorted(rmses, key=rmses.get)
        if "PINN" in sorted_m:
            ranks.append(sorted_m.index("PINN") + 1)
    if ranks:
        wins = sum(1 for rk in ranks if rk == 1)
        print(f"  {reef_name}: avg rank {np.mean(ranks):.1f}, "
              f"wins {wins}/{len(ranks)}, ranks={ranks}")

# Overall
all_ranks = []
for r in all_results:
    rmses = {m_: r[f"{m_}_rmse"] for m_ in methods if r.get(f"{m_}_rmse") is not None}
    sorted_m = sorted(rmses, key=rmses.get)
    if "PINN" in sorted_m:
        all_ranks.append(sorted_m.index("PINN") + 1)
if all_ranks:
    wins = sum(1 for rk in all_ranks if rk == 1)
    top2 = sum(1 for rk in all_ranks if rk <= 2)
    print(f"\n  Overall: avg rank {np.mean(all_ranks):.1f}, "
          f"wins {wins}/{len(all_ranks)} ({100*wins/len(all_ranks):.0f}%), "
          f"top-2 {top2}/{len(all_ranks)} ({100*top2/len(all_ranks):.0f}%)")

# Physics params
print(f"\n  === Learned Physical Parameters ===")
for r in all_results:
    p = r.get("PINN_params", {})
    if p:
        k = p.get("kappa", 0)
        a = p.get("alpha", 0)
        kd = p.get("Kd", None)
        kd_str = f", Kd={kd:.4f}" if kd is not None else ""
        print(f"  {r['reef']:<20} {r['holdout_depth']:.1f}m / {r['n_training_depths']}d: "
              f"κ={k:.6f}, α={a:.4f}{kd_str}")

print(f"\nTotal wall time: {elapsed/60:.1f} min")
