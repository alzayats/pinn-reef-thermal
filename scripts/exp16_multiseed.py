"""
Exp 16: Reviewer Response — Multi-Seed PINN + Physics-Only FD Baseline
======================================================================
Addresses remaining reviewer concerns:
  1. Multi-seed PINN (≥5 seeds) for uncertainty quantification
  2. Physics-only finite-difference baseline to isolate neural network value

Based on exp14 with two additions:
  - --seeds N flag (default 5) to run N seeds per experiment
  - FD baseline using 1D heat equation with literature-value parameters
  - Reports mean ± std across seeds for PINN, plus FD baseline RMSE

Same reefs, holdouts, and sparsity levels as exp14.
"""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(line_buffering=True)

import argparse
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
    fd_predict,
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

# === Reef configurations (same as exp14) ===
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


def train_pinn_enhanced(train_data, holdout_data, meta, t_sst, T_sst, seed=42):
    """Train PINN with ALL changelog enhancements. Accepts seed for reproducibility."""
    physics = ReefThermalPhysics(
        kappa=5e-4, Kd=0.25, Q_max=350.0, rho_cp=4.1e6,
        T_mean=meta["T_mean"], T_amp=1.2,
        z_max=meta["z_max"], t_days=meta["t_max"] / 86400.0,
        utc_offset=10.0,
    )

    z_pde, t_pde = generate_stratified_collocation_points(
        meta["z_max"], meta["t_max"], N_PDE,
        logger_depths=list(train_data["depths"].values()))

    T_range = meta.get("T_range", (20.0, 30.0))
    T_scale = max(1.0, (T_range[1] - T_range[0]) * 0.5)

    pinn = ReefPINN(
        physics=physics,
        learn_kappa=True,
        learn_Kd=True,
        use_hard_bc=True,
        n_fourier=N_FOURIER,
        seed=seed,
        init_kappa=1e-3,
        init_Kd=0.2,
        pde_chunk_size=PDE_CHUNK,
        pde_depth_scale=20.0,
        w_bc_bottom=0.1,
        kappa_mode='log_linear',
        t_sst=t_sst,
        T_sst=T_sst,
        use_modified_mlp=True,
        hidden_dim=128,
        n_hidden=5,
        t_start_utc_hour=meta.get("t_start_utc_hour", 0.0),
        start_day_of_year=meta.get("start_day_of_year", 0.0),
        T_scale=T_scale,
        activation='tanh',
        grad_clip=1.0,
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
        w_pde_schedule=W_PDE_SCHEDULE,
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


# === Argument parsing ===
parser = argparse.ArgumentParser(description="Exp16: Multi-seed PINN + FD baseline")
parser.add_argument("--seeds", type=int, default=5,
                    help="Number of random seeds for PINN (default: 5)")
parser.add_argument("--reef", type=str, default=None,
                    help="Run only this reef (e.g. davies_reef)")
args = parser.parse_args()

N_SEEDS = args.seeds
SEED_LIST = list(range(42, 42 + N_SEEDS))  # [42, 43, 44, 45, 46]
print(f"\nMulti-seed PINN with {N_SEEDS} seeds: {SEED_LIST}")
print(f"FD physics-only baseline: kappa=2.5e-4, Kd=0.1\n")

# Filter reefs if requested
reef_configs = REEF_CONFIGS
if args.reef:
    reef_configs = [c for c in REEF_CONFIGS if c["name"] == args.reef]
    if not reef_configs:
        print(f"Unknown reef: {args.reef}")
        sys.exit(1)

# === Main loop ===
all_results = []
t_total = time_module.time()

for reef_cfg in reef_configs:
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

    data = prepare_pinn_data(df, sst_df=None, use_hourly=True)
    meta = data["metadata"]
    all_depths = sorted(data["depths"].values())
    print(f"\nAll depths ({len(all_depths)}): {all_depths}")
    print(f"Total hourly points: {len(data['z_data']):,}")
    print(f"T range: {meta['T_range'][0]:.1f}-{meta['T_range'][1]:.1f}°C")

    # Load real CRW satellite SST
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

    # Run FD baseline once per reef (same for all holdouts/sparsity)
    print(f"\n  Running FD physics-only baseline...")
    fd_t0 = time_module.time()
    # Pre-compute FD solution on full grid (reused across holdouts)
    from scipy.interpolate import interp1d, RegularGridInterpolator
    from scipy.linalg import solve_banded

    fd_nz = 100
    fd_dt = 3600.0
    fd_kappa = 2.5e-4
    fd_Kd = 0.1
    fd_Q_max = 350.0
    fd_rho_cp = 4.1e6
    fd_utc_offset = 10.0

    z_max = meta["z_max"]
    t_max = meta["t_max"]
    t_start_utc_hour = meta.get("t_start_utc_hour", 0.0)

    dz = z_max / (fd_nz - 1)
    nt_fd = int(t_max / fd_dt) + 1
    z_fd = np.linspace(0, z_max, fd_nz)
    t_fd = np.linspace(0, t_max, nt_fd)

    sst_interp = interp1d(t_sst, T_sst, kind='linear',
                          fill_value=(T_sst[0], T_sst[-1]),
                          bounds_error=False)

    T_field = np.zeros((fd_nz, nt_fd))
    T_field[:, 0] = float(sst_interp(0.0))

    r = fd_kappa * fd_dt / (dz ** 2)
    print(f"  FD grid: {fd_nz} × {nt_fd} ({nt_fd} hourly steps), r={r:.4f}")

    for n in range(nt_fd - 1):
        t_next = t_fd[n + 1]
        local_hour = ((t_next / 3600.0 + t_start_utc_hour) + fd_utc_offset) % 24.0
        I_surface = fd_Q_max * max(0.0, np.sin(np.pi * (local_hour - 6) / 12))
        S = (I_surface * fd_Kd / fd_rho_cp) * np.exp(-fd_Kd * z_fd) * fd_dt

        main_diag = np.ones(fd_nz) * (1 + 2 * r)
        upper_diag = np.ones(fd_nz - 1) * (-r)
        lower_diag = np.ones(fd_nz - 1) * (-r)
        rhs = T_field[:, n].copy() + S

        main_diag[0] = 1.0
        upper_diag[0] = 0.0
        rhs[0] = float(sst_interp(t_next))

        main_diag[-1] = 1 + r
        lower_diag[-1] = -r
        rhs[-1] = T_field[-1, n] + S[-1]

        ab = np.zeros((3, fd_nz))
        ab[0, 1:] = upper_diag
        ab[1, :] = main_diag
        ab[2, :-1] = lower_diag
        T_field[:, n + 1] = solve_banded((1, 1), ab, rhs)

    fd_interp = RegularGridInterpolator(
        (z_fd, t_fd), T_field, method='linear',
        bounds_error=False, fill_value=None)
    fd_time = time_module.time() - fd_t0
    print(f"  FD solved in {fd_time:.1f}s")

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
                "n_seeds": N_SEEDS,
            }

            # Statistical baselines (deterministic — run once)
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

            # FD physics-only baseline
            fd_pts = np.column_stack([holdout_data["z"], holdout_data["t"]])
            T_fd = fd_interp(fd_pts).astype(np.float32)
            fd_rmse = float(np.sqrt(np.mean((T_fd[eval_idx] - T_true_eval) ** 2)))
            row["FD_rmse"] = fd_rmse
            print(f"  {'FD':<12} RMSE: {fd_rmse:.4f}°C  (κ={fd_kappa}, Kd={fd_Kd})")

            # Multi-seed PINN
            seed_rmses = []
            seed_maes = []
            seed_params = []
            seed_times = []

            for si, seed in enumerate(SEED_LIST):
                print(f"  Training PINN (seed {seed}, {si+1}/{N_SEEDS})...")
                try:
                    rmse, mae, params, wt = train_pinn_enhanced(
                        train_data, holdout_data, meta, t_sst, T_sst, seed=seed)
                    seed_rmses.append(rmse)
                    seed_maes.append(mae)
                    seed_params.append(params)
                    seed_times.append(wt)
                    print(f"    RMSE: {rmse:.4f}°C, MAE: {mae:.4f}°C ({wt:.0f}s)")
                except Exception as e:
                    print(f"    PINN FAILED (seed {seed}): {e}")
                    import traceback
                    traceback.print_exc()

            if seed_rmses:
                row["PINN_rmse_mean"] = float(np.mean(seed_rmses))
                row["PINN_rmse_std"] = float(np.std(seed_rmses))
                row["PINN_rmse_all"] = seed_rmses
                row["PINN_mae_mean"] = float(np.mean(seed_maes))
                row["PINN_mae_std"] = float(np.std(seed_maes))
                row["PINN_params_all"] = seed_params
                row["PINN_time_total"] = float(np.sum(seed_times))
                # For backward compat with analysis scripts
                row["PINN_rmse"] = row["PINN_rmse_mean"]
                row["PINN_mae"] = row["PINN_mae_mean"]

                print(f"\n  PINN ({N_SEEDS} seeds):")
                print(f"    RMSE: {row['PINN_rmse_mean']:.4f} ± {row['PINN_rmse_std']:.4f}°C")
                print(f"    MAE:  {row['PINN_mae_mean']:.4f} ± {row['PINN_mae_std']:.4f}°C")
                print(f"    Per-seed RMSE: {[f'{r:.4f}' for r in seed_rmses]}")
            else:
                row["PINN_rmse"] = None

            # Best method (using PINN mean)
            method_rmses = {}
            for k, v in row.items():
                if k.endswith("_rmse") and v is not None and k not in [
                    "PINN_rmse_mean", "PINN_rmse_std"
                ] and not k.endswith("_all"):
                    method_rmses[k.replace("_rmse", "")] = v
            if method_rmses:
                best = min(method_rmses, key=method_rmses.get)
                row["best"] = best
                print(f"  ** Best: {best} ({method_rmses[best]:.4f}°C)")

            all_results.append(row)

            # Save incremental results
            results_path = os.path.join(OUT_DIR, "exp16_results.json")
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

elapsed = time_module.time() - t_total

# Final save
results_path = os.path.join(OUT_DIR, "exp16_results.json")
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2, default=str)

# === Summary ===
methods = ["GP", "IDW", "NN", "RF", "FD", "PINN"]

print(f"\n{'='*100}")
print(f"  EXP 16: Multi-Seed PINN ({N_SEEDS} seeds) + FD Baseline")
print(f"  {len(all_results)} experiments across {len(reef_configs)} reefs, {elapsed/60:.1f} min total")
print(f"{'='*100}")

for reef_cfg in reef_configs:
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
            if m_ == "PINN":
                print(f"{'PINN (mean±std)':>20}  ", end="")
            else:
                print(f"{m_:>10}  ", end="")
        print("Best")
        print("  " + "─" * 105)
        for r in sorted(subset, key=lambda x: -x["n_training_depths"]):
            n = r["n_training_depths"]
            print(f"  {n:>8}  ", end="")
            for m_ in methods:
                if m_ == "PINN":
                    mean = r.get("PINN_rmse_mean")
                    std = r.get("PINN_rmse_std")
                    if mean is not None and std is not None:
                        print(f"{mean:>8.4f}±{std:<8.4f}  ", end="")
                    else:
                        print(f"{'':>20}  ", end="")
                else:
                    v = r.get(f"{m_}_rmse")
                    print(f"{v:>10.4f}  " if v is not None else "       -    ", end="")
            print(r.get("best", "-"))

# Cross-reef summary
print(f"\n  === Cross-Reef Summary ===")
print(f"  {'Reef':<20} {'Holdout':>8} {'N':>3}  ", end="")
for m_ in methods:
    if m_ == "PINN":
        print(f"{'PINN (mean±std)':>20}  ", end="")
    else:
        print(f"{m_:>10}  ", end="")
print("Best")
print("  " + "─" * 120)
for r in all_results:
    print(f"  {r['reef']:<20} {r['holdout_depth']:>6.1f}m {r['n_training_depths']:>3}  ", end="")
    for m_ in methods:
        if m_ == "PINN":
            mean = r.get("PINN_rmse_mean")
            std = r.get("PINN_rmse_std")
            if mean is not None and std is not None:
                print(f"{mean:>8.4f}±{std:<8.4f}  ", end="")
            else:
                print(f"{'':>20}  ", end="")
        else:
            v = r.get(f"{m_}_rmse")
            print(f"{v:>10.4f}  " if v is not None else "       -    ", end="")
    print(r.get("best", "-"))

# PINN variability analysis
print(f"\n  === PINN Seed Variability ===")
for r in all_results:
    rmses = r.get("PINN_rmse_all", [])
    if rmses:
        cv = np.std(rmses) / np.mean(rmses) * 100 if np.mean(rmses) > 0 else 0
        print(f"  {r['reef']:<20} {r['holdout_depth']:.1f}m / {r['n_training_depths']}d: "
              f"{np.mean(rmses):.4f} ± {np.std(rmses):.4f} "
              f"(CV={cv:.1f}%, range={min(rmses):.4f}–{max(rmses):.4f})")

# FD vs PINN comparison
print(f"\n  === FD vs PINN Comparison ===")
fd_wins = 0
pinn_wins = 0
for r in all_results:
    fd = r.get("FD_rmse")
    pinn = r.get("PINN_rmse_mean")
    if fd is not None and pinn is not None:
        winner = "FD" if fd < pinn else "PINN"
        if winner == "FD":
            fd_wins += 1
        else:
            pinn_wins += 1
        print(f"  {r['reef']:<20} {r['holdout_depth']:.1f}m / {r['n_training_depths']}d: "
              f"FD={fd:.4f}, PINN={pinn:.4f} → {winner}")
print(f"\n  FD wins: {fd_wins}/{fd_wins+pinn_wins}, "
      f"PINN wins: {pinn_wins}/{fd_wins+pinn_wins}")

# Learned parameters across seeds
print(f"\n  === Learned Parameters (mean across seeds) ===")
for r in all_results:
    params_all = r.get("PINN_params_all", [])
    if params_all:
        kappas = [p.get("kappa", 0) for p in params_all]
        Kds = [p.get("Kd", 0) for p in params_all if p.get("Kd") is not None]
        print(f"  {r['reef']:<20} {r['holdout_depth']:.1f}m / {r['n_training_depths']}d: "
              f"κ={np.mean(kappas):.6f}±{np.std(kappas):.6f}, "
              f"Kd={np.mean(Kds):.4f}±{np.std(Kds):.4f}" if Kds else "")

print(f"\nTotal wall time: {elapsed/60:.1f} min")
