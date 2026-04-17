"""Train a PINN on a synthetic reef, CPU-executable in under five minutes.

This example generates a small synthetic dataset from a known thermal
diffusivity (kappa) and light attenuation (Kd), trains a PINN on three
logger depths plus the surface SST series, and compares the recovered
physical parameters against ground truth.

Run with::

    python examples/train_on_new_reef.py

On CPU with ``pip install -e .[cpu]``, a full run takes roughly two to four
minutes and uses well under one gigabyte of RAM. No GPU required.

The plot ``synthetic_depth_profile.png`` is saved next to this script.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# Force CPU before JAX is imported so this example is reproducible on any machine.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinn_reef_thermal import (
    ReefPINN,
    ReefThermalPhysics,
    generate_stratified_collocation_points,
)

# --- Ground-truth physics -------------------------------------------------
TRUE_KAPPA = 3.5e-4   # m^2 / s
TRUE_KD = 0.12        # 1/m
LOGGER_DEPTHS = [2.0, 8.0, 15.0]  # three logger depths (surface is z=0)
Z_MAX = 20.0
N_DAYS = 180
DT_HOURS = 1.0


def generate_synthetic_reef(seed: int = 0) -> dict:
    """Solve the 1D heat equation with TRUE_KAPPA, TRUE_KD and sample loggers."""
    physics = ReefThermalPhysics(
        kappa=TRUE_KAPPA, Kd=TRUE_KD,
        T_mean=28.2, T_amp=1.0,
        z_max=Z_MAX, t_days=N_DAYS,
        Q_max=350.0, rho_cp=4.1e6, utc_offset=10.0,
    )
    z_fd, t_fd, T_fd = physics.solve_fd(nz=128, dt=DT_HOURS * 3600.0)

    rng = np.random.default_rng(seed)

    # Surface boundary condition: satellite SST at daily cadence.
    t_bc_days = np.arange(0, N_DAYS, 1.0) + 0.17  # 04:00 UTC, matches CRW cadence
    t_bc = (t_bc_days * 86400.0).astype(np.float32)
    T_bc = np.interp(t_bc, t_fd, T_fd[0]).astype(np.float32)
    T_bc = T_bc + rng.normal(0.0, 0.05, size=T_bc.shape).astype(np.float32)

    # Logger observations at three depths, hourly, with mild noise.
    z_data_list, t_data_list, T_data_list = [], [], []
    for depth in LOGGER_DEPTHS:
        iz = int(np.argmin(np.abs(z_fd - depth)))
        t_hr = t_fd.astype(np.float32)
        T_hr = T_fd[iz] + rng.normal(0.0, 0.02, size=t_hr.shape).astype(np.float32)
        z_data_list.append(np.full_like(t_hr, depth, dtype=np.float32))
        t_data_list.append(t_hr)
        T_data_list.append(T_hr)
    z_data = np.concatenate(z_data_list)
    t_data = np.concatenate(t_data_list)
    T_data = np.concatenate(T_data_list)

    metadata = {
        "z_max": float(Z_MAX),
        "t_max": float(N_DAYS * 86400.0),
        "T_mean": float(np.mean(T_data)),
        "T_range": (float(T_data.min()), float(T_data.max())),
        "t_start_utc_hour": 0.0,
        "start_day_of_year": 0.0,
    }
    return dict(
        z_data=z_data, t_data=t_data, T_data=T_data,
        z_bc=np.zeros_like(t_bc), t_bc=t_bc, T_bc=T_bc,
        depths={f"{d:.1f}m": d for d in LOGGER_DEPTHS},
        metadata=metadata,
        truth=dict(z=z_fd, t=t_fd, T=T_fd,
                   kappa=TRUE_KAPPA, Kd=TRUE_KD),
    )


def train_pinn(data: dict) -> ReefPINN:
    """Train a small PINN configuration that finishes on CPU in a few minutes."""
    meta = data["metadata"]
    physics = ReefThermalPhysics(
        kappa=5e-4, Kd=0.25,
        T_mean=meta["T_mean"], T_amp=1.0,
        z_max=meta["z_max"], t_days=meta["t_max"] / 86400.0,
        utc_offset=10.0,
    )
    z_pde, t_pde = generate_stratified_collocation_points(
        meta["z_max"], meta["t_max"], n_points=5000,
        logger_depths=list(data["depths"].values()), seed=0,
    )
    pinn = ReefPINN(
        physics=physics, learn_kappa=True, learn_Kd=True,
        use_hard_bc=True, n_fourier=4, seed=0,
        init_kappa=1e-3, init_Kd=0.2,
        pde_chunk_size=0, pde_depth_scale=20.0,
        w_bc_bottom=0.05, kappa_mode="constant",
        t_sst=data["t_bc"], T_sst=data["T_bc"],
        use_modified_mlp=True, hidden_dim=64, n_hidden=3,
        T_scale=2.0, activation="tanh", grad_clip=1.0,
        t_start_utc_hour=0.0, start_day_of_year=0.0,
    )
    pinn.train(
        z_pde=z_pde, t_pde=t_pde,
        z_data=data["z_data"], t_data=data["t_data"], T_data=data["T_data"],
        z_bc=data["z_bc"], t_bc=data["t_bc"], T_bc=data["T_bc"],
        n_epochs=2000, lr=1e-3, print_every=200,
        w_pde=1.0, w_data=10.0, w_bc=0.0,
    )
    return pinn


def plot_depth_profile(data: dict, pinn: ReefPINN, out_path: Path) -> None:
    """Compare the PINN reconstruction to ground truth at a mid-simulation snapshot."""
    truth = data["truth"]
    t_snap = truth["t"][len(truth["t"]) // 2]
    z_grid = np.linspace(0.0, Z_MAX, 64)
    T_pinn = np.array([float(pinn.predict(z, t_snap)) for z in z_grid])
    T_true = np.interp(z_grid, truth["z"], truth["T"][:, len(truth["t"]) // 2])

    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=140)
    ax.plot(T_true, z_grid, "k-", lw=2, label="ground truth")
    ax.plot(T_pinn, z_grid, "tab:orange", lw=2, ls="--", label="PINN")
    for d in LOGGER_DEPTHS:
        ax.axhline(d, color="tab:blue", alpha=0.3, lw=1)
    ax.invert_yaxis()
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Synthetic reef: depth profile at mid-simulation")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    print("Generating synthetic reef data...")
    data = generate_synthetic_reef()
    print(f"  Loggers: {len(data['z_data']):,} observations at "
          f"{len(LOGGER_DEPTHS)} depths")
    print(f"  SST BC: {len(data['z_bc']):,} samples")
    print(f"  True kappa = {TRUE_KAPPA:.3e} m^2/s, true Kd = {TRUE_KD:.3f} 1/m")

    print("\nTraining PINN for 2000 epochs (CPU-friendly configuration)...")
    pinn = train_pinn(data)

    est = pinn.get_estimated_params()
    print("\n=== Recovered vs ground truth ===")
    if "kappa" in est:
        rel = 100.0 * (est["kappa"] - TRUE_KAPPA) / TRUE_KAPPA
        print(f"  kappa: true = {TRUE_KAPPA:.3e}, recovered = {est['kappa']:.3e} "
              f"({rel:+.1f} %)")
    if "Kd" in est:
        rel = 100.0 * (est["Kd"] - TRUE_KD) / TRUE_KD
        print(f"  Kd:    true = {TRUE_KD:.3f},     recovered = {est['Kd']:.3f}     "
              f"({rel:+.1f} %)")

    out_path = Path(__file__).with_name("synthetic_depth_profile.png")
    plot_depth_profile(data, pinn, out_path)
    print(f"\nDepth profile plot saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
