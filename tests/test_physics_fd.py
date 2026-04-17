"""Tests for the finite-difference reference solver and the FD baseline."""

from __future__ import annotations

import numpy as np
import pytest

from pinn_reef_thermal import ReefThermalPhysics, fd_predict


def test_fd_solver_runs_without_divergence():
    phys = ReefThermalPhysics(z_max=15.0, t_days=2)
    z, t, T = phys.solve_fd(nz=32, dt=600.0)
    assert T.shape == (32, t.size)
    assert np.isfinite(T).all()
    assert T.min() > 20.0 and T.max() < 40.0


def test_fd_solver_dirichlet_surface_exact():
    phys = ReefThermalPhysics(z_max=15.0, t_days=1)
    z, t, T = phys.solve_fd(nz=32, dt=600.0)
    expected = phys.surface_temperature(t)
    # Skip t=0: the initial condition is T_mean (isothermal), not the
    # surface BC. All subsequent Dirichlet updates must be enforced exactly.
    assert np.allclose(T[0, 1:], expected[1:], atol=1e-10)


def test_fd_baseline_interpolation_sanity():
    """fd_predict should reproduce the surface SST when queried at z=0."""
    t_sst = np.linspace(0, 2 * 86400, 49)
    T_sst = 28.0 + 0.5 * np.sin(2 * np.pi * t_sst / 86400.0)
    meta = dict(z_max=10.0, t_max=float(t_sst[-1]), t_start_utc_hour=0.0)
    holdout = dict(
        z=np.zeros(5, dtype=np.float32),
        t=np.linspace(3600.0, t_sst[-1] - 3600.0, 5).astype(np.float32),
        T=np.zeros(5, dtype=np.float32),
    )
    T_pred = fd_predict(holdout, t_sst.astype(np.float32), T_sst.astype(np.float32),
                        meta, nz=64, dt=1800.0)
    T_expected = np.interp(holdout["t"], t_sst, T_sst)
    assert np.allclose(T_pred, T_expected, atol=0.2)
