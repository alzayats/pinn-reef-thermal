"""End-to-end smoke tests for ReefPINN on synthetic data.

These tests force CPU via ``JAX_PLATFORMS=cpu`` in ``conftest.py`` and run a
severely truncated training loop (ten epochs at hidden_dim=32). They exist to
verify the model constructs, the forward pass and loss are finite, the hard
boundary condition is honoured, and optimiser steps do not introduce NaN or
inf. They are not expected to produce physically accurate reconstructions.
"""

from __future__ import annotations

import numpy as np
import pytest

from pinn_reef_thermal import (
    ReefPINN,
    ReefThermalPhysics,
    generate_stratified_collocation_points,
)


def _make_toy_dataset():
    rng = np.random.default_rng(0)
    z_data = np.tile(np.array([2.0, 6.0, 12.0], dtype=np.float32), 96)
    t_data = np.repeat(np.linspace(0, 5 * 86400, 96, dtype=np.float32), 3)
    T_data = 28.0 + 0.3 * np.sin(2 * np.pi * t_data / 86400.0) - 0.05 * z_data
    T_data = T_data.astype(np.float32) + rng.normal(0, 0.02, size=T_data.shape).astype(np.float32)

    t_bc = np.linspace(0, 5 * 86400, 48, dtype=np.float32)
    T_bc = (28.0 + 0.4 * np.sin(2 * np.pi * t_bc / 86400.0)).astype(np.float32)

    physics = ReefThermalPhysics(z_max=15.0, t_days=5, T_mean=28.0)
    z_pde, t_pde = generate_stratified_collocation_points(
        physics.z_max, physics.t_max, 256, logger_depths=[2.0, 6.0, 12.0], seed=0
    )
    return physics, z_data, t_data, T_data, t_bc, T_bc, z_pde, t_pde


def _build_pinn(physics, t_bc, T_bc, *, hidden_dim=32, n_hidden=2):
    return ReefPINN(
        physics=physics,
        learn_kappa=True, learn_Kd=True,
        use_hard_bc=True,
        n_fourier=2, seed=0,
        init_kappa=1e-3, init_Kd=0.2,
        pde_chunk_size=0,
        t_sst=t_bc, T_sst=T_bc,
        use_modified_mlp=True,
        hidden_dim=hidden_dim, n_hidden=n_hidden,
        T_scale=2.0, activation="tanh",
        grad_clip=1.0,
    )


def test_pinn_trains_without_nan():
    physics, z_data, t_data, T_data, t_bc, T_bc, z_pde, t_pde = _make_toy_dataset()
    pinn = _build_pinn(physics, t_bc, T_bc)
    history = pinn.train(
        z_pde=z_pde, t_pde=t_pde,
        z_data=z_data, t_data=t_data, T_data=T_data,
        z_bc=np.zeros_like(t_bc), t_bc=t_bc, T_bc=T_bc,
        n_epochs=10, lr=1e-3, print_every=1000,
        w_pde=1.0, w_data=10.0, w_bc=0.0,
    )
    assert np.isfinite(history["total"]).all()
    assert np.isfinite(history["data"]).all()
    assert np.isfinite(history["pde"]).all()


def test_hard_bc_exact_at_surface():
    """T(0, t) must equal the interpolated satellite SST to machine precision."""
    physics, z_data, t_data, T_data, t_bc, T_bc, z_pde, t_pde = _make_toy_dataset()
    pinn = _build_pinn(physics, t_bc, T_bc)
    # Need to train at least one step so trained_params is set.
    pinn.train(
        z_pde=z_pde, t_pde=t_pde,
        z_data=z_data, t_data=t_data, T_data=T_data,
        z_bc=np.zeros_like(t_bc), t_bc=t_bc, T_bc=T_bc,
        n_epochs=2, lr=1e-3, print_every=1000,
        w_pde=1.0, w_data=10.0, w_bc=0.0,
    )
    for t_query in t_bc[::8]:
        T_pred = float(pinn.predict(0.0, t_query))
        T_expected = float(np.interp(t_query, t_bc, T_bc))
        assert abs(T_pred - T_expected) < 1e-4, (t_query, T_pred, T_expected)


def test_pinn_predictions_are_finite():
    physics, z_data, t_data, T_data, t_bc, T_bc, z_pde, t_pde = _make_toy_dataset()
    pinn = _build_pinn(physics, t_bc, T_bc)
    pinn.train(
        z_pde=z_pde, t_pde=t_pde,
        z_data=z_data, t_data=t_data, T_data=T_data,
        z_bc=np.zeros_like(t_bc), t_bc=t_bc, T_bc=T_bc,
        n_epochs=5, lr=1e-3, print_every=1000,
        w_pde=1.0, w_data=10.0, w_bc=0.0,
    )
    z_grid = np.linspace(0, physics.z_max, 16)
    t_grid = np.linspace(0, physics.t_max, 16)
    field = pinn.predict_field(z_grid, t_grid)
    assert np.isfinite(field).all()
    assert 20.0 < field.min() and field.max() < 36.0
