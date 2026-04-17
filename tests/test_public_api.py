"""Smoke tests that the package exposes the documented public API."""

from __future__ import annotations

import pinn_reef_thermal as prt


def test_version_exposed():
    assert isinstance(prt.__version__, str)
    assert prt.__version__.count(".") >= 2


def test_all_names_importable():
    for name in prt.__all__:
        assert hasattr(prt, name), f"Public name {name!r} is listed in __all__ but missing"


def test_pinn_alias():
    assert prt.PINN is prt.ReefPINN


def test_required_api_present():
    """The brief mandates these names at minimum."""
    required = {
        "PINN",
        "fit_pinn",
        "load_reef_data",
        "compute_dhd",
        # baseline functions
        "gp_predict",
        "idw_predict",
        "nn_predict",
        "rf_predict",
        "fd_predict",
        "satellite_predict",
        "run_all_baselines",
    }
    missing = required - set(prt.__all__)
    assert not missing, f"Missing required public names: {missing}"
