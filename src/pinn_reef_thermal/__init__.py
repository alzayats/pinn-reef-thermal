"""pinn_reef_thermal: Physics-Informed Neural Networks for Depth-Resolved Coral Reef Thermal Reconstruction.

This package implements a physics-informed neural network framework for reconstructing
depth-resolved thermal fields on coral reefs from satellite sea surface temperature
(NOAA Coral Reef Watch) and sparse in-situ temperature logger observations (AIMS
temperature logger program, or comparable networks).

The framework embeds the one-dimensional vertical heat equation with depth-attenuated
solar heating as a soft constraint, enforces satellite SST as a hard surface boundary
condition at depth zero, and jointly learns effective thermal diffusivity (kappa) and
light attenuation (Kd) from data.

Public API
----------
Core framework

- :class:`PINN` (alias for :class:`ReefPINN`) - the physics-informed neural network.
- :func:`fit_pinn` - convenience wrapper that constructs a PINN from pre-prepared
  training arrays and returns the trained model and training history.
- :class:`ReefThermalPhysics` - physical constants and the reference finite-difference
  solver for the 1D vertical heat equation.

Data loading

- :func:`load_reef_data` - Phase 1 loader for daily aggregated AIMS CSVs plus CRW SST
  (used for the paper's daily-aggregate results).
- :func:`load_reef_subdaily` - Phase 2 loader for the full sub-daily AIMS archive,
  handling both the 19-column logger and 30-column weather station schemas.
- :func:`prepare_pinn_data` - convert a loaded sub-daily DataFrame into PINN training
  arrays with optional hourly aggregation.
- :func:`prepare_holdout`, :func:`prepare_holdout_v2` - leave-one-depth-out splits
  for holdout validation.
- :func:`generate_collocation_points`, :func:`generate_stratified_collocation_points` -
  sample PDE collocation points, optionally concentrated near logger depths.
- :func:`aggregate_hourly`, :func:`find_overlap_window` - sub-daily utilities.
- :func:`download_crw_sst`, :data:`REEF_COORDS` - acquire Coral Reef Watch SST from
  NOAA ERDDAP for a location and date range.

Baselines

- :func:`gp_predict`, :func:`idw_predict`, :func:`nn_predict`, :func:`rf_predict` -
  statistical baselines (Gaussian process, inverse distance weighting, nearest
  neighbour, random forest).
- :func:`fd_predict` - physics-only finite-difference baseline with literature
  parameters.
- :func:`satellite_predict` - satellite-only reference (treats SST as depth-uniform).
- :func:`run_all_baselines` - run every baseline on a training/holdout split and
  return results as a dict.

Thermal stress metrics

- :func:`compute_dhd` - cumulative Degree Heating Days above a bleaching threshold.
- :func:`compute_mmm_threshold` - Maximum of Monthly Means plus one degree Celsius.

See the ``examples/`` directory for end-to-end workflows, including a CPU-executable
synthetic example (``examples/train_on_new_reef.py``) that generates data with known
thermal parameters, trains a PINN, and compares recovered kappa and Kd against ground
truth.
"""

from __future__ import annotations

from .baselines import (
    fd_predict,
    gp_predict,
    idw_predict,
    nn_predict,
    rf_predict,
    run_all_baselines,
    satellite_predict,
)
from .data_loader import (
    generate_collocation_points,
    generate_stratified_collocation_points,
    load_reef_data,
    prepare_holdout,
)
from .data_loader_v2 import (
    REEF_COORDS,
    REEF_PATTERNS,
    aggregate_hourly,
    download_crw_sst,
    find_overlap_window,
    load_reef_subdaily,
    prepare_holdout_v2,
    prepare_pinn_data,
)
from .metrics import compute_dhd, compute_mmm_threshold
from .physics import ReefThermalPhysics
from .pinn import ReefPINN
from .train import fit_pinn

__version__ = "1.0.0"

# Alias exposing the brief's public API name. Kept after imports so ruff does
# not complain about mid-module assignment, and near the end of the file so
# the aliasing is easy to find during refactors.
PINN = ReefPINN

__all__ = [
    "__version__",
    # Core
    "PINN",
    "ReefPINN",
    "ReefThermalPhysics",
    "fit_pinn",
    # Data (Phase 1)
    "load_reef_data",
    "generate_collocation_points",
    "generate_stratified_collocation_points",
    "prepare_holdout",
    # Data (Phase 2)
    "load_reef_subdaily",
    "prepare_pinn_data",
    "prepare_holdout_v2",
    "aggregate_hourly",
    "find_overlap_window",
    "download_crw_sst",
    "REEF_COORDS",
    "REEF_PATTERNS",
    # Baselines
    "gp_predict",
    "idw_predict",
    "nn_predict",
    "rf_predict",
    "fd_predict",
    "satellite_predict",
    "run_all_baselines",
    # Metrics
    "compute_dhd",
    "compute_mmm_threshold",
]
