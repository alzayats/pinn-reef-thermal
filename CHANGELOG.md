# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-17

Initial public release accompanying the Environmental Modelling and Software
submission "Depth-Resolved Coral Reef Thermal Fields from Satellite SST and
Sparse In-Situ Loggers Using Physics-Informed Neural Networks" (Saleh and
Rahimi Azghadi, 2026).

This release provides the exact code used to produce the results reported in
the paper. It is preserved for reproducibility; future feature additions will
appear in subsequent minor releases.

### Added

- **Physics-informed neural network framework** (`src/pinn.py`)
  - Modified multi-layer perceptron with multiplicative gating and Fourier
    feature encoding.
  - Hard surface boundary condition enforcing satellite SST exactly at depth
    zero.
  - Joint learning of effective thermal diffusivity and light attenuation.
  - Curriculum-weighted physics loss with gradient clipping.
- **Reef thermal physics module** (`src/physics.py`)
  - One-dimensional vertical heat equation residual with depth-attenuated
    solar heating.
- **Baseline methods** (`src/baselines.py`)
  - Gaussian process regression.
  - Inverse distance weighting.
  - Nearest neighbour interpolation.
  - Random forest regression.
  - Physics-only finite-difference solver with literature-value parameters.
  - Satellite-only reference.
- **Data loaders** (`src/data_loader.py`, `src/data_loader_v2.py`)
  - Phase 1 loader for daily aggregated NOAA Coral Reef Watch SST and AIMS
    logger data.
  - Phase 2 loader for sub-daily AIMS archive CSVs, supporting both the
    19-column logger schema and the 30-column weather station schema.
  - Depth extraction from metadata columns or filename `@Xm` convention.
- **Experiment pipelines** (`scripts/`)
  - `exp14_holdout_validation.py`: 30 leave-one-depth-out holdout experiments
    across four Great Barrier Reef sites (Davies, Myrmidon, Rib, Kelso) at
    five sparsity levels.
  - `exp15_thermal_stress.py`: depth-resolved Degree Heating Day profiles
    comparing PINN predictions, satellite SST, and logger observations.
  - `exp16_multiseed.py`: five-seed uncertainty quantification (150 PINN
    runs) plus physics-only finite-difference baseline comparison.
  - `run_experiments.py`: single entry point running all three experiment
    pipelines.
  - `download_data.py`: automated acquisition of CRW satellite SST and
    instructions for manual AIMS logger downloads.
- **Publication figures** (`scripts/make_figures.py`)
  - Six Nature-quality figures rendered in both PNG and PDF.
  - Okabe-Ito colour palette, Arial 8 pt, pdf.fonttype 42 for vector text.
- **Pre-computed results** (`results/`)
  - `exp14_results.json`: 30 holdout validation experiments.
  - `exp15_results.json`: depth-resolved DHD profiles for four reefs.
  - `exp16_results.json`: 150 PINN runs plus 30 finite-difference baseline
    runs.
  - Enables figure regeneration and result inspection without re-running
    training on a GPU.
- **Repository infrastructure**
  - MIT licence.
  - Conda environment specification (`environment.yml`) for GPU
    reproduction with JAX CUDA 12.
  - README with installation, data setup, quick start, and full
    reproduction instructions.
  - CITATION.cff for software citation metadata.

### Known limitations in this release

- Full reproduction of the 150-run multi-seed experiment requires a GPU with
  at least 8 GB VRAM; approximately 13 hours on an RTX 4090.
- The package is installed via `conda env create`; a pip-installable
  distribution with a CPU fallback is planned for v1.1.0.
- Pre-computed figures can be regenerated from the JSON results without a
  GPU, but training new PINNs on a CPU is not currently supported in this
  release.

[1.0.0]: https://github.com/alzayats/pinn-reef-thermal/releases/tag/v1.0.0
