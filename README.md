# Depth-Resolved Coral Reef Thermal Fields from Satellite SST and Sparse In-Situ Loggers Using Physics-Informed Neural Networks

A physics-informed neural network framework for reconstructing depth-resolved
thermal fields on coral reefs from NOAA Coral Reef Watch satellite SST and
sparse in-situ temperature loggers. Python package name: `pinn-reef-thermal`.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19624895.svg)](https://doi.org/10.5281/zenodo.19624895)
[![test](https://github.com/alzayats/pinn-reef-thermal/actions/workflows/test.yml/badge.svg)](https://github.com/alzayats/pinn-reef-thermal/actions/workflows/test.yml)
[![docs](https://github.com/alzayats/pinn-reef-thermal/actions/workflows/docs.yml/badge.svg)](https://alzayats.github.io/pinn-reef-thermal/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![pip install](https://img.shields.io/badge/pip%20install-editable-orange.svg)](#installation)

---

## Overview

This repository accompanies the paper

> **Depth-Resolved Coral Reef Thermal Fields from Satellite SST and Sparse
> In-Situ Loggers Using Physics-Informed Neural Networks.** Saleh A. and
> Rahimi Azghadi M. *Environmental Modelling and Software*, 2026.

The framework embeds the one-dimensional vertical heat equation with
depth-attenuated solar heating as a soft constraint, enforces satellite SST
as a hard surface boundary condition, and jointly learns effective thermal
diffusivity (`kappa`) and light attenuation (`Kd`) from data. Headline
results across four Great Barrier Reef sites (Davies, Myrmidon, Rib, Kelso):

- **PINN outperforms the physics-only finite-difference baseline in 27 of
  30 configurations (90 percent)**, confirming that the neural network adds
  substantive value beyond the governing equations with fixed parameters.
- **Stable under sparsity**: PINN holdout RMSE remains near 0.27 C with
  only three training depths; statistical baselines collapse above 1.8 C.
- **Depth-resolved thermal stress**: satellite-only Degree Heating Days
  overestimate subsurface bleaching stress by 75 percent at 9 m at Rib
  Reef during the 2020 and 2022 mass bleaching events.
- **Low initialisation sensitivity**: five-seed replicates give median
  coefficient of variation near 1 percent across 30 configurations.

Full experimental details in the manuscript; the trained models and
pre-computed results in this repository reproduce every figure and table.

---

## Installation

### CPU path (laptop or CI, no GPU required)

```bash
git clone https://github.com/alzayats/pinn-reef-thermal.git
cd pinn-reef-thermal
pip install -e ".[cpu]"
```

The CPU install uses `jax[cpu]` and resolves on Linux, macOS, and Windows.
The CPU-executable synthetic example (`examples/train_on_new_reef.py`)
runs in two to four minutes on a laptop. Full paper reproduction requires
a GPU; see below.

### GPU path (full paper reproduction, 8 GB+ VRAM)

```bash
git clone https://github.com/alzayats/pinn-reef-thermal.git
cd pinn-reef-thermal
pip install -e ".[gpu]"
```

The GPU install uses `jax[cuda12]` and requires CUDA 12. An RTX 4090 or
equivalent reproduces the paper's 150-run multi-seed experiment in
approximately 13 hours; memory footprint is roughly 8 GB VRAM.

### Conda alternative

```bash
conda env create -f environment.yml
conda activate pinn-reef-thermal
pip install -e .
```

Equivalent to the GPU path; kept for users who prefer conda for their CUDA
toolchain.

---

## Using this framework on your own reef

Two CPU-executable examples ship with the package.

### Example 1: synthetic reef

`examples/train_on_new_reef.py` generates a small synthetic dataset from
known `kappa` and `Kd`, trains a PINN on three logger depths plus a
surface SST series, and prints the recovered physical parameters alongside
the ground-truth values.

```bash
python examples/train_on_new_reef.py
```

Runs in under five minutes on CPU. Writes `synthetic_depth_profile.png`
next to the script.

There is also a Jupyter twin at `examples/train_on_new_reef.ipynb`.

### Example 2: adapt your own logger and SST CSVs

`examples/extend_to_new_reef.py` shows how to convert existing logger and
satellite SST CSVs to the package's expected schema and produce PINN-ready
training arrays in one line of code.

Expected schemas:

| File     | Columns                                           |
|----------|---------------------------------------------------|
| Loggers  | `time` (datetime UTC), `depth` (float m), `temperature` (float C) |
| SST      | `time` (datetime UTC), `CRW_SST` (float C)        |

Once your CSVs are in this form, the full workflow is:

```python
import pandas as pd
from pinn_reef_thermal import prepare_pinn_data, fit_pinn

loggers = pd.read_csv("my_loggers.csv", parse_dates=["time"])
sst = pd.read_csv("my_sst.csv", parse_dates=["time"])
reef_data = prepare_pinn_data(loggers, sst_df=sst, use_hourly=True)

pinn, history = fit_pinn(reef_data, n_epochs=5000)
print(pinn.get_estimated_params())
```

Or via the CLI:

```bash
pinn-reef train \
  --loggers my_loggers.csv \
  --sst my_sst.csv \
  --output ./my_reef_outputs \
  --epochs 5000
```

Downloading CRW satellite SST for your reef's coordinates is automated:

```python
from pinn_reef_thermal import download_crw_sst
sst = download_crw_sst(
    lat=-18.83, lon=147.63,
    date_from="2024-11-01", date_to="2025-02-28",
)
```

---

## Data for reproducing the paper

The paper's experiments rely on the AIMS temperature logger archive. The
raw CSVs are approximately 8.5 GB and must be downloaded separately.

### Automated

```bash
python scripts/download_data.py
```

This fetches CRW satellite SST from NOAA ERDDAP automatically. For AIMS
logger data it prints instructions for manual download from the
[AIMS Time Series Explorer](https://apps.aims.gov.au/ts-explorer/).

### Manual

1. Visit <https://apps.aims.gov.au/ts-explorer/>.
2. Download temperature logger CSVs for Davies, Myrmidon, Rib, and Kelso
   reefs.
3. Place all CSVs in `data/ALZ/`.

The sub-daily loader
(`pinn_reef_thermal.load_reef_subdaily`) handles both the Weather Station
(30-column) and Temperature Logger (19-column) AIMS schemas.

---

## Reproducing the paper

### Pre-computed results (no GPU required)

The `results/` directory contains the exact JSON outputs from the paper's
experiments:

- `exp14_results.json`: 30 holdout validation experiments.
- `exp15_results.json`: depth-resolved DHD profiles for four reefs.
- `exp16_results.json`: 150 PINN runs (30 configurations x 5 seeds) plus
  30 finite-difference baseline runs.

Regenerate every paper figure from these JSONs on CPU:

```bash
python scripts/make_figures.py --skip-map
```

Figures land in `figures/` as both PNG and PDF.

### Re-running experiments (GPU required)

```bash
# All experiments (~13 hours on RTX 4090)
python scripts/run_experiments.py

# Or individually:
python scripts/run_experiments.py --exp 14   # holdout validation ~2h
python scripts/run_experiments.py --exp 15   # thermal stress    ~1h
python scripts/run_experiments.py --exp 16   # multi-seed + FD  ~10h
```

| Experiment | Description                                               |
|------------|-----------------------------------------------------------|
| Exp 14     | Leave-one-depth-out validation, 4 reefs, 5 sparsity levels |
| Exp 15     | Depth-resolved DHD bleaching stress profiles              |
| Exp 16     | 5-seed uncertainty + physics-only finite-difference baseline |

---

## Package layout

```
.
|-- README.md
|-- LICENSE                          MIT
|-- CHANGELOG.md                     Keep-a-Changelog
|-- CITATION.cff                     Software citation metadata
|-- pyproject.toml                   PEP 621 build + dependencies
|-- environment.yml                  conda alternative for GPU reproduction
|-- FAQ.md                           Common questions (data, hardware, portability)
|-- CONTRIBUTING.md                  How to run tests, lint, propose changes
|
|-- src/
|   `-- pinn_reef_thermal/           Core library, pip-installable
|       |-- __init__.py              Public API surface
|       |-- pinn.py                  PINN model (Modified MLP, hard BC, learned kappa/Kd)
|       |-- physics.py               1D heat equation, reference FD solver
|       |-- baselines.py             GP, IDW, NN, RF, FD, satellite baselines
|       |-- data_loader.py           Phase 1 loader (daily aggregates)
|       |-- data_loader_v2.py        Phase 2 loader (sub-daily AIMS CSVs, both schemas)
|       |-- metrics.py               compute_dhd, compute_mmm_threshold
|       |-- train.py                 fit_pinn convenience wrapper
|       `-- cli.py                   pinn-reef CLI entry point
|
|-- scripts/                         Experiment pipelines and figure generation
|   |-- download_data.py
|   |-- run_experiments.py
|   |-- exp14_holdout_validation.py
|   |-- exp15_thermal_stress.py
|   |-- exp16_multiseed.py
|   `-- make_figures.py
|
|-- examples/                        CPU-executable walkthroughs
|   |-- train_on_new_reef.py         Synthetic reef, recovers kappa and Kd
|   |-- train_on_new_reef.ipynb      Jupyter twin of the above
|   `-- extend_to_new_reef.py        Adapt your own logger CSVs
|
|-- tests/                           pytest smoke tests (CPU)
|-- results/                         Pre-computed JSON for every paper experiment
|-- docs/                            pdoc build target (hosted on GitHub Pages)
|-- data/                            populated by scripts/download_data.py
`-- .github/workflows/               CI for tests, lint, docs deployment
```

---

## Development

Run the test suite (under one minute on CPU):

```bash
pip install -e ".[cpu,dev]"
pytest tests/ -v
```

Check lint:

```bash
ruff check src/pinn_reef_thermal
```

Build the API docs locally:

```bash
pdoc -o site -d numpy pinn_reef_thermal
python -m http.server -d site
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor workflow.

---

## Citation

If you use this framework in your research, please cite both the paper and
the software:

```bibtex
@article{saleh2026pinn,
  author  = {Saleh, Alzayat and Rahimi Azghadi, Mostafa},
  title   = {Depth-Resolved Coral Reef Thermal Fields from Satellite SST
             and Sparse In-Situ Loggers Using Physics-Informed Neural Networks},
  journal = {Environmental Modelling \& Software},
  year    = {2026}
}

@software{saleh2026pinnreefsoftware,
  author  = {Saleh, Alzayat and Rahimi Azghadi, Mostafa},
  title   = {pinn-reef-thermal: Physics-Informed Neural Networks for
             Depth-Resolved Coral Reef Thermal Reconstruction},
  year    = {2026},
  version = {1.0.0},
  doi     = {10.5281/zenodo.19624895},
  url     = {https://github.com/alzayats/pinn-reef-thermal}
}
```

Machine-readable citation metadata is in [`CITATION.cff`](CITATION.cff).

---

## Acknowledgements

Temperature logger data from the Australian Institute of Marine Science
(AIMS) temperature logger network. Satellite SST from NOAA Coral Reef
Watch, accessed via CoastWatch ERDDAP.

Open access publishing for the accompanying paper was facilitated by
James Cook University, as part of the Elsevier - James Cook University
agreement via the Council of Australasian University Librarians.

---

## License

MIT. See [LICENSE](LICENSE).
