# Frequently asked questions

## What is the minimum number of loggers I need to run this framework on my own reef?

Technically two depths is enough for the PINN to run without errors, but
holdout accuracy degrades sharply below three. The paper evaluates sparsity
from 17 depths down to 2; at 2 training depths the coefficient of variation
across seeds grows to 14 percent and the extrapolation to the holdout
depth has RMSE up to 0.75 C. At 3 or more training depths, performance
is stable and RMSE is typically 0.25 to 0.5 C.

**Practical recommendation**: plan for three or more logger depths
spanning the reef's vertical range. For a reef with loggers at 2 m, 8 m,
and 15 m, the framework can reliably reconstruct temperatures at 5 m, 11 m,
or any other depth in between.

## Can I use this without AIMS data?

Yes. The data loaders expect two CSV files, not the specific AIMS
schemas:

- Logger CSV: `time` (datetime UTC), `depth` (float m),
  `temperature` (float C).
- SST CSV: `time` (datetime UTC), `CRW_SST` (float C).

See `examples/extend_to_new_reef.py` for a worked example that adapts
arbitrary logger exports (SBE56, HOBO Pendant, RBR Solo, thermistor
strings) to this schema. The `pinn_reef_thermal.load_reef_subdaily`
function is specific to AIMS; `pinn_reef_thermal.prepare_pinn_data`
accepts any DataFrame in the expected schema.

Satellite SST is recommended for the surface boundary condition.
Alternatives (in-situ surface logger, buoy SST, modelled SST) will work
if they sample the surface at daily or finer cadence. To download Coral
Reef Watch SST for your reef:

```python
from pinn_reef_thermal import download_crw_sst
sst = download_crw_sst(
    lat=-18.83, lon=147.63,
    date_from="2024-11-01", date_to="2025-02-28",
)
```

## Does it work on non-Great-Barrier-Reef systems?

The framework is transferable in principle: the 1D vertical heat equation
is universal, and `kappa` and `Kd` are reef-specific parameters learned
from data.

Caveats for reefs outside the GBR:

1. **Turbidity and water clarity** vary widely. At Caribbean reefs with
   resuspended sediments, `Kd` may be 5 to 10 times higher than at the
   GBR outer shelf. The learned `Kd` will absorb this, but identifiability
   degrades when `Kd` is very large because light does not penetrate far
   enough to help constrain `kappa`.
2. **Tidal and internal-wave mixing** on reefs with strong currents
   (Indonesian Throughflow, eastern Pacific reef slopes) violates the 1D
   assumption. The framework still produces temperature estimates, but
   the learned `kappa` absorbs lateral and vertical processes that are
   not purely diffusive.
3. **Latitude and solar forcing**: the `utc_offset` parameter of
   `ReefThermalPhysics` must match your reef's local time to align the
   solar source term. Default is 10, for Australian Eastern Standard
   Time.

We recommend validating on a held-out depth before trusting the
reconstruction at unmonitored depths.

## Why JAX and not PyTorch?

Three reasons.

First, PINNs require second-order derivatives of the network output with
respect to its inputs (`d^2 T / d z^2`). JAX's `grad(grad(f))` composition
is clean, efficient, and captures exactly once into the JIT trace. The
PyTorch equivalent (`torch.autograd.functional.hessian` or nested
`grad()` calls) is workable but involves more boilerplate and more
opaque control flow.

Second, functional-style parameter trees map naturally onto PINN training,
where we want per-parameter learning rates for network weights versus the
learned physical parameters `kappa` and `Kd`. JAX's `jax.tree.map` handles
this in a handful of lines.

Third, `jit` compilation of the full loss-plus-gradient pass in a single
trace gives a five- to ten-fold speedup on GPU versus an eager PyTorch
training loop.

A PyTorch backend is on the roadmap (see CHANGELOG and issue tracker),
but the v1 release ships only the JAX implementation to keep experimental
reproducibility tight.

## How long does training take, and what hardware do I need?

| Scenario                                              | Time         | Hardware                 |
|-------------------------------------------------------|--------------|--------------------------|
| Synthetic CPU example (2000 epochs, hidden_dim=64)    | 2-4 min      | Any laptop               |
| Single reef, paper settings (15000 epochs)            | 5-15 min     | RTX 4090 or equivalent   |
| Full paper reproduction (150 runs + 30 FD baselines)  | approx. 13 h | RTX 4090                 |

The paper's models use approximately 8 GB of VRAM; JAX preallocates 75
percent of the GPU memory by default, which can make `nvidia-smi`
readings misleading. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` if you
want an accurate memory reading, at a modest speed penalty.

## Why does my recovered `kappa` disagree with the literature?

Two reasons. First, `kappa` is an effective turbulent diffusivity, not a
molecular one. It absorbs effects from wind-driven mixing, convective
overturning at night, and any sub-grid processes not explicitly
represented in the 1D model; literature estimates vary by two orders of
magnitude across reef environments for this reason. Second, `kappa` is
harder to identify than `Kd` from surface-and-depth observations.

Kd recovery is typically good to within 10 to 20 percent. Kappa
recovery is typically within a factor of two to five of the reef-specific
ground truth when one exists. The synthetic example in
`examples/train_on_new_reef.py` demonstrates this pattern: Kd comes back
cleanly; kappa is noisier.

## Does the framework output uncertainty?

Not yet in v1. The paper's multi-seed experiments (5 seeds per
configuration) are the closest thing: coefficient of variation across
seeds is under 2 percent for most holdout configurations, so point
estimates are stable.

A probabilistic output layer (multi-seed ensembling or Monte Carlo
dropout) is on the roadmap. See the Discussion section of the
manuscript for discussion of how this would address the systematic DHD
underestimation at shallow depths.

## Can I use this operationally?

The v1 release is research-quality. It has been used to produce the
paper's results; the CLI and CI pipeline are designed to support
automated retraining, but this has not been tested in a production
context. The Discussion of the manuscript sketches an operational
deployment: a single training run per reef, triggered seasonally or
annually, with daily inference against updated CRW SST.

## Where are the pre-trained models?

The repository ships pre-computed experimental outputs
(`results/exp*_results.json`) but does not ship trained PINN weights.
This is because `kappa` and `Kd` vary substantially by reef and by
observation window: a PINN trained on Davies Reef 2011-2014 does not
transfer well to Myrmidon Reef 2020-2024 without retraining. Training
on a consumer GPU takes 5-15 minutes per reef, so on-demand retraining
is cheap.

If you want pre-trained weights for a specific reef-year pair, open an
issue with the request.
