# Contributing to pinn-reef-thermal

Thank you for considering a contribution. This project ships the research
code accompanying an Environmental Modelling and Software submission, so
stability and reproducibility of the version-1 results take precedence over
aggressive refactoring. Bug fixes, documentation improvements, new
examples, and carefully-scoped features are all welcome.

If you are unsure whether a change is in scope, open an issue first with
`[discussion]` in the title; we prefer the conversation before the pull
request.

---

## Development setup

Clone the repository and install in editable mode with the CPU and dev
extras:

```bash
git clone https://github.com/alzayats/pinn-reef-thermal.git
cd pinn-reef-thermal
pip install -e ".[cpu,dev]"
```

CPU is sufficient for running the test suite and the synthetic example.
For full paper reproduction you need a CUDA 12 GPU; swap `.[cpu]` for
`.[gpu]`.

---

## Running tests and lint

```bash
pytest tests/ -v
ruff check src/pinn_reef_thermal
```

Both commands are enforced by CI on Python 3.10 and 3.11. Pull requests
that break either check will not be merged.

The test suite takes under a minute on CPU. It includes:

- `test_public_api.py`: verifies all names in the documented public API
  are importable and that `PINN` aliases `ReefPINN`.
- `test_metrics.py`: numerical correctness of `compute_dhd` and
  `compute_mmm_threshold`.
- `test_physics_fd.py`: reference finite-difference solver and the
  `fd_predict` physics-only baseline.
- `test_data_loader_schemas.py`: both the Weather Station (30-column)
  and Temperature Logger (19-column) AIMS CSV formats.
- `test_pinn_smoke.py`: end-to-end PINN training on synthetic data (ten
  epochs, CPU). Asserts no NaN or inf, and that the hard boundary
  condition is enforced exactly at z = 0.

New features should ship with tests in the same style.

---

## Style

- The `ruff` configuration in `pyproject.toml` is authoritative. Run
  `ruff check src/pinn_reef_thermal --fix` before pushing.
- Australian English in docstrings and comments: modelling, behaviour,
  colour, centre.
- No em dashes anywhere (commas, semicolons, colons instead).
- NumPy-style docstrings on every public function, class, and method.
  pdoc consumes these to generate the hosted API documentation.
- Type hints on public signatures where they clarify intent.

---

## Commit and pull-request guidance

- One logical change per commit. Conventional Commit prefixes (`feat:`,
  `fix:`, `docs:`, `test:`, `refactor:`, `ci:`) are encouraged but not
  required.
- Pull requests should describe what changed, why, and any follow-up
  work. Reference an open issue when applicable.
- If your change affects the public API, mention it in
  `CHANGELOG.md` under an `[Unreleased]` section.
- Do not modify the v1.0.0 paper-reproduction results
  (`results/exp*_results.json`) unless the change is explicitly a
  post-paper correction. Those files are the authoritative outputs of
  the published experiments.

---

## Reporting a bug or requesting a feature

Use the issue templates at <https://github.com/alzayats/pinn-reef-thermal/issues/new/choose>.
There are three:

- **Bug report** for a reproducible failure.
- **Feature request** for a proposed new capability.
- **Reef data question** for questions about adapting the framework to
  logger networks other than AIMS, or to reefs outside the Great Barrier
  Reef.

If you include code in a bug report, prefer the CPU example in
`examples/train_on_new_reef.py` as a minimal reproducer.

---

## Security

For security-sensitive issues, email the corresponding author rather than
opening a public issue: `alzayat.saleh@my.jcu.edu.au`.

---

## Licence

By contributing, you agree that your contributions will be licensed under
the MIT Licence, consistent with the rest of the project.
