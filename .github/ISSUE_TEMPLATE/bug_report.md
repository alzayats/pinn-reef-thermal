---
name: Bug report
about: Report a reproducible failure or incorrect behaviour
title: "[bug] "
labels: bug
---

**Describe the bug**
A clear and concise description of what went wrong.

**Minimal reproducer**
Prefer adapting `examples/train_on_new_reef.py` to the smallest change
that triggers the bug. Paste the full error traceback.

```python
# your reproducer
```

**Expected behaviour**
What you expected to happen instead.

**Environment**
- OS:
- Python version: (output of `python --version`)
- pinn-reef-thermal version: (output of `python -c "import pinn_reef_thermal as p; print(p.__version__)"`)
- JAX version: (output of `python -c "import jax; print(jax.__version__)"`)
- Installation method: (`pip install -e .[cpu]`, `.[gpu]`, conda env, etc.)
- Hardware: CPU only, or GPU model and CUDA version

**Additional context**
Anything else we should know: custom data, unusual environment, partial
reproductions.
