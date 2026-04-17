# API documentation

The API reference for `pinn_reef_thermal` is auto-generated from in-source
NumPy-style docstrings using [pdoc](https://pdoc.dev/) and deployed to GitHub
Pages on every push to `main` via `.github/workflows/docs.yml`.

Hosted docs: <https://alzayats.github.io/pinn-reef-thermal/>

## Build the docs locally

```bash
pip install -e .[dev]
pdoc -o site -d numpy pinn_reef_thermal
python -m http.server -d site
```

Open <http://localhost:8000/> in your browser.

## Layout

The `site/` directory produced by `pdoc` contains a single entry point
(`site/pinn_reef_thermal.html`) that indexes the submodules. All public
names are re-exported from the package root; use the root page for
discovery and click through to module pages for full signatures and notes.

## Logo

The GitHub Actions workflow references `docs/assets/logo.png`. The file is
optional; if not present, `pdoc` falls back to its default styling. Add a
128x128 pixel PNG at that path to brand the rendered documentation.
