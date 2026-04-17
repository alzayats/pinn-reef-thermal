"""Command-line interface for pinn-reef-thermal.

Entry point: ``pinn-reef``. Installed by the ``[project.scripts]`` table in
``pyproject.toml``. Two subcommands are provided:

- ``pinn-reef train`` trains a PINN on a pair of CSV files (logger data and
  CRW SST) and writes the trained model's predictions plus the recovered
  physical parameters to an output directory.
- ``pinn-reef reproduce`` re-runs one of the paper's experiment pipelines
  (exp14 holdout validation, exp15 thermal stress, exp16 multi-seed) by
  dispatching to the scripts in ``scripts/``.

Both subcommands are thin wrappers around the library API; the full machinery
lives in the ``pinn_reef_thermal`` package and the ``scripts/`` directory.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _find_scripts_dir() -> Path:
    """Locate the repository's ``scripts/`` directory.

    The CLI is intended to be run from a checkout of the repository where the
    ``scripts/`` directory sits next to the installed package. When the package
    is installed editably with ``pip install -e .`` the import location is
    ``<repo>/src/pinn_reef_thermal``, so the scripts directory is two levels up.

    If the scripts directory cannot be found (for example, because the package
    was installed into site-packages from a wheel), ``reproduce`` prints an
    informative error pointing the user at the GitHub repository.
    """
    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        pkg_dir.parent.parent / "scripts",
        pkg_dir.parent.parent.parent / "scripts",
        Path.cwd() / "scripts",
    ]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "run_experiments.py").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate the scripts/ directory. The reproduce subcommand "
        "requires a checkout of the pinn-reef-thermal repository. Clone it "
        "from https://github.com/alzayats/pinn-reef-thermal and run this "
        "command from the repository root."
    )


def _cmd_train(args: argparse.Namespace) -> int:
    """Handle ``pinn-reef train``."""
    import numpy as np
    import pandas as pd

    from . import (
        fit_pinn,
        prepare_pinn_data,
    )

    loggers_path = Path(args.loggers)
    sst_path = Path(args.sst)
    output_dir = Path(args.output)

    if not loggers_path.is_file():
        print(f"Error: logger CSV not found at {loggers_path}", file=sys.stderr)
        return 2
    if not sst_path.is_file():
        print(f"Error: SST CSV not found at {sst_path}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading loggers from {loggers_path}")
    loggers = pd.read_csv(loggers_path, parse_dates=["time"])
    if "time" not in loggers.columns or "depth" not in loggers.columns or "temperature" not in loggers.columns:
        print(
            "Error: logger CSV must contain columns: time, depth, temperature. "
            "See examples/extend_to_new_reef.py for the expected schema.",
            file=sys.stderr,
        )
        return 2

    print(f"Loading SST from {sst_path}")
    sst = pd.read_csv(sst_path, parse_dates=["time"])
    if "time" not in sst.columns or "CRW_SST" not in sst.columns:
        print(
            "Error: SST CSV must contain columns: time, CRW_SST.",
            file=sys.stderr,
        )
        return 2

    print("Preparing PINN training arrays")
    reef_data = prepare_pinn_data(loggers, sst_df=sst, use_hourly=True)

    print(f"Training PINN for {args.epochs} epochs (seed={args.seed})")
    pinn, history = fit_pinn(
        reef_data,
        n_epochs=args.epochs,
        seed=args.seed,
        verbose=True,
    )

    recovered = pinn.get_estimated_params()
    print(f"Recovered parameters: {recovered}")

    # Prediction grid
    meta = reef_data["metadata"]
    z_grid = np.linspace(0.0, float(meta["z_max"]), 64)
    t_grid = np.linspace(0.0, float(meta["t_max"]), 200)
    T_field = pinn.predict_field(z_grid, t_grid)

    np.savez(
        output_dir / "predictions.npz",
        z=z_grid,
        t=t_grid,
        T=T_field,
    )

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "recovered_params": recovered,
                "n_epochs": args.epochs,
                "seed": args.seed,
                "n_data": len(reef_data["z_data"]),
                "n_bc": len(reef_data["z_bc"]),
                "reef_metadata": {k: v for k, v in meta.items()
                                   if not isinstance(v, (list, tuple, dict))},
            },
            f,
            indent=2,
            default=str,
        )

    print(f"Predictions saved to {output_dir / 'predictions.npz'}")
    print(f"Metadata saved to {output_dir / 'metadata.json'}")
    return 0


def _cmd_reproduce(args: argparse.Namespace) -> int:
    """Handle ``pinn-reef reproduce``."""
    experiment_scripts = {
        14: "exp14_holdout_validation.py",
        15: "exp15_thermal_stress.py",
        16: "exp16_multiseed.py",
    }
    if args.experiment not in experiment_scripts:
        print(
            f"Error: unknown experiment {args.experiment}. "
            f"Valid choices: {sorted(experiment_scripts)}",
            file=sys.stderr,
        )
        return 2

    try:
        scripts_dir = _find_scripts_dir()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 3

    script_path = scripts_dir / experiment_scripts[args.experiment]
    if not script_path.is_file():
        print(f"Error: experiment script not found at {script_path}", file=sys.stderr)
        return 3

    cmd = [sys.executable, str(script_path)]
    print(f"Running: {' '.join(cmd)}")
    env = os.environ.copy()
    # Prepend repo root to PYTHONPATH so scripts' relative src imports still work
    # even before the package is pip installed.
    env["PYTHONPATH"] = f"{script_path.parent.parent}:{env.get('PYTHONPATH', '')}"
    return subprocess.call(cmd, env=env)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser for the ``pinn-reef`` command."""
    parser = argparse.ArgumentParser(
        prog="pinn-reef",
        description=(
            "Physics-informed neural network framework for depth-resolved "
            "coral reef thermal reconstruction."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser(
        "train",
        help="Train a PINN on logger and SST CSV files.",
        description=(
            "Train a PINN on logger observations and satellite SST, and save "
            "the trained model's depth-resolved temperature field plus the "
            "recovered physical parameters. See examples/extend_to_new_reef.py "
            "for the expected CSV schemas."
        ),
    )
    train.add_argument(
        "--loggers",
        required=True,
        help="Path to a CSV of logger observations with columns: time, depth, temperature.",
    )
    train.add_argument(
        "--sst",
        required=True,
        help="Path to a CSV of Coral Reef Watch SST with columns: time, CRW_SST.",
    )
    train.add_argument(
        "--output",
        required=True,
        help="Output directory. Created if it does not exist.",
    )
    train.add_argument(
        "--epochs",
        type=int,
        default=15000,
        help="Number of training epochs. Default 15000 (paper setting).",
    )
    train.add_argument(
        "--seed",
        type=int,
        default=42,
        help="PRNG seed. Default 42.",
    )
    train.set_defaults(func=_cmd_train)

    reproduce = sub.add_parser(
        "reproduce",
        help="Re-run one of the paper's experiment pipelines.",
        description=(
            "Dispatches to the experiment scripts in scripts/. Must be run "
            "from a checkout of the repository; see the README for details."
        ),
    )
    reproduce.add_argument(
        "--experiment",
        type=int,
        required=True,
        choices=[14, 15, 16],
        help="Which paper experiment to reproduce.",
    )
    reproduce.set_defaults(func=_cmd_reproduce)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point installed as the ``pinn-reef`` command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
