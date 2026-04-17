"""Adapt your own reef logger data to the pinn-reef-thermal schema.

This example demonstrates how a reef manager with temperature loggers and access
to satellite SST can adapt their own data to the format expected by
:func:`pinn_reef_thermal.prepare_pinn_data` and train a PINN on a new reef not
covered by the AIMS Time Series Explorer.

Expected schemas
----------------
Loggers: a pandas DataFrame with columns ``time`` (datetime64 UTC), ``depth``
(float, metres below surface), and ``temperature`` (float, degrees Celsius).
One row per (time, depth) observation. Sub-hourly sampling is supported; the
loader aggregates to hourly by default.

SST: a pandas DataFrame with columns ``time`` (datetime64 UTC) and ``CRW_SST``
(float, degrees Celsius). Daily or sub-daily cadence are both fine. NOAA Coral
Reef Watch ERDDAP is the recommended source; see
:func:`pinn_reef_thermal.download_crw_sst` for a helper that downloads SST for
a latitude/longitude and date range.

This script synthesises plausible AIMS-shaped CSVs on the fly so it runs
without external data, then shows the one-line adaptation that converts them
into PINN training arrays.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from pinn_reef_thermal import prepare_pinn_data


def build_example_loggers(tmp_dir: Path) -> Path:
    """Write a synthetic logger CSV in the expected schema.

    Replace this function with your own code that reads your existing logger
    records (thermistor string, HOBO Pendant, SBE56, etc.) and writes them out
    as a CSV with the three required columns.
    """
    rng = np.random.default_rng(0)
    time = pd.date_range("2024-11-01", "2025-02-28", freq="1h", tz="UTC")
    rows = []
    for depth in [1.0, 5.0, 10.0, 18.0]:
        # 1 C cooler per 10 m, plus a warm-season trend and diurnal cycle
        trend = 28.0 + 1.5 * np.sin(np.linspace(0, 2 * np.pi, len(time)))
        diurnal = 0.4 * np.sin(2 * np.pi * time.hour / 24.0)
        depth_cool = -0.1 * depth
        noise = rng.normal(0, 0.03, size=len(time))
        temp = trend + diurnal + depth_cool + noise
        rows.append(pd.DataFrame({"time": time, "depth": depth, "temperature": temp}))
    df = pd.concat(rows, ignore_index=True)

    path = tmp_dir / "my_reef_loggers.csv"
    df.to_csv(path, index=False)
    return path


def build_example_sst(tmp_dir: Path) -> Path:
    """Write a synthetic SST CSV matching the ERDDAP CRW schema."""
    rng = np.random.default_rng(1)
    time = pd.date_range("2024-11-01", "2025-02-28", freq="1D", tz="UTC")
    sst = 28.5 + 1.7 * np.sin(np.linspace(0, 2 * np.pi, len(time))) + rng.normal(0, 0.08, size=len(time))
    df = pd.DataFrame({"time": time, "CRW_SST": sst})

    path = tmp_dir / "my_reef_sst.csv"
    df.to_csv(path, index=False)
    return path


def main() -> int:
    tmp_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "pinn_reef_thermal_example"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    loggers_csv = build_example_loggers(tmp_dir)
    sst_csv = build_example_sst(tmp_dir)
    print(f"Wrote example loggers to {loggers_csv}")
    print(f"Wrote example SST to     {sst_csv}")

    # --- Adapt your data to PINN arrays ------------------------------------
    # This is the only required step: read your two CSVs and call
    # prepare_pinn_data. It handles hourly aggregation, time alignment, and
    # metadata extraction.
    loggers = pd.read_csv(loggers_csv, parse_dates=["time"])
    sst = pd.read_csv(sst_csv, parse_dates=["time"])
    loggers["time"] = pd.to_datetime(loggers["time"], utc=True)
    sst["time"] = pd.to_datetime(sst["time"], utc=True)

    reef_data = prepare_pinn_data(loggers, sst_df=sst, use_hourly=True)

    meta = reef_data["metadata"]
    print("\nPINN-ready dataset:")
    print(f"  z_data:  {reef_data['z_data'].shape} ({reef_data['z_data'].dtype})")
    print(f"  t_data:  {reef_data['t_data'].shape}")
    print(f"  T_data:  {reef_data['T_data'].shape}, "
          f"range {reef_data['T_data'].min():.2f}-{reef_data['T_data'].max():.2f} C")
    print(f"  z_bc/t_bc/T_bc: {len(reef_data['z_bc'])} SST samples")
    print(f"  depths:  {sorted(reef_data['depths'].values())}")
    print(f"  z_max:   {meta['z_max']:.1f} m")
    print(f"  t_max:   {meta['t_max'] / 86400.0:.1f} days")

    print(
        "\nNext steps:\n"
        "  - Call fit_pinn(reef_data) to train (see train_on_new_reef.py).\n"
        "  - Or use the pinn-reef CLI:\n"
        f"      pinn-reef train --loggers {loggers_csv} --sst {sst_csv} "
        f"--output {tmp_dir}/outputs --epochs 2000"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
