"""
Data loading and preparation for PINN training.

Converts daily AIMS logger data + CRW SST into PINN training arrays:
  - z_data, t_data, T_data: logger observations at (depth, time)
  - z_bc, t_bc, T_bc: surface boundary condition from satellite SST
  - z_pde, t_pde: collocation points for PDE residual
"""

import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")


def load_reef_data(reef_key, subsites=None, use_minmax=True):
    """
    Load and prepare reef data for PINN training.

    Parameters
    ----------
    reef_key : str
        e.g. 'rib_reef', 'davies_reef'
    subsites : list of str, optional
        Which subsites to include. If None, use all.
    use_minmax : bool
        If True, augment data points using daily min (~6am) and max (~2pm)
        in addition to daily mean (~noon).

    Returns
    -------
    dict with keys:
        z_data, t_data, T_data : arrays of logger observations
        z_bc, t_bc, T_bc : surface BC from satellite SST
        depths : dict mapping subsite -> depth
        metadata : dict with reef info
    """
    logger_path = os.path.join(DATA_DIR, f"logger_{reef_key}.csv")
    sst_path = os.path.join(DATA_DIR, f"sst_{reef_key}.csv")

    logger = pd.read_csv(logger_path)
    sst = pd.read_csv(sst_path)

    # Filter subsites
    if subsites is not None:
        logger = logger[logger["subsite"].isin(subsites)]

    # Parse dates → seconds from start
    logger["date"] = pd.to_datetime(logger["time"])
    t_start = logger["date"].min()
    logger["t_seconds"] = (logger["date"] - t_start).dt.total_seconds()

    sst["date"] = pd.to_datetime(sst["time"].str[:10])
    sst["CRW_SST"] = pd.to_numeric(sst["CRW_SST"], errors="coerce")
    sst = sst.dropna(subset=["CRW_SST"])
    # Average SST across bounding box grid cells
    sst_daily = sst.groupby("date").agg({"CRW_SST": "mean"}).reset_index()
    sst_daily["t_seconds"] = (sst_daily["date"] - t_start).dt.total_seconds()
    # Keep only SST within logger time range
    sst_daily = sst_daily[sst_daily["t_seconds"] >= 0]

    # Get depths
    depths = {}
    for sub in logger["subsite"].unique():
        depths[sub] = logger[logger["subsite"] == sub]["assigned_depth"].iloc[0]

    t_max = logger["t_seconds"].max()
    z_max = max(depths.values()) * 1.2  # domain extends modestly beyond deepest logger

    # Build data arrays
    z_list, t_list, T_list = [], [], []

    for _, row in logger.iterrows():
        depth = row["assigned_depth"]
        t_sec = row["t_seconds"]

        # Daily mean → assume represents noon (12:00)
        t_noon = t_sec + 12 * 3600  # offset to noon
        z_list.append(depth)
        t_list.append(t_noon)
        T_list.append(row["qc_val"])

        if use_minmax:
            # Daily min → assume ~6am (coolest)
            t_6am = t_sec + 6 * 3600
            z_list.append(depth)
            t_list.append(t_6am)
            T_list.append(row["qc_min"])

            # Daily max → assume ~2pm (warmest)
            t_2pm = t_sec + 14 * 3600
            z_list.append(depth)
            t_list.append(t_2pm)
            T_list.append(row["qc_max"])

    z_data = np.array(z_list, dtype=np.float32)
    t_data = np.array(t_list, dtype=np.float32)
    T_data = np.array(T_list, dtype=np.float32)

    # Surface BC from satellite SST
    # CRW SST is nighttime-only → assign to ~4am (pre-dawn)
    z_bc = np.zeros(len(sst_daily), dtype=np.float32)
    t_bc = (sst_daily["t_seconds"].values + 4 * 3600).astype(np.float32)
    T_bc = sst_daily["CRW_SST"].values.astype(np.float32)

    # Clip t values to valid range
    valid_t_max = t_max + 24 * 3600
    mask_data = t_data <= valid_t_max
    z_data, t_data, T_data = z_data[mask_data], t_data[mask_data], T_data[mask_data]
    mask_bc = t_bc <= valid_t_max
    z_bc, t_bc, T_bc = z_bc[mask_bc], t_bc[mask_bc], T_bc[mask_bc]

    # Time metadata for PINN time encoding
    t_start_utc_hour = t_start.hour + t_start.minute / 60.0
    start_day_of_year = float(t_start.dayofyear)

    metadata = {
        "reef_key": reef_key,
        "t_start": str(t_start),
        "t_max": float(valid_t_max),
        "z_max": float(z_max),
        "n_data": len(z_data),
        "n_bc": len(z_bc),
        "depths": depths,
        "T_range": (float(T_data.min()), float(T_data.max())),
        "T_mean": float(np.mean(T_data)),
        "t_start_utc_hour": float(t_start_utc_hour),
        "start_day_of_year": float(start_day_of_year),
    }

    return {
        "z_data": z_data,
        "t_data": t_data,
        "T_data": T_data,
        "z_bc": z_bc,
        "t_bc": t_bc,
        "T_bc": T_bc,
        "t_sst": t_bc,
        "T_sst": T_bc,
        "depths": depths,
        "metadata": metadata,
    }


def generate_collocation_points(z_max, t_max, n_points, seed=42):
    """Generate random collocation points for PDE residual."""
    rng = np.random.RandomState(seed)
    z_pde = rng.uniform(0, z_max, n_points).astype(np.float32)
    t_pde = rng.uniform(0, t_max, n_points).astype(np.float32)
    return z_pde, t_pde


def generate_stratified_collocation_points(z_max, t_max, n_points,
                                           logger_depths=None, seed=42):
    """Generate stratified collocation points with more density near logger depths.

    Allocates 50% of points uniformly, 50% concentrated near logger depths
    (within ±1m of each logger). This ensures the PDE is well-sampled where
    data constraints exist.

    Parameters
    ----------
    z_max : float
    t_max : float
    n_points : int
    logger_depths : list of float, optional
        If None, falls back to uniform sampling.
    seed : int
    """
    rng = np.random.RandomState(seed)

    if logger_depths is None or len(logger_depths) == 0:
        return generate_collocation_points(z_max, t_max, n_points, seed)

    # 50% uniform, 50% near loggers
    n_uniform = n_points // 2
    n_strat = n_points - n_uniform

    # Uniform component
    z_uni = rng.uniform(0, z_max, n_uniform).astype(np.float32)
    t_uni = rng.uniform(0, t_max, n_uniform).astype(np.float32)

    # Stratified: distribute equally among logger depths
    depths = sorted(set(logger_depths))
    n_per_depth = max(1, n_strat // len(depths))
    z_strat, t_strat = [], []
    for d in depths:
        n_d = n_per_depth
        # Sample z within ±1m of logger depth, clipped to [0, z_max]
        z_lo = max(0, d - 1.0)
        z_hi = min(z_max, d + 1.0)
        z_strat.append(rng.uniform(z_lo, z_hi, n_d).astype(np.float32))
        t_strat.append(rng.uniform(0, t_max, n_d).astype(np.float32))

    z_strat = np.concatenate(z_strat)
    t_strat = np.concatenate(t_strat)

    z_pde = np.concatenate([z_uni, z_strat])
    t_pde = np.concatenate([t_uni, t_strat])

    return z_pde, t_pde


def prepare_holdout(data, holdout_subsite):
    """
    Split data for leave-one-depth-out validation.

    Returns training data (without holdout) and holdout data.
    """
    z_d, t_d, T_d = data["z_data"], data["t_data"], data["T_data"]
    holdout_depth = data["depths"][holdout_subsite]

    # Mask: holdout points are at the holdout depth
    mask = np.abs(z_d - holdout_depth) < 0.1  # tolerance for float comparison

    train_data = {
        "z_data": z_d[~mask],
        "t_data": t_d[~mask],
        "T_data": T_d[~mask],
        "z_bc": data["z_bc"],
        "t_bc": data["t_bc"],
        "T_bc": data["T_bc"],
        "t_sst": data.get("t_sst", data["t_bc"]),
        "T_sst": data.get("T_sst", data["T_bc"]),
        "depths": {k: v for k, v in data["depths"].items() if k != holdout_subsite},
        "metadata": data["metadata"],
    }

    holdout_data = {
        "z": z_d[mask],
        "t": t_d[mask],
        "T": T_d[mask],
        "depth": holdout_depth,
        "subsite": holdout_subsite,
    }

    return train_data, holdout_data
