"""
Data loader for Phase 2 sub-daily AIMS data (data/ALZ/).

Reads per-deployment CSVs from the AIMS Time Series Explorer,
handles both Weather Station (30-col) and Logger (19-col) formats,
applies QC filtering, merges per reef, and prepares PINN training arrays.

Usage:
    reef_data = load_reef_subdaily("davies_reef")
    # or with time window:
    reef_data = load_reef_subdaily("davies_reef",
                                   date_from="2010-01-01", date_to="2016-12-31")
"""

import glob
import os
import re

import numpy as np
import pandas as pd

ALZ_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ALZ")

# Reef name patterns in filenames -> canonical key
REEF_PATTERNS = {
    "davies_reef": ["Davies Reef", "Davies"],
    "rib_reef": ["Rib Reef"],
    "myrmidon_reef": ["Myrmidon Reef", "Myrmidon"],
    "kelso_reef": ["Kelso Reef"],
}


def _identify_reef(filename):
    """Map filename to canonical reef key."""
    for key, patterns in REEF_PATTERNS.items():
        for p in patterns:
            if p in filename:
                return key
    return None


def _parse_depth_from_filename(filename):
    """Extract depth from @Xm pattern in filename."""
    m = re.search(r"@([\d.]+)m", filename)
    return float(m.group(1)) if m else None


def _read_file(filepath):
    """Read a single AIMS CSV, normalise to common schema.

    Returns DataFrame with columns: [time, depth, temperature, qc_ok]
    """
    fname = os.path.basename(filepath)

    # Skip readme files
    if "readme" in fname.lower():
        return None

    # Skip duplicate files
    if "duplicate" in fname.lower():
        return None

    # Peek at header to determine format
    with open(filepath) as f:
        header = f.readline().strip().split(",")

    try:
        if "qc_value" in header:
            # Format 1: Weather Station / Sensor Float (30 cols)
            df = pd.read_csv(
                filepath,
                usecols=["time", "depth", "qc_value", "qc_flag"],
                dtype={"qc_flag": str},
                low_memory=False,
            )
            df = df.rename(columns={"qc_value": "temperature"})
            df["qc_ok"] = df["qc_flag"].isin(["Good", "Probably Good"])
            df = df.drop(columns=["qc_flag"])

        elif "qc_val" in header:
            # Format 2: Temperature Logger (19 cols)
            cols_to_read = ["time", "qc_val", "qc_flag"]
            df = pd.read_csv(filepath, usecols=cols_to_read, low_memory=False)
            df = df.rename(columns={"qc_val": "temperature"})
            df["qc_ok"] = df["qc_flag"] == 1
            df = df.drop(columns=["qc_flag"])

            # Depth from filename
            depth = _parse_depth_from_filename(fname)
            if depth is None:
                # Try deployment metadata subsite naming
                # e.g. DAVCHS12M -> 12m, RIBCHS6M -> 6m
                m = re.search(r"CHS?(\d+)M", fname)
                if m:
                    depth = float(m.group(1))
                else:
                    # Try reading subsite from file to get depth from mapping
                    df_sub = pd.read_csv(filepath, usecols=["subsite"], nrows=1)
                    subsite = df_sub["subsite"].iloc[0]
                    # Hardcoded fallback depths from deployment metadata
                    depth_map = {
                        "DAVCHS3M": 3.0, "DAVCHS6M": 6.0, "DAVCHS12M": 12.0,
                        "DAVCHN3M": 3.0,
                        "RIBCHS2M": 2.0, "RIBCHS6M": 6.0, "RIBCHS9M": 9.0,
                        "MYRFL1": 3.9, "MYRSL1": 8.0, "MYRDSL1": 20.0,
                        "KELFL1": 2.0, "KELFL2": 2.0, "KELSL1": 8.0,
                        "DAVFL1": 1.6, "DAVSL1": 8.0,
                        "RIBFL1": 2.0, "RIBSL1": 9.0,
                    }
                    depth = depth_map.get(subsite)

            if depth is None:
                return None

            df["depth"] = depth
        else:
            return None

        # Parse time
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time", "temperature"])

        # Convert temperature to float
        df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
        df = df.dropna(subset=["temperature"])

        # Basic sanity filter
        df = df[(df["temperature"] > 10) & (df["temperature"] < 40)]

        return df[["time", "depth", "temperature", "qc_ok"]]

    except Exception as e:
        print(f"  Error reading {fname}: {e}")
        return None


def scan_reef_files(reef_key, alz_dir=None):
    """List all CSV files for a given reef."""
    if alz_dir is None:
        alz_dir = ALZ_DIR

    all_files = sorted(glob.glob(os.path.join(alz_dir, "*.csv")))
    reef_files = []
    for f in all_files:
        if _identify_reef(os.path.basename(f)) == reef_key:
            reef_files.append(f)
    return reef_files


def load_reef_subdaily(reef_key, date_from=None, date_to=None,
                       qc_only=True, alz_dir=None, verbose=True):
    """
    Load all sub-daily data for a reef, merged and QC-filtered.

    Parameters
    ----------
    reef_key : str
        e.g. 'davies_reef'
    date_from, date_to : str, optional
        Date range filter (ISO format). If None, use all data.
    qc_only : bool
        If True, keep only QC-good observations.
    alz_dir : str, optional
        Path to ALZ directory.

    Returns
    -------
    DataFrame with columns: [time, depth, temperature]
    """
    reef_files = scan_reef_files(reef_key, alz_dir)
    if verbose:
        print(f"Loading {reef_key}: {len(reef_files)} files")

    dfs = []
    for f in reef_files:
        df = _read_file(f)
        if df is not None and len(df) > 0:
            if qc_only:
                df = df[df["qc_ok"]]
            df = df.drop(columns=["qc_ok"])
            dfs.append(df)

    if not dfs:
        raise ValueError(f"No data found for {reef_key}")

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sort_values("time").reset_index(drop=True)

    # Drop duplicates (same time + depth)
    merged = merged.drop_duplicates(subset=["time", "depth"], keep="first")

    # Date range filter
    if date_from:
        merged = merged[merged["time"] >= pd.Timestamp(date_from, tz="UTC")]
    if date_to:
        merged = merged[merged["time"] <= pd.Timestamp(date_to, tz="UTC")]

    if verbose:
        depths = sorted(merged["depth"].unique())
        print(f"  Total observations: {len(merged):,}")
        print(f"  Date range: {merged['time'].min()} to {merged['time'].max()}")
        print(f"  Depths ({len(depths)}): {[f'{d:.1f}' for d in depths]}")
        for d in depths:
            dm = merged[merged["depth"] == d]
            print(f"    {d:>5.1f}m: {len(dm):>8,} obs, "
                  f"{dm['time'].min().date()} to {dm['time'].max().date()}, "
                  f"T: {dm['temperature'].min():.1f}-{dm['temperature'].max():.1f}°C")

    return merged


def aggregate_hourly(df):
    """Aggregate sub-daily data to hourly means per depth.

    Returns DataFrame with columns: [time, depth, temperature]
    """
    df = df.copy()
    df["hour"] = df["time"].dt.floor("h")
    hourly = (df.groupby(["hour", "depth"])
              .agg(temperature=("temperature", "mean"))
              .reset_index()
              .rename(columns={"hour": "time"}))
    return hourly


def find_overlap_window(df, min_depths=3, min_days=90):
    """Find the best time window where the most depths overlap.

    Returns (date_from, date_to, list_of_depths).
    """
    depths = sorted(df["depth"].unique())

    # For each depth, get its date range
    depth_ranges = {}
    for d in depths:
        dm = df[df["depth"] == d]
        depth_ranges[d] = (dm["time"].min(), dm["time"].max())

    # Find windows where >= min_depths overlap
    best_window = None
    best_score = 0  # n_depths * n_days

    # Try all pairs of depth start/end times as candidate window boundaries
    all_starts = sorted(set(v[0] for v in depth_ranges.values()))
    all_ends = sorted(set(v[1] for v in depth_ranges.values()))

    for start in all_starts:
        for end in all_ends:
            if (end - start).days < min_days:
                continue
            # Count depths active in this window
            active = [d for d, (s, e) in depth_ranges.items()
                      if s <= start and e >= end]
            if len(active) >= min_depths:
                score = len(active) * (end - start).days
                if score > best_score:
                    best_score = score
                    best_window = (start, end, active)

    if best_window is None:
        # Fallback: just use the full range with whatever depths we have
        return df["time"].min(), df["time"].max(), depths

    return best_window


def prepare_pinn_data(df, sst_df=None, use_hourly=True):
    """
    Convert a merged reef DataFrame into PINN training arrays.

    Parameters
    ----------
    df : DataFrame
        Output of load_reef_subdaily() or aggregate_hourly().
    sst_df : DataFrame, optional
        Satellite SST with columns [time, CRW_SST]. If None, no BC data.
    use_hourly : bool
        If True, aggregate to hourly before creating arrays.

    Returns
    -------
    dict with keys:
        z_data, t_data, T_data : logger observations
        z_bc, t_bc, T_bc : surface BC from satellite SST (empty if no sst_df)
        depths : dict mapping depth_value -> depth_value (for compatibility)
        metadata : dict with reef info
    """
    if use_hourly:
        df = aggregate_hourly(df)

    df = df.sort_values("time").reset_index(drop=True)

    t_start = df["time"].min()
    df["t_seconds"] = (df["time"] - t_start).dt.total_seconds().astype(np.float64)

    z_data = df["depth"].values.astype(np.float32)
    t_data = df["t_seconds"].values.astype(np.float32)
    T_data = df["temperature"].values.astype(np.float32)

    z_max = float(z_data.max()) * 1.2
    t_max = float(t_data.max())

    # Depths dict (for compatibility with prepare_holdout)
    unique_depths = sorted(df["depth"].unique())
    depths = {f"{d:.1f}m": float(d) for d in unique_depths}

    # Surface BC from SST
    if sst_df is not None and len(sst_df) > 0:
        sst = sst_df.copy()
        sst["time"] = pd.to_datetime(sst["time"], utc=True)
        sst = sst.dropna(subset=["CRW_SST"])
        sst["t_seconds"] = (sst["time"] - t_start).dt.total_seconds()
        sst = sst[(sst["t_seconds"] >= 0) & (sst["t_seconds"] <= t_max)]

        z_bc = np.zeros(len(sst), dtype=np.float32)
        t_bc = sst["t_seconds"].values.astype(np.float32)
        T_bc = sst["CRW_SST"].values.astype(np.float32)
    else:
        z_bc = np.array([], dtype=np.float32)
        t_bc = np.array([], dtype=np.float32)
        T_bc = np.array([], dtype=np.float32)

    # Time offsets for correct diurnal/seasonal encoding
    t_start_utc_hour = t_start.hour + t_start.minute / 60.0 + t_start.second / 3600.0
    start_day_of_year = float(t_start.timetuple().tm_yday) - 1  # 0-indexed (Jan 1 = 0)

    metadata = {
        "t_start": str(t_start),
        "t_max": float(t_max),
        "z_max": float(z_max),
        "n_data": len(z_data),
        "n_bc": len(z_bc),
        "depths": depths,
        "T_range": (float(T_data.min()), float(T_data.max())),
        "T_mean": float(np.mean(T_data)),
        "t_start_utc_hour": t_start_utc_hour,
        "start_day_of_year": start_day_of_year,
    }

    return {
        "z_data": z_data,
        "t_data": t_data,
        "T_data": T_data,
        "z_bc": z_bc,
        "t_bc": t_bc,
        "T_bc": T_bc,
        "depths": depths,
        "metadata": metadata,
    }


def prepare_holdout_v2(data, holdout_depth, tol=0.3):
    """
    Split data for leave-one-depth-out validation.

    Parameters
    ----------
    data : dict from prepare_pinn_data()
    holdout_depth : float
        The depth to hold out.
    tol : float
        Tolerance for matching depth (metres).

    Returns training data (without holdout) and holdout data.
    """
    z_d = data["z_data"]
    t_d = data["t_data"]
    T_d = data["T_data"]

    mask = np.abs(z_d - holdout_depth) < tol

    # Find the depth key to remove
    holdout_key = None
    for k, v in data["depths"].items():
        if abs(v - holdout_depth) < tol:
            holdout_key = k
            break

    train_depths = {k: v for k, v in data["depths"].items() if k != holdout_key}

    train_data = {
        "z_data": z_d[~mask],
        "t_data": t_d[~mask],
        "T_data": T_d[~mask],
        "z_bc": data["z_bc"],
        "t_bc": data["t_bc"],
        "T_bc": data["T_bc"],
        "depths": train_depths,
        "metadata": data["metadata"],
    }

    holdout_data = {
        "z": z_d[mask],
        "t": t_d[mask],
        "T": T_d[mask],
        "depth": holdout_depth,
        "subsite": holdout_key or f"{holdout_depth:.1f}m",
    }

    return train_data, holdout_data


def download_crw_sst(lat, lon, date_from, date_to, bbox_size=0.025):
    """Download CRW SST from ERDDAP for a given location and date range.

    Tries CoastWatch (redirects to PacIOOS) with follow-redirects enabled.
    CRW_SST (dhw_5km) is available from ~1985 to present at 5 km resolution.
    """
    import requests

    base = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/NOAA_DHW.csv"
    query = (
        f"?CRW_SST[({date_from}T12:00:00Z):1:({date_to}T12:00:00Z)]"
        f"[({lat-bbox_size}):1:({lat+bbox_size})]"
        f"[({lon-bbox_size}):1:({lon+bbox_size})]"
    )
    url = base + query

    r = requests.get(url, timeout=180, allow_redirects=True)
    r.raise_for_status()

    lines = r.text.strip().split("\n")
    # Skip the units row (row 2)
    header = lines[0]
    data_lines = [header] + lines[2:]

    from io import StringIO
    sst = pd.read_csv(StringIO("\n".join(data_lines)))
    sst["CRW_SST"] = pd.to_numeric(sst["CRW_SST"], errors="coerce")
    sst = sst.dropna(subset=["CRW_SST"])

    # Average across grid cells
    sst_daily = sst.groupby("time").agg(CRW_SST=("CRW_SST", "mean")).reset_index()
    return sst_daily


# Reef coordinates for SST download
REEF_COORDS = {
    "davies_reef": (-18.8316, 147.6345),
    "rib_reef": (-18.473, 146.8774),
    "myrmidon_reef": (-18.2660, 147.3830),
    "kelso_reef": (-18.4317, 146.9883),
}
