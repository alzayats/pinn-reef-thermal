"""
Baseline methods for reef thermal reconstruction.

All methods share the interface:
    predict(train_data, holdout_data) -> T_pred

Where:
    train_data: dict with z_data, t_data, T_data, z_bc, t_bc, T_bc, depths
    holdout_data: dict with z, t, T, depth, subsite
"""

import numpy as np
from scipy.spatial import cKDTree


def _normalise_zt(z, t, z_scale, t_scale):
    """Normalise z and t to comparable scales."""
    return np.column_stack([z / z_scale, t / t_scale])


def _get_scales(train_data):
    """Get normalisation scales from training data."""
    z_all = np.concatenate([train_data["z_data"], train_data["z_bc"]])
    t_all = np.concatenate([train_data["t_data"], train_data["t_bc"]])
    z_scale = max(z_all.max() - z_all.min(), 1.0)
    t_scale = max(t_all.max() - t_all.min(), 1.0)
    return z_scale, t_scale


def _get_noon_mask(t_data, z_data, train_data):
    """Get mask for noon-only values (every 3rd point when minmax augmented).

    Heuristic: for each unique depth, take only points closest to noon
    (i.e., the first of each triplet if data was augmented with 6am, noon, 2pm).
    If data doesn't look augmented, return all points.
    """
    # Check if data appears to be minmax-augmented (3x points)
    unique_depths = sorted(set(np.round(z_data, 1)))
    if len(unique_depths) == 0:
        return np.ones(len(t_data), dtype=bool)

    # Count points per depth — if roughly 3x what we'd expect for daily, subsample
    counts = [np.sum(np.abs(z_data - d) < 0.5) for d in unique_depths]
    median_count = np.median(counts)

    if median_count < 50:
        # Small dataset, use all
        return np.ones(len(t_data), dtype=bool)

    # Take every 3rd point (the noon/mean values come first in the triplet)
    mask = np.zeros(len(t_data), dtype=bool)
    for d in unique_depths:
        depth_mask = np.abs(z_data - d) < 0.5
        indices = np.where(depth_mask)[0]
        # Take every 3rd starting from 0 (noon values)
        mask[indices[::3]] = True
    return mask


# --- Gaussian Process Regression ---

def gp_predict(train_data, holdout_data, z_scale=None, t_scale=None):
    """GP regression baseline with RBF + White kernel.

    Uses noon-only values to keep O(n³) tractable.
    Returns (T_pred, T_std).
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    z_train = train_data["z_data"]
    t_train = train_data["t_data"]
    T_train = train_data["T_data"]

    # Subsample to noon-only to keep GP tractable
    noon_mask = _get_noon_mask(t_train, z_train, train_data)
    z_tr = z_train[noon_mask]
    t_tr = t_train[noon_mask]
    T_tr = T_train[noon_mask]

    # Further subsample if still too large (GP is O(n³))
    max_pts = 500
    if len(z_tr) > max_pts:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(z_tr), max_pts, replace=False)
        z_tr, t_tr, T_tr = z_tr[idx], t_tr[idx], T_tr[idx]

    if z_scale is None or t_scale is None:
        z_scale, t_scale = _get_scales(train_data)

    X_train = _normalise_zt(z_tr, t_tr, z_scale, t_scale)
    X_test = _normalise_zt(holdout_data["z"], holdout_data["t"], z_scale, t_scale)

    kernel = RBF(length_scale=[1.0, 1.0]) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                  normalize_y=True, alpha=1e-6)
    gp.fit(X_train, T_tr)
    T_pred, T_std = gp.predict(X_test, return_std=True)

    return T_pred.astype(np.float32), T_std.astype(np.float32)


# --- Inverse Distance Weighting ---

def idw_predict(train_data, holdout_data, power=2, z_scale=None, t_scale=None):
    """Inverse Distance Weighting baseline."""
    if z_scale is None or t_scale is None:
        z_scale, t_scale = _get_scales(train_data)

    X_train = _normalise_zt(train_data["z_data"], train_data["t_data"],
                            z_scale, t_scale)
    X_test = _normalise_zt(holdout_data["z"], holdout_data["t"],
                           z_scale, t_scale)
    T_train = train_data["T_data"]

    tree = cKDTree(X_train)
    # Use k nearest neighbours to avoid computing all distances
    k = min(20, len(X_train))
    dists, idxs = tree.query(X_test, k=k)

    # Avoid division by zero
    dists = np.maximum(dists, 1e-10)
    weights = 1.0 / dists ** power
    weights /= weights.sum(axis=1, keepdims=True)

    T_pred = np.sum(weights * T_train[idxs], axis=1)
    return T_pred.astype(np.float32)


# --- Nearest Neighbor ---

def nn_predict(train_data, holdout_data, z_scale=None, t_scale=None):
    """Nearest neighbour baseline."""
    if z_scale is None or t_scale is None:
        z_scale, t_scale = _get_scales(train_data)

    X_train = _normalise_zt(train_data["z_data"], train_data["t_data"],
                            z_scale, t_scale)
    X_test = _normalise_zt(holdout_data["z"], holdout_data["t"],
                           z_scale, t_scale)

    tree = cKDTree(X_train)
    _, idxs = tree.query(X_test, k=1)
    T_pred = train_data["T_data"][idxs]
    return T_pred.astype(np.float32)


# --- Random Forest ---

def rf_predict(train_data, holdout_data):
    """Random Forest baseline with time-encoded features."""
    from sklearn.ensemble import RandomForestRegressor

    T_DAY = 86400.0

    def _features(z, t):
        t_norm = t / t.max() if t.max() > 0 else t
        return np.column_stack([
            z,
            np.sin(2 * np.pi * t / T_DAY),
            np.cos(2 * np.pi * t / T_DAY),
            t_norm,
        ])

    X_train = _features(train_data["z_data"], train_data["t_data"])
    X_test = _features(holdout_data["z"], holdout_data["t"])

    rf = RandomForestRegressor(n_estimators=200, max_depth=20,
                               min_samples_leaf=5, n_jobs=-1,
                               random_state=42)
    rf.fit(X_train, train_data["T_data"])
    T_pred = rf.predict(X_test)
    return T_pred.astype(np.float32)


# --- Finite-Difference Physics-Only ---

def fd_predict(holdout_data, t_sst, T_sst, meta,
               kappa=2.5e-4, Kd=0.1, Q_max=350.0, rho_cp=4.1e6,
               nz=100, dt=3600.0, utc_offset=10.0):
    """Physics-only baseline: 1D heat equation via implicit Euler.

    Uses real satellite SST as surface Dirichlet BC and literature-value
    physical parameters. No neural network — isolates the physics contribution.

    Parameters
    ----------
    holdout_data : dict with z, t, T arrays
    t_sst, T_sst : 1D arrays of satellite SST (seconds since t_start, °C)
    meta : dict with z_max, t_max, t_start_utc_hour
    kappa : thermal diffusivity (m²/s), default 2.5e-4
    Kd : light attenuation coefficient (m⁻¹), default 0.1
    Q_max : peak solar irradiance (W/m²), default 350
    rho_cp : volumetric heat capacity (J/m³/K), default 4.1e6
    nz : number of vertical grid points
    dt : timestep (seconds), default 3600 (hourly)
    utc_offset : UTC offset for solar source (hours), default 10 (AEST)

    Returns
    -------
    T_pred : predicted temperature at holdout points (float32)
    """
    from scipy.interpolate import RegularGridInterpolator, interp1d
    from scipy.linalg import solve_banded

    z_max = meta["z_max"]
    t_max = meta["t_max"]
    t_start_utc_hour = meta.get("t_start_utc_hour", 0.0)

    # FD grid
    dz = z_max / (nz - 1)
    nt = int(t_max / dt) + 1
    z = np.linspace(0, z_max, nz)
    t = np.linspace(0, t_max, nt)

    # Interpolate daily SST to FD timesteps
    sst_interp = interp1d(t_sst, T_sst, kind='linear',
                          fill_value=(T_sst[0], T_sst[-1]),
                          bounds_error=False)

    # Solar source: depth-dependent heating, daytime only
    def solar_source(z_arr, t_val):
        local_hour = ((t_val / 3600.0 + t_start_utc_hour) + utc_offset) % 24.0
        I_surface = Q_max * max(0.0, np.sin(np.pi * (local_hour - 6) / 12))
        return (I_surface * Kd / rho_cp) * np.exp(-Kd * z_arr)

    # Solve: implicit Euler
    T_field = np.zeros((nz, nt))
    T_field[:, 0] = float(sst_interp(0.0))  # isothermal IC from first SST

    r = kappa * dt / (dz ** 2)

    for n in range(nt - 1):
        t_next = t[n + 1]
        S = solar_source(z, t_next) * dt

        main_diag = np.ones(nz) * (1 + 2 * r)
        upper_diag = np.ones(nz - 1) * (-r)
        lower_diag = np.ones(nz - 1) * (-r)
        rhs = T_field[:, n].copy() + S

        # Dirichlet BC at surface (real SST)
        main_diag[0] = 1.0
        upper_diag[0] = 0.0
        rhs[0] = float(sst_interp(t_next))

        # Neumann BC at bottom (insulating)
        main_diag[-1] = 1 + r
        lower_diag[-1] = -r
        rhs[-1] = T_field[-1, n] + S[-1]

        ab = np.zeros((3, nz))
        ab[0, 1:] = upper_diag
        ab[1, :] = main_diag
        ab[2, :-1] = lower_diag
        T_field[:, n + 1] = solve_banded((1, 1), ab, rhs)

    # Interpolate FD solution to holdout (z, t) points
    interp_fd = RegularGridInterpolator(
        (z, t), T_field, method='linear',
        bounds_error=False, fill_value=None)
    pts = np.column_stack([holdout_data["z"], holdout_data["t"]])
    T_pred = interp_fd(pts)

    return T_pred.astype(np.float32)


# --- Satellite-only ---

def satellite_predict(train_data, holdout_data):
    """Satellite-only baseline: use CRW_SST at z=0 for all depths.

    For each holdout time, find nearest SST observation.
    """
    t_bc = train_data["t_bc"]
    T_bc = train_data["T_bc"]

    if len(t_bc) == 0:
        return None

    T_pred = []
    for th in holdout_data["t"]:
        idx = np.argmin(np.abs(t_bc - th))
        T_pred.append(T_bc[idx])
    return np.array(T_pred, dtype=np.float32)


# --- Run all baselines ---

def run_all_baselines(train_data, holdout_data):
    """Run all baselines and return dict of results.

    Returns dict: method_name -> T_pred array
    """
    z_scale, t_scale = _get_scales(train_data)

    results = {}

    # GP
    try:
        T_gp, T_gp_std = gp_predict(train_data, holdout_data,
                                     z_scale=z_scale, t_scale=t_scale)
        results["GP"] = T_gp
        results["GP_std"] = T_gp_std
    except Exception as e:
        print(f"  GP failed: {e}")

    # IDW
    results["IDW"] = idw_predict(train_data, holdout_data,
                                 z_scale=z_scale, t_scale=t_scale)

    # Nearest Neighbor
    results["NN"] = nn_predict(train_data, holdout_data,
                               z_scale=z_scale, t_scale=t_scale)

    # Random Forest
    try:
        results["RF"] = rf_predict(train_data, holdout_data)
    except Exception as e:
        print(f"  RF failed: {e}")

    # Satellite-only (skip if no SST data)
    sat = satellite_predict(train_data, holdout_data)
    if sat is not None:
        results["Satellite"] = sat

    return results
