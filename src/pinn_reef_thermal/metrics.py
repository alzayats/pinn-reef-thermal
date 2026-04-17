"""Thermal stress metrics for coral bleaching assessment.

This module implements the two standard quantities used to convert a temperature
time series into an indicator of accumulated heat stress:

- :func:`compute_dhd` computes the cumulative Degree Heating Days above a bleaching
  threshold. It is a finer-grained relative of the operational Degree Heating Weeks
  metric used by NOAA Coral Reef Watch, and is useful when the underlying time series
  has sub-weekly resolution (hourly or sub-hourly logger data, or a continuous PINN
  reconstruction) where the weekly integration of DHW would discard information.
- :func:`compute_mmm_threshold` computes the Maximum of Monthly Means (MMM) plus one
  degree Celsius, the standard bleaching threshold used by NOAA Coral Reef Watch.

Both functions work on arbitrary temperature time series, so they can be applied to
in-situ logger observations, PINN reconstructions, or satellite SST with the same
interface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_dhd(T_series, t_seconds, threshold, max_dt_days=2.0):
    """Compute cumulative Degree Heating Days above a bleaching threshold.

    Degree Heating Days accumulate the product of temperature excess above the
    bleaching threshold and the duration spent at that excess. Formally, for a
    temperature time series ``T(t)`` and threshold ``T*``,

    .. math::

        \\text{DHD}(t) = \\int_0^{t} \\max(0, T(\\tau) - T^*) \\, d\\tau

    in units of degrees Celsius times days. The discrete approximation uses the
    trapezoidal-style increment ``max(0, T_i - T*) * dt_i`` where ``dt_i`` is the
    duration associated with sample ``i``.

    Parameters
    ----------
    T_series : array-like of float
        Temperature samples in degrees Celsius. One-dimensional.
    t_seconds : array-like of float
        Sample times in seconds, same length as ``T_series``, monotonically
        non-decreasing.
    threshold : float
        Bleaching threshold in degrees Celsius. Samples at or below ``threshold``
        contribute zero heat stress; samples above contribute ``T - threshold``
        multiplied by their duration.
    max_dt_days : float, optional
        Maximum allowed duration in days attributable to a single sample. Gaps in
        the time series larger than this cap (for example, a logger turned off
        between deployments) are clipped so that a two-week gap does not inflate
        DHD by crediting two weeks of excess to a single post-gap sample. Default
        ``2.0`` days, which is longer than any legitimate sub-hourly to daily
        sample interval used in this package.

    Returns
    -------
    dhd : numpy.ndarray of float
        Cumulative DHD at each sample, same length as ``T_series``. The last
        element is the total accumulated stress over the whole series. Units
        are degree-Celsius-days.

    Notes
    -----
    The bleaching threshold is conventionally the Maximum of Monthly Means plus
    one degree Celsius; see :func:`compute_mmm_threshold`. DHD is an hourly or
    sub-hourly analogue of the operational Degree Heating Week metric used by
    NOAA Coral Reef Watch (one DHW equals seven DHD when the latter is applied
    to the same threshold).

    Examples
    --------
    >>> import numpy as np
    >>> t = np.arange(0, 10 * 86400, 3600, dtype=float)  # ten days, hourly
    >>> T = 29.5 + 0.5 * np.sin(2 * np.pi * t / 86400)    # diurnal around 29.5 C
    >>> dhd = compute_dhd(T, t, threshold=29.0)
    >>> float(dhd[-1])  # doctest: +ELLIPSIS
    4.9...
    """
    T_series = np.asarray(T_series, dtype=float)
    t_seconds = np.asarray(t_seconds, dtype=float)

    dt_days = np.diff(t_seconds, prepend=t_seconds[0]) / 86400.0
    dt_days = np.minimum(dt_days, max_dt_days)
    excess = np.maximum(0.0, T_series - threshold)
    return np.cumsum(excess * dt_days)


def compute_mmm_threshold(sst_df, offset=1.0):
    """Compute the bleaching threshold as Maximum of Monthly Means plus an offset.

    The Maximum of Monthly Means (MMM) is the maximum across the twelve calendar
    monthly mean SST values for a site; it is conventionally used as the
    climatological warm-season temperature. NOAA Coral Reef Watch adds one degree
    Celsius to the MMM to define the bleaching alert threshold.

    Parameters
    ----------
    sst_df : pandas.DataFrame
        DataFrame with at least two columns: ``"time"`` (datetime) and
        ``"CRW_SST"`` (float, degrees Celsius). Any additional columns are
        ignored. Rows with missing SST are skipped silently.
    offset : float, optional
        Degrees Celsius added to the MMM to form the threshold. Default
        ``1.0`` per NOAA Coral Reef Watch convention. Set to zero to return the
        bare MMM climatology.

    Returns
    -------
    threshold : float
        ``MMM + offset`` in degrees Celsius.

    Notes
    -----
    This implementation uses calendar months rather than rolling thirty-day
    windows. A single-site climatology of at least one year is recommended;
    multi-year SST histories yield more stable MMM estimates.

    Examples
    --------
    >>> import pandas as pd
    >>> t = pd.date_range("2020-01-01", "2023-12-31", freq="D")
    >>> sst = 27.5 + 2.0 * (t.month.isin([12, 1, 2, 3])).astype(float)
    >>> df = pd.DataFrame({"time": t, "CRW_SST": sst})
    >>> compute_mmm_threshold(df)
    30.5
    """
    df = sst_df.copy()
    df = df.dropna(subset=["CRW_SST"])
    df["month"] = pd.to_datetime(df["time"]).dt.month
    monthly_means = df.groupby("month")["CRW_SST"].mean()
    mmm = float(monthly_means.max())
    return mmm + float(offset)
