"""Tests for thermal stress metrics (compute_dhd, compute_mmm_threshold)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pinn_reef_thermal import compute_dhd, compute_mmm_threshold


def test_compute_dhd_zero_when_below_threshold():
    t = np.arange(0, 10 * 86400, 3600, dtype=float)
    T = np.full_like(t, 28.0)
    dhd = compute_dhd(T, t, threshold=29.0)
    assert np.allclose(dhd, 0.0)


def test_compute_dhd_constant_excess():
    """One degree above threshold for ten days should give 10 C.days."""
    t = np.arange(0, 10 * 86400, 3600, dtype=float)
    T = np.full_like(t, 30.0)
    dhd = compute_dhd(T, t, threshold=29.0)
    assert dhd[-1] == pytest.approx(10.0, abs=0.05)


def test_compute_dhd_monotonic():
    rng = np.random.default_rng(0)
    t = np.arange(0, 30 * 86400, 3600, dtype=float)
    T = 29.0 + rng.uniform(-0.5, 1.5, size=t.shape)
    dhd = compute_dhd(T, t, threshold=29.0)
    # Cumulative sum of non-negative values must be monotonic non-decreasing.
    assert np.all(np.diff(dhd) >= -1e-9)


def test_compute_dhd_gap_clipping():
    """A long gap must not inflate DHD through a single post-gap sample."""
    # Two samples with 30 days between them, then a short follow-up.
    t = np.array([0.0, 30 * 86400.0, 30 * 86400.0 + 3600.0])
    T = np.array([30.0, 30.0, 30.0])
    dhd = compute_dhd(T, t, threshold=29.0, max_dt_days=2.0)
    # First sample contributes (30-29)*(0/86400) = 0 (prepend first).
    # Second sample is capped at 2 days worth of excess: 1.0 * 2.0 = 2.0.
    # Third sample: 1 hour of excess = ~0.042.
    assert dhd[-1] < 3.0


def test_compute_mmm_threshold_calendar_months():
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
    # A hot summer (Dec-Mar in southern hemisphere): 30 C, otherwise 26 C.
    hot = dates.month.isin([12, 1, 2, 3])
    sst = np.where(hot, 30.0, 26.0)
    df = pd.DataFrame({"time": dates, "CRW_SST": sst})
    assert compute_mmm_threshold(df) == pytest.approx(31.0)
    assert compute_mmm_threshold(df, offset=0.0) == pytest.approx(30.0)


def test_compute_mmm_threshold_skips_nan():
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    sst = np.full(len(dates), 28.0)
    sst[::5] = np.nan
    df = pd.DataFrame({"time": dates, "CRW_SST": sst})
    # Dropping NaNs should still yield 28 C monthly means.
    assert compute_mmm_threshold(df) == pytest.approx(29.0)
