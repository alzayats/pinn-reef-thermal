"""Tests that the sub-daily loader handles both AIMS CSV formats."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pinn_reef_thermal.data_loader_v2 import _read_file


def _write_weather_station_csv(path: Path) -> None:
    """Shape: Weather Station / Sensor Float format with qc_value and qc_flag."""
    n = 48
    df = pd.DataFrame({
        "time": pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC"),
        "depth": np.full(n, 3.0),
        "qc_value": 28.0 + 0.2 * np.sin(np.arange(n)),
        "qc_flag": ["Good"] * n,
        # Extra columns to reach the 30-col shape; _read_file only reads
        # the four required ones, so filler values are sufficient.
        **{f"extra{i}": np.zeros(n) for i in range(26)},
    })
    df.to_csv(path, index=False)


def _write_logger_csv(path: Path, depth: float = 6.0) -> None:
    """Shape: Temperature Logger format with qc_val and numeric qc_flag.

    The logger format has no explicit depth column; depth is encoded in
    the filename via the ``@Xm`` convention.
    """
    n = 48
    df = pd.DataFrame({
        "time": pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC"),
        "qc_val": 28.0 + 0.3 * np.sin(np.arange(n)),
        "qc_flag": [1] * n,
        **{f"extra{i}": np.zeros(n) for i in range(16)},
    })
    df.to_csv(path, index=False)


def test_read_file_weather_station_schema(tmp_path):
    csv_path = tmp_path / "Davies Reef Weather Station.csv"
    _write_weather_station_csv(csv_path)
    df = _read_file(str(csv_path))
    assert df is not None
    assert set(df.columns) == {"time", "depth", "temperature", "qc_ok"}
    assert df["qc_ok"].all()
    assert df["depth"].iloc[0] == pytest.approx(3.0)
    assert 27.5 < df["temperature"].mean() < 28.5


def test_read_file_logger_schema_depth_from_filename(tmp_path):
    csv_path = tmp_path / "Davies Reef Logger @6m.csv"
    _write_logger_csv(csv_path, depth=6.0)
    df = _read_file(str(csv_path))
    assert df is not None
    assert set(df.columns) == {"time", "depth", "temperature", "qc_ok"}
    assert df["depth"].iloc[0] == pytest.approx(6.0)
    assert df["qc_ok"].all()


def test_read_file_skips_readme(tmp_path):
    readme = tmp_path / "readme.csv"
    readme.write_text("this is a readme file\n")
    assert _read_file(str(readme)) is None
