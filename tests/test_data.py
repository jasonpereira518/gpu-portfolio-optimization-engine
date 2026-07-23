"""Data generation and quality-check tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.universe import synthetic_prices
from data.validate_data import clean_prices, validate_prices


def test_synthetic_prices_are_deterministic():
    a = synthetic_prices(20, n_days=300, seed=1).prices
    b = synthetic_prices(20, n_days=300, seed=1).prices
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_prices_differ_across_seeds():
    a = synthetic_prices(20, n_days=300, seed=1).prices
    b = synthetic_prices(20, n_days=300, seed=2).prices
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_synthetic_prices_are_positive_and_correctly_shaped():
    data = synthetic_prices(37, n_days=250, seed=3)
    assert data.prices.shape == (250, 37)
    assert (data.prices.to_numpy() > 0).all()
    assert data.universe_size == 37
    assert data.label == "synthetic-37"


def test_synthetic_returns_have_factor_structure():
    """A few dominant eigenvalues, not a flat noise spectrum.

    If the generator produced i.i.d. noise the covariance matrix would be
    near-diagonal, the QP would be trivial, and the benchmark would be
    measuring the wrong problem.
    """
    returns = synthetic_prices(60, n_days=1500, seed=4).prices.pct_change().dropna()
    eigenvalues = np.sort(np.linalg.eigvalsh(returns.cov().to_numpy()))[::-1]
    assert eigenvalues[0] / eigenvalues[10] > 5.0


def test_validate_flags_all_nan_column():
    prices = synthetic_prices(5, n_days=100, seed=0).prices.copy()
    prices["SYN00002"] = np.nan
    report = validate_prices(prices)
    assert "SYN00002" in report.all_nan_columns
    assert not report.ok


def test_validate_flags_stale_column():
    prices = synthetic_prices(5, n_days=100, seed=0).prices.copy()
    prices["SYN00001"] = 42.0
    assert "SYN00001" in validate_prices(prices).stale_columns


def test_validate_flags_suspected_split():
    prices = synthetic_prices(5, n_days=100, seed=0).prices.copy()
    prices.iloc[50:, 0] /= 2.0  # unadjusted 2-for-1 split
    report = validate_prices(prices)
    assert "SYN00000" in report.suspected_splits


def test_validate_flags_nonpositive_prices():
    prices = synthetic_prices(4, n_days=50, seed=0).prices.copy()
    prices.iloc[10, 0] = -1.0
    report = validate_prices(prices)
    assert "SYN00000" in report.nonpositive_prices
    assert not report.ok


def test_validate_passes_clean_data():
    report = validate_prices(synthetic_prices(10, n_days=500, seed=0).prices)
    assert report.ok
    assert report.missing_fraction == 0.0
    assert "PASS" in report.render()


def test_clean_prices_drops_thin_columns():
    prices = synthetic_prices(6, n_days=200, seed=0).prices.copy()
    prices.iloc[:150, 0] = np.nan  # only 25% coverage
    cleaned = clean_prices(prices, min_coverage=0.95)
    assert "SYN00000" not in cleaned.columns
    assert cleaned.shape[1] == 5


def test_clean_prices_does_not_bridge_long_gaps():
    """Forward-fill is capped so a long halt cannot become fake zero volatility."""
    prices = synthetic_prices(3, n_days=100, seed=0).prices.copy()
    prices.iloc[40:60, 0] = np.nan
    cleaned = clean_prices(prices, min_coverage=0.5, ffill_limit=5)
    assert "SYN00000" not in cleaned.columns
