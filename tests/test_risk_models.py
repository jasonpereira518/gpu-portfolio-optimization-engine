"""Risk-model correctness. These run everywhere; GPU tests skip cleanly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.universe import synthetic_prices
from pipeline.cpu_baseline import (
    build_risk_model,
    compute_returns,
    ledoit_wolf_covariance,
    pca_factor_covariance,
    sample_covariance,
)
from pipeline.risk_model import TRADING_DAYS


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return synthetic_prices(40, n_days=1000, seed=42).prices


@pytest.fixture(scope="module")
def returns(prices) -> pd.DataFrame:
    return compute_returns(prices)


def test_returns_have_no_lookahead_shift(prices):
    """returns[t] must be (p[t]/p[t-1] - 1), not the forward return."""
    returns = compute_returns(prices)
    expected = prices.iloc[5] / prices.iloc[4] - 1.0
    np.testing.assert_allclose(returns.iloc[4].to_numpy(), expected.to_numpy(), rtol=1e-12)


def test_sample_covariance_matches_pandas(returns):
    got = sample_covariance(returns)
    want = returns.cov().to_numpy() * TRADING_DAYS
    np.testing.assert_allclose(got, want, rtol=1e-12)


@pytest.mark.parametrize("estimator", ["sample", "ledoit_wolf", "pca_factor"])
def test_covariance_is_symmetric_and_psd(prices, estimator):
    model = build_risk_model(prices, estimator=estimator)
    cov = model.symmetrized()
    np.testing.assert_allclose(cov, cov.T, atol=0)
    eigenvalues = np.linalg.eigvalsh(cov)
    assert eigenvalues.min() > -1e-10, f"{estimator} produced eigenvalue {eigenvalues.min()}"


def test_ledoit_wolf_matches_naive_loop_implementation(returns):
    """The vectorized beta^2 identity must equal the textbook loop form."""
    x = returns.to_numpy(dtype=np.float64)
    t, n = x.shape
    xc = x - x.mean(axis=0, keepdims=True)
    sample = (xc.T @ xc) / t
    mu = np.trace(sample) / n
    delta_sq = np.sum((sample - mu * np.eye(n)) ** 2) / n
    beta_sq = sum(np.sum((np.outer(xc[i], xc[i]) - sample) ** 2) / n for i in range(t)) / t**2
    shrink = min(beta_sq, delta_sq) / delta_sq
    want = (shrink * mu * np.eye(n) + (1 - shrink) * sample) * TRADING_DAYS

    np.testing.assert_allclose(ledoit_wolf_covariance(returns), want, rtol=1e-10)


def test_ledoit_wolf_is_better_conditioned_than_sample(returns):
    """The entire point of shrinkage: usable conditioning when T is not >> N."""
    short = returns.iloc[:60]  # 60 days, 40 assets — sample cov is near-singular
    assert np.linalg.cond(ledoit_wolf_covariance(short)) < np.linalg.cond(sample_covariance(short))


def test_pca_factor_recovers_diagonal_variance(returns):
    """Total variance per asset must be preserved by the factor decomposition."""
    cov = pca_factor_covariance(returns, n_factors=30)
    sample = sample_covariance(returns)
    # With k close to full rank, the factor model reproduces the diagonal closely.
    np.testing.assert_allclose(np.diag(cov), np.diag(sample), rtol=0.05)


def test_pca_factor_rejects_too_many_factors(returns):
    with pytest.raises(ValueError, match="cannot fit"):
        pca_factor_covariance(returns.iloc[:5], n_factors=50)


def test_risk_model_rejects_mismatched_shapes():
    from pipeline.risk_model import RiskModel

    with pytest.raises(ValueError, match="does not match"):
        RiskModel(np.zeros(3), np.zeros((2, 2)), ["a", "b", "c"], "sample", "cpu")
    with pytest.raises(ValueError, match="tickers"):
        RiskModel(np.zeros(3), np.zeros((3, 3)), ["a"], "sample", "cpu")


def test_nearest_psd_clips_negative_eigenvalues():
    from pipeline.risk_model import RiskModel

    cov = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3 and -1
    model = RiskModel(np.zeros(2), cov, ["a", "b"], "sample", "cpu")
    assert np.linalg.eigvalsh(model.nearest_psd()).min() >= -1e-12
