"""CPU reference pipeline: pandas + NumPy.

This is the correctness reference for the whole project. Every GPU result is
checked against these functions before any timing is reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.risk_model import TRADING_DAYS, RiskModel


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

def compute_returns(prices: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
    """Daily returns from an adjusted-close frame."""
    if method == "simple":
        returns = prices.pct_change()
    elif method == "log":
        returns = np.log(prices).diff()
    else:
        raise ValueError(f"unknown return method {method!r}")
    return returns.dropna(how="all").dropna(axis=1, how="all")


def rolling_features(prices: pd.DataFrame, windows: tuple[int, ...] = (21, 63, 252)) -> dict:
    """Rolling volatility and momentum — the bulk-array work cuDF accelerates.

    Kept separate from the risk model so the benchmark can time feature
    engineering independently of covariance estimation.
    """
    returns = compute_returns(prices)
    out: dict[str, pd.DataFrame] = {}
    for w in windows:
        out[f"vol_{w}"] = returns.rolling(w).std() * np.sqrt(TRADING_DAYS)
        out[f"mom_{w}"] = prices.pct_change(w)
    out["returns"] = returns
    return out


# --------------------------------------------------------------------------
# Covariance estimators
# --------------------------------------------------------------------------

def sample_covariance(returns: pd.DataFrame) -> np.ndarray:
    """Annualized sample covariance."""
    return returns.cov().to_numpy() * TRADING_DAYS


def ledoit_wolf_covariance(returns: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf shrinkage toward a scaled identity target.

    At realistic universe sizes T (days) is not much larger than N (assets),
    and the sample covariance is badly conditioned or outright singular there.
    Mean-variance optimization is notoriously sensitive to exactly those small
    eigenvalues, so a raw sample matrix produces extreme, unstable weights.
    Shrinkage is the standard fix and is implemented here in closed form
    (Ledoit & Wolf 2004) rather than pulled from sklearn, so the GPU version
    can mirror it line for line.

    The beta^2 term is written using the identity

        sum_t ||x_t x_t' - S||_F^2 = sum_t (x_t' x_t)^2 - T ||S||_F^2

    rather than as an explicit loop over T outer products. The loop form is
    O(T n^2) — about 2.3e10 operations at n=3,000 over ten years of data, i.e.
    minutes to hours — and benchmarking a GPU against a baseline written that
    way would be measuring the baseline's implementation, not the hardware.
    The CPU and GPU paths use the same algebra so the comparison is fair.
    """
    x = returns.to_numpy(dtype=np.float64)
    t, n = x.shape
    x = x - x.mean(axis=0, keepdims=True)

    sample = (x.T @ x) / t

    # Normalized Frobenius inner product: <A,B> = trace(A B^T) / n
    mu = np.trace(sample) / n
    delta_sq = np.sum((sample - mu * np.eye(n)) ** 2) / n

    row_sq_norms = np.einsum("ij,ij->i", x, x)  # x_t' x_t for each observation
    beta_sq = (np.sum(row_sq_norms**2) - t * np.sum(sample**2)) / (n * t**2)
    beta_sq = min(max(beta_sq, 0.0), delta_sq)

    shrink = 0.0 if delta_sq == 0 else beta_sq / delta_sq
    shrunk = shrink * mu * np.eye(n) + (1.0 - shrink) * sample
    return shrunk * TRADING_DAYS


def pca_factor_covariance(returns: pd.DataFrame, n_factors: int = 10) -> np.ndarray:
    """Factor-model covariance:  Sigma = B F B' + D.

    B  (n x k) factor loadings, from the top-k principal components of returns
    F  (k x k) factor covariance, diagonal by construction of the PCA basis
    D  (n x n) diagonal idiosyncratic variance = residual variance per asset

    The residual variance is computed as (total variance - explained variance)
    per asset and floored at zero: that is what keeps Sigma PSD even when the
    k-factor reconstruction slightly over-explains a given name.
    """
    x = returns.to_numpy(dtype=np.float64)
    t, n = x.shape
    k = n_factors
    # Silently capping k would change the risk model out from under the caller
    # and make two "10-factor" runs incomparable, so this is an error.
    if k > min(t, n) - 1 or k < 1:
        raise ValueError(
            f"cannot fit {n_factors} factors to a {t}x{n} return matrix "
            f"(max {min(t, n) - 1})"
        )

    xc = x - x.mean(axis=0, keepdims=True)

    # Economy SVD is the numerically stable route to PCA; components are rows
    # of Vt, factor scores are U*S.
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    loadings = vt[:k].T  # (n, k)
    factor_var = (s[:k] ** 2) / t  # variance of each factor score
    factor_cov = np.diag(factor_var)

    systematic = loadings @ factor_cov @ loadings.T
    total_var = np.einsum("ij,ij->j", xc, xc) / t
    idio_var = np.maximum(total_var - np.diag(systematic), 0.0)

    return (systematic + np.diag(idio_var)) * TRADING_DAYS


COV_ESTIMATORS = {
    "sample": sample_covariance,
    "ledoit_wolf": ledoit_wolf_covariance,
    "pca_factor": pca_factor_covariance,
}


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def build_risk_model(
    prices: pd.DataFrame,
    estimator: str = "ledoit_wolf",
    n_factors: int = 10,
) -> RiskModel:
    """prices (dates x tickers, adjusted close) -> RiskModel on CPU."""
    returns = compute_returns(prices).dropna(axis=1, how="any")
    if returns.empty:
        raise ValueError("no usable returns after dropping incomplete columns")

    exp_returns = returns.mean().to_numpy(dtype=np.float64) * TRADING_DAYS

    if estimator == "pca_factor":
        cov = pca_factor_covariance(returns, n_factors=n_factors)
    elif estimator in COV_ESTIMATORS:
        cov = COV_ESTIMATORS[estimator](returns)
    else:
        raise ValueError(f"unknown estimator {estimator!r}; choose from {list(COV_ESTIMATORS)}")

    return RiskModel(
        exp_returns=exp_returns,
        cov=cov,
        tickers=list(returns.columns),
        estimator=estimator,
        backend="cpu",
    )
