"""The contract between the feature/risk stage and the optimizer stage.

Both the CPU (pandas/NumPy) and GPU (cuDF/cuML) pipelines produce a
``RiskModel``. Everything downstream — CVXPY, cuOpt, backtest, parity tests —
consumes only this, which is what makes the two paths comparable at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRADING_DAYS = 252


@dataclass(frozen=True)
class RiskModel:
    """Annualized expected returns and covariance for one rebalance date."""

    exp_returns: np.ndarray  # (n,)
    cov: np.ndarray  # (n, n), symmetric PSD
    tickers: list[str]
    estimator: str  # "sample" | "ledoit_wolf" | "pca_factor"
    backend: str  # "cpu" | "gpu"

    def __post_init__(self) -> None:
        n = len(self.exp_returns)
        if self.cov.shape != (n, n):
            raise ValueError(f"cov shape {self.cov.shape} does not match {n} assets")
        if len(self.tickers) != n:
            raise ValueError(f"{len(self.tickers)} tickers for {n} assets")

    @property
    def n_assets(self) -> int:
        return len(self.exp_returns)

    def symmetrized(self) -> np.ndarray:
        """Covariance forced exactly symmetric.

        Float32 GPU reductions and float64 CPU reductions both leave asymmetry
        at the 1e-8 level; CVXPY's ``quad_form`` and cuOpt's quadratic term
        both want an exactly symmetric matrix, so normalize once here rather
        than in each solver.
        """
        return 0.5 * (self.cov + self.cov.T)

    def nearest_psd(self, epsilon: float = 0.0) -> np.ndarray:
        """Clip negative eigenvalues.

        Shrinkage and factor estimators are PSD by construction, but a sample
        covariance with n > T is singular and float error can push its small
        eigenvalues slightly negative, which makes the QP non-convex and the
        solve fail for reasons that have nothing to do with the solver.
        """
        sym = self.symmetrized()
        vals, vecs = np.linalg.eigh(sym)
        if vals.min() >= epsilon:
            return sym
        return (vecs * np.maximum(vals, epsilon)) @ vecs.T
