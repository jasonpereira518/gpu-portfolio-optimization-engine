"""The contract between the feature/risk stage and the optimizer stage.

Both the CPU (pandas/NumPy) and GPU (cuDF/cuML) pipelines produce a
``RiskModel``. Everything downstream — CVXPY, cuOpt, backtest, parity tests —
consumes only this, which is what makes the two paths comparable at all.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

TRADING_DAYS = 252

# Estimators whose output is positive semidefinite by construction: Ledoit-Wolf
# is a convex combination of a PSD sample matrix and a positive multiple of the
# identity, and the PCA factor model is B F B' + D with F and D non-negative.
# Only the sample covariance (singular when n > T) can come out indefinite.
PSD_BY_CONSTRUCTION = frozenset({"ledoit_wolf", "pca_factor"})


@dataclass(frozen=True)
class RiskModel:
    """Annualized expected returns and covariance for one rebalance date."""

    exp_returns: np.ndarray  # (n,)
    cov: np.ndarray  # (n, n), symmetric PSD
    tickers: list[str]
    estimator: str  # "sample" | "ledoit_wolf" | "pca_factor"
    backend: str  # "cpu" | "gpu"
    psd_by_construction: bool = False  # lets nearest_psd() skip its eigendecomposition

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

        A sample covariance with n > T is singular and float error can push its
        small eigenvalues slightly negative, which makes the QP non-convex and
        the solve fail for reasons that have nothing to do with the solver.

        The repair is a full eigendecomposition — O(n^3), seconds at n=3,000 —
        so it is skipped for models that are PSD by construction (see
        ``PSD_BY_CONSTRUCTION``) or already repaired. Otherwise every solver
        would pay for it inside its timed solve.
        """
        sym = self.symmetrized()
        if self.psd_by_construction and epsilon == 0.0:
            return sym
        vals, vecs = np.linalg.eigh(sym)
        if vals.min() >= epsilon:
            return sym
        return (vecs * np.maximum(vals, epsilon)) @ vecs.T

    def with_repaired_cov(self) -> RiskModel:
        """This model with its covariance repaired once and flagged as PSD.

        Benchmarks time this as its own stage and hand the result to solvers,
        so CPU linear algebra is never booked as solver time on either backend.
        """
        return replace(self, cov=self.nearest_psd(), psd_by_construction=True)
