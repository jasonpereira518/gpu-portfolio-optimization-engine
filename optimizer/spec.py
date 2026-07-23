"""Problem specification shared by the CVXPY and cuOpt formulations.

Both solvers read the same ``PortfolioSpec`` and return the same ``Solution``.
Any difference in their answers is then attributable to the solver, not to two
subtly different problems — which is the whole point of the parity check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PortfolioSpec:
    """A fully-invested mean-variance problem.

    minimize   w' Sigma w  -  risk_aversion * mu' w
    subject to sum(w) == 1
               min_weight <= w_i <= max_weight
               optional: |w - w_prev|_1 <= turnover_budget
               optional: per-group exposure caps

    ``risk_aversion`` multiplies the return term (rather than the risk term) so
    that 0 is the minimum-variance portfolio and larger values move along the
    efficient frontier toward return-seeking.

    ``min_weight`` defaults to 0 (long-only, the realistic case for the
    backtest). Setting it negative permits shorts, which is what the
    closed-form validation cases need — those have analytic solutions only
    without the non-negativity constraint.
    """

    risk_aversion: float = 1.0
    max_weight: float = 0.10
    min_weight: float = 0.0
    turnover_budget: float | None = None
    w_prev: np.ndarray | None = None
    group_labels: list[str] | None = None  # len n, e.g. GICS sector per asset
    group_max_weight: float | None = None

    def __post_init__(self) -> None:
        if self.max_weight <= 0:
            raise ValueError("max_weight must be positive")
        if self.min_weight > self.max_weight:
            raise ValueError(
                f"min_weight={self.min_weight} exceeds max_weight={self.max_weight}"
            )
        if self.turnover_budget is not None and self.w_prev is None:
            raise ValueError("turnover_budget requires w_prev")
        if (self.group_labels is None) != (self.group_max_weight is None):
            raise ValueError("group_labels and group_max_weight must be set together")

    def validate_for(self, n_assets: int) -> None:
        """Catch infeasible-by-construction specs before handing them to a solver.

        A solver reporting INFEASIBLE on a problem that was arithmetically
        impossible from the start is a wasted debugging hour; this turns it
        into an immediate, readable error.
        """
        if self.max_weight * n_assets < 1.0 - 1e-12:
            raise ValueError(
                f"infeasible: max_weight={self.max_weight} across {n_assets} assets caps "
                f"total weight at {self.max_weight * n_assets:.4f} < 1"
            )
        if self.min_weight * n_assets > 1.0 + 1e-12:
            raise ValueError(
                f"infeasible: min_weight={self.min_weight} across {n_assets} assets forces "
                f"total weight >= {self.min_weight * n_assets:.4f} > 1"
            )
        if self.w_prev is not None and len(self.w_prev) != n_assets:
            raise ValueError(f"w_prev has {len(self.w_prev)} entries, expected {n_assets}")
        if self.group_labels is not None and len(self.group_labels) != n_assets:
            raise ValueError(
                f"group_labels has {len(self.group_labels)} entries, expected {n_assets}"
            )

    def group_indices(self) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = {}
        for i, label in enumerate(self.group_labels or []):
            groups.setdefault(label, []).append(i)
        return groups


@dataclass(frozen=True)
class Solution:
    """A solved portfolio, plus enough metadata to audit the solve."""

    weights: np.ndarray
    objective: float
    solve_time: float  # solver-reported time, excludes model construction
    build_time: float  # time spent constructing the model
    status: str
    backend: str  # "cvxpy" | "cuopt"
    solver: str  # concrete solver name, e.g. "CLARABEL" | "PDLP"

    @property
    def total_time(self) -> float:
        return self.build_time + self.solve_time

    def check(self, spec: PortfolioSpec, tol: float = 1e-6) -> list[str]:
        """Return a list of constraint violations; empty means feasible.

        Solvers report OPTIMAL against their own internal tolerances, which are
        not always the tolerances you care about. This re-checks the returned
        weights against the spec directly.
        """
        w = self.weights
        violations = []
        if abs(w.sum() - 1.0) > tol:
            violations.append(f"budget: sum(w)={w.sum():.8f}")
        if w.min() < spec.min_weight - tol:
            violations.append(f"lower bound: min(w)={w.min():.8f}")
        if w.max() > spec.max_weight + tol:
            violations.append(f"upper bound: max(w)={w.max():.8f}")
        if spec.turnover_budget is not None and spec.w_prev is not None:
            turnover = float(np.abs(w - spec.w_prev).sum())
            if turnover > spec.turnover_budget + tol:
                violations.append(f"turnover: {turnover:.8f} > {spec.turnover_budget}")
        if spec.group_max_weight is not None:
            for label, idx in spec.group_indices().items():
                exposure = float(w[idx].sum())
                if exposure > spec.group_max_weight + tol:
                    violations.append(f"group {label}: {exposure:.8f}")
        return violations


def objective_value(weights: np.ndarray, cov: np.ndarray, mu: np.ndarray, risk_aversion: float) -> float:
    """Evaluate the mean-variance objective outside any solver.

    Used to compare CVXPY and cuOpt solutions on a common scale — comparing
    each solver's self-reported objective would compare two different internal
    scalings.
    """
    return float(weights @ cov @ weights - risk_aversion * (mu @ weights))
