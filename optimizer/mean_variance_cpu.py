"""CPU reference optimizer: CVXPY.

This is the correctness oracle for the cuOpt formulation and the timing
baseline for the solve stage.
"""

from __future__ import annotations

import time

import cvxpy as cp
import numpy as np

from optimizer.spec import PortfolioSpec, Solution, objective_value
from pipeline.risk_model import RiskModel


def solve_mean_variance_cpu(
    risk_model: RiskModel,
    spec: PortfolioSpec | None = None,
    solver: str | None = None,
    psd_clip: bool = True,
    verbose: bool = False,
) -> Solution:
    """Solve the mean-variance QP with CVXPY.

    ``psd_clip`` eigenvalue-clips the covariance first. CVXPY otherwise
    rejects a numerically-indefinite matrix in ``quad_form`` with a DCP error,
    which is a data-conditioning problem masquerading as a modeling error.
    """
    spec = spec or PortfolioSpec()
    n = risk_model.n_assets
    spec.validate_for(n)

    build_t0 = time.perf_counter()

    cov = risk_model.nearest_psd() if psd_clip else risk_model.symmetrized()
    mu = risk_model.exp_returns

    w = cp.Variable(n)
    risk = cp.quad_form(w, cp.psd_wrap(cov))
    ret = mu @ w
    objective = cp.Minimize(risk - spec.risk_aversion * ret)

    constraints = [cp.sum(w) == 1, w >= spec.min_weight, w <= spec.max_weight]

    if spec.turnover_budget is not None:
        constraints.append(cp.norm1(w - spec.w_prev) <= spec.turnover_budget)

    if spec.group_max_weight is not None:
        for idx in spec.group_indices().values():
            constraints.append(cp.sum(w[idx]) <= spec.group_max_weight)

    problem = cp.Problem(objective, constraints)
    build_time = time.perf_counter() - build_t0

    solve_t0 = time.perf_counter()
    problem.solve(solver=solver, verbose=verbose)
    wall_solve = time.perf_counter() - solve_t0

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"CVXPY did not solve: status={problem.status}")

    weights = np.asarray(w.value, dtype=np.float64).ravel()
    # Tiny negatives (-1e-11) are solver noise, not short positions; clip and
    # renormalize so downstream P&L math sees an exactly valid portfolio.
    weights = np.clip(weights, spec.min_weight, spec.max_weight)
    weights = weights / weights.sum()

    stats = problem.solver_stats
    reported = getattr(stats, "solve_time", None)

    return Solution(
        weights=weights,
        objective=objective_value(weights, cov, mu, spec.risk_aversion),
        solve_time=float(reported) if reported else wall_solve,
        build_time=build_time,
        status=problem.status,
        backend="cvxpy",
        solver=stats.solver_name if stats else (solver or "default"),
    )


def min_variance_closed_form(cov: np.ndarray) -> np.ndarray:
    """Unconstrained-except-budget minimum-variance weights: Sigma^-1 1 / 1'Sigma^-1 1.

    Used in the test suite as a ground truth that involves no solver at all,
    so a bug shared by CVXPY and cuOpt could not hide behind agreement.
    """
    ones = np.ones(cov.shape[0])
    inv_cov_ones = np.linalg.solve(cov, ones)
    return inv_cov_ones / (ones @ inv_cov_ones)


def tangency_closed_form(cov: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Budget-constrained maximum-Sharpe weights (shorts allowed): Sigma^-1 mu normalized."""
    inv_cov_mu = np.linalg.solve(cov, mu)
    return inv_cov_mu / inv_cov_mu.sum()
