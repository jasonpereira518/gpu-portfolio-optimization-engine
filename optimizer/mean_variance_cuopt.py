"""GPU mean-variance QP via NVIDIA cuOpt.

Solves exactly the ``PortfolioSpec`` that ``mean_variance_cpu`` solves, so the
two are directly comparable on both objective value and weights.

Model-construction notes (these are where a naive formulation breaks down,
and they are the actual engineering content of this file). Semantics are as
read from NVIDIA/cuopt v26.08.00 ``linear_programming/problem.py``;
``tests/test_cuopt_formulation.py`` pins the resulting model without a GPU.

1.  **The return term travels inside the objective expression.**
    ``Problem.setObjective`` zeroes every linear objective coefficient before
    applying the expression it is given, so coefficients passed earlier as
    ``addVariable(obj=...)`` are silently discarded and the solve degrades to
    minimum-variance — a portfolio that still looks entirely plausible.

2.  **Linear terms as one ``LinearExpression``** holding all n coefficients
    (budget, group caps, return term). Chaining ``expr = expr + mu_i * w_i``
    builds n intermediate expressions, quadratic in n. Coefficients are Python
    lists: ``QuadraticExpression + LinearExpression`` concatenates coefficient
    lists with ``+``, which an ndarray would turn into elementwise addition.

3.  **Quadratic term as one sparse matrix covering every variable.** cuOpt adds
    a matrix-form ``QuadraticExpression`` positionally onto a
    NumVariables x NumVariables matrix, so when turnover auxiliaries exist the
    covariance is zero-padded to cover them (the weights are added first). The
    matrix is handed over as a scipy sparse matrix rather than nested Python
    lists, which at n=3,000 would mean building 9M Python floats.

4.  **Objective convention is probed, not assumed** — see
    ``cuopt_compat.quadratic_convention``.

The n^2 covariance is still the scaling ceiling: at n=3,000 it is 9M nonzeros
however it is passed. Model build time is therefore reported separately from
solve time in every benchmark, since conflating them would credit the GPU
solver with a Python-side cost (or blame it for one).
"""

from __future__ import annotations

import time

import numpy as np
from scipy.sparse import coo_matrix

from optimizer.cuopt_compat import (
    CuOptApi,
    CuOptUnavailable,
    is_optimal,
    load_cuopt,
    quadratic_convention,
    status_name,
)
from optimizer.spec import PortfolioSpec, Solution, objective_value
from pipeline.risk_model import RiskModel


def build_mean_variance_problem(
    cov: np.ndarray,
    mu: np.ndarray,
    spec: PortfolioSpec,
    api: CuOptApi | None = None,
    scale: float | None = None,
):
    """Construct the cuOpt QP. Returns (problem, weight_vars, api).

    ``api`` and ``scale`` default to the installed cuOpt and its probed
    quadratic convention; tests pass a stand-in to check the model without a GPU.
    """
    api = api or load_cuopt()
    n = len(mu)

    prob = api.Problem("Mean-Variance Portfolio")

    # Weights first: the padded risk matrix below relies on them occupying the
    # leading n variable indices.
    weights = [
        prob.addVariable(lb=spec.min_weight, ub=spec.max_weight, name=f"w_{i}")
        for i in range(n)
    ]

    # Budget: sum(w) == 1
    budget = api.LinearExpression(weights, [1.0] * n, 0.0)
    prob.addConstraint(budget == 1.0, name="budget")

    # Group (e.g. sector) exposure caps
    if spec.group_max_weight is not None:
        for label, idx in spec.group_indices().items():
            group_vars = [weights[i] for i in idx]
            expr = api.LinearExpression(group_vars, [1.0] * len(idx), 0.0)
            prob.addConstraint(expr <= float(spec.group_max_weight), name=f"group_{label}")

    # Turnover: |w - w_prev|_1 <= budget, linearized with auxiliary variables
    #   u_i >= w_i - w_prev_i,  u_i >= w_prev_i - w_i,  sum(u) <= T
    # This is the standard LP-representable absolute value; it keeps the problem
    # a QP rather than forcing a mixed-integer formulation.
    if spec.turnover_budget is not None:
        w_prev = np.asarray(spec.w_prev, dtype=np.float64)
        aux = [prob.addVariable(lb=0.0, ub=2.0, name=f"u_{i}") for i in range(n)]
        for i in range(n):
            prob.addConstraint(aux[i] - weights[i] >= float(-w_prev[i]), name=f"tp_{i}")
            prob.addConstraint(aux[i] + weights[i] >= float(w_prev[i]), name=f"tn_{i}")
        turnover = api.LinearExpression(aux, [1.0] * n, 0.0)
        prob.addConstraint(turnover <= float(spec.turnover_budget), name="turnover")

    # Risk term w'Σw, scaled into cuOpt's convention so the objective is
    # comparable with CVXPY's, and zero-padded over any auxiliary variables.
    scale = quadratic_convention() if scale is None else scale
    risk = coo_matrix(np.asarray(cov, dtype=np.float64) * scale)
    num_vars = prob.NumVariables
    qmatrix = coo_matrix((risk.data, (risk.row, risk.col)), shape=(num_vars, num_vars))
    quad_risk = api.QuadraticExpression(qmatrix, prob.getVariables())

    # Return term -λμ'w inside the objective expression: setObjective discards
    # coefficients set any other way.
    ret = api.LinearExpression(weights, (-spec.risk_aversion * np.asarray(mu)).tolist(), 0.0)

    prob.setObjective(quad_risk + ret, sense=api.MINIMIZE)
    return prob, weights, api


def solve_mean_variance_cuopt(
    risk_model: RiskModel,
    spec: PortfolioSpec | None = None,
    time_limit: float = 60.0,
    optimality_tolerance: float | None = 1e-8,
    psd_clip: bool = True,
) -> Solution:
    """Solve the mean-variance QP on GPU. Requires a CUDA machine with cuOpt."""
    spec = spec or PortfolioSpec()
    n = risk_model.n_assets
    spec.validate_for(n)

    cov = risk_model.nearest_psd() if psd_clip else risk_model.symmetrized()
    mu = risk_model.exp_returns

    build_t0 = time.perf_counter()
    prob, weight_vars, api = build_mean_variance_problem(cov, mu, spec)

    settings = api.SolverSettings()
    settings.set_parameter("time_limit", float(time_limit))
    if optimality_tolerance is not None:
        try:
            settings.set_optimality_tolerance(optimality_tolerance)
        except Exception:  # older releases expose it only as a parameter
            settings.set_parameter("optimality_tolerance", optimality_tolerance)
    build_time = time.perf_counter() - build_t0

    solve_t0 = time.perf_counter()
    prob.solve(settings)
    wall_solve = time.perf_counter() - solve_t0

    if not is_optimal(prob):
        raise RuntimeError(
            f"cuOpt did not reach optimality: status={status_name(prob)}. "
            f"Increase time_limit (currently {time_limit}s) or loosen the tolerance."
        )

    weights = np.array([v.getValue() for v in weight_vars], dtype=np.float64)
    weights = np.clip(weights, spec.min_weight, spec.max_weight)
    weights = weights / weights.sum()

    return Solution(
        weights=weights,
        objective=objective_value(weights, cov, mu, spec.risk_aversion),
        solve_time=float(prob.SolveTime),
        build_time=build_time,
        status=status_name(prob),
        backend="cuopt",
        solver=f"cuopt-{api.version}",
    )


def solve_available(risk_model: RiskModel, spec: PortfolioSpec | None = None, **kwargs) -> Solution:
    """Solve on GPU if cuOpt is present, otherwise fall back to CVXPY.

    Convenience for the dashboard and the backtest driver only. The benchmark
    suite never calls this — an implicit fallback would silently turn a "GPU
    result" into a CPU one.
    """
    from optimizer.mean_variance_cpu import solve_mean_variance_cpu

    try:
        return solve_mean_variance_cuopt(risk_model, spec, **kwargs)
    except CuOptUnavailable:
        return solve_mean_variance_cpu(risk_model, spec)
