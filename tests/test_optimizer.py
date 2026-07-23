"""Optimizer correctness, checked against closed-form solutions where they exist.

The closed-form cases matter more than the CVXPY-vs-cuOpt comparison: if both
solvers were fed the same mis-stated problem they would agree with each other
and both be wrong. These tests have no solver on the reference side.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.universe import synthetic_prices
from optimizer.cuopt_compat import cuopt_available
from optimizer.mean_variance_cpu import (
    min_variance_closed_form,
    solve_mean_variance_cpu,
    tangency_closed_form,
)
from optimizer.spec import PortfolioSpec, objective_value
from pipeline.cpu_baseline import build_risk_model
from pipeline.risk_model import RiskModel

requires_cuopt = pytest.mark.skipif(not cuopt_available(), reason="cuOpt requires a CUDA host")


@pytest.fixture(scope="module")
def model() -> RiskModel:
    prices = synthetic_prices(30, n_days=1200, seed=5).prices
    return build_risk_model(prices, estimator="ledoit_wolf")


def two_asset_model(rho: float = 0.0) -> RiskModel:
    """A 2-asset problem small enough to solve with a pencil."""
    vol = np.array([0.20, 0.30])
    corr = np.array([[1.0, rho], [rho, 1.0]])
    cov = np.outer(vol, vol) * corr
    return RiskModel(np.array([0.08, 0.12]), cov, ["A", "B"], "manual", "cpu")


# ---------------------------------------------------------------------------
# Closed-form ground truth
# ---------------------------------------------------------------------------

def test_two_asset_min_variance_matches_hand_derivation():
    """For uncorrelated assets, min-variance weights are proportional to 1/variance.

        w_A = (1/0.04) / (1/0.04 + 1/0.09) = 0.6923...
    """
    model = two_asset_model(rho=0.0)
    spec = PortfolioSpec(risk_aversion=0.0, max_weight=1.0, min_weight=-1.0)
    solution = solve_mean_variance_cpu(model, spec)

    inv_var = 1.0 / np.array([0.04, 0.09])
    expected = inv_var / inv_var.sum()
    np.testing.assert_allclose(solution.weights, expected, atol=1e-6)


@pytest.mark.parametrize("rho", [-0.5, 0.0, 0.3, 0.8])
def test_min_variance_matches_closed_form(model, rho):
    cov = model.nearest_psd()
    spec = PortfolioSpec(risk_aversion=0.0, max_weight=1.0, min_weight=-1.0)
    solution = solve_mean_variance_cpu(model, spec)
    np.testing.assert_allclose(solution.weights, min_variance_closed_form(cov), atol=1e-5)


def test_solution_objective_is_at_least_as_good_as_closed_form(model):
    """The solver must not be beaten by the analytic solution on its own objective."""
    cov = model.nearest_psd()
    spec = PortfolioSpec(risk_aversion=0.0, max_weight=1.0, min_weight=-1.0)
    solution = solve_mean_variance_cpu(model, spec)
    analytic = min_variance_closed_form(cov)

    solver_obj = objective_value(solution.weights, cov, model.exp_returns, 0.0)
    analytic_obj = objective_value(analytic, cov, model.exp_returns, 0.0)
    assert solver_obj <= analytic_obj + 1e-9


def test_tangency_closed_form_is_budget_normalized(model):
    weights = tangency_closed_form(model.nearest_psd(), model.exp_returns)
    assert abs(weights.sum() - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Constraint handling
# ---------------------------------------------------------------------------

def test_weights_respect_box_and_budget(model):
    spec = PortfolioSpec(risk_aversion=1.0, max_weight=0.08)
    solution = solve_mean_variance_cpu(model, spec)
    assert solution.check(spec) == []


def test_position_cap_actually_binds(model):
    """A tight cap must change the answer, or the constraint is not being applied."""
    loose = solve_mean_variance_cpu(model, PortfolioSpec(risk_aversion=5.0, max_weight=1.0))
    tight = solve_mean_variance_cpu(model, PortfolioSpec(risk_aversion=5.0, max_weight=0.05))
    assert loose.weights.max() > 0.05 + 1e-6
    assert tight.weights.max() <= 0.05 + 1e-6


def test_turnover_budget_limits_trading(model):
    w_prev = np.full(model.n_assets, 1.0 / model.n_assets)
    spec = PortfolioSpec(
        risk_aversion=10.0, max_weight=0.20, turnover_budget=0.10, w_prev=w_prev
    )
    solution = solve_mean_variance_cpu(model, spec)
    assert float(np.abs(solution.weights - w_prev).sum()) <= 0.10 + 1e-5
    assert solution.check(spec) == []


def test_group_caps_are_enforced(model):
    labels = ["tech" if i % 2 == 0 else "energy" for i in range(model.n_assets)]
    spec = PortfolioSpec(
        risk_aversion=5.0, max_weight=0.30, group_labels=labels, group_max_weight=0.55
    )
    solution = solve_mean_variance_cpu(model, spec)
    assert solution.check(spec) == []


def test_higher_risk_aversion_seeks_more_return(model):
    """Monotonicity along the efficient frontier — a sign-error canary."""
    conservative = solve_mean_variance_cpu(model, PortfolioSpec(risk_aversion=0.0, max_weight=0.2))
    aggressive = solve_mean_variance_cpu(model, PortfolioSpec(risk_aversion=20.0, max_weight=0.2))

    mu = model.exp_returns
    cov = model.nearest_psd()
    assert mu @ aggressive.weights > mu @ conservative.weights
    assert aggressive.weights @ cov @ aggressive.weights >= conservative.weights @ cov @ conservative.weights - 1e-12


# ---------------------------------------------------------------------------
# Infeasibility is caught before the solver, with a readable message
# ---------------------------------------------------------------------------

def test_impossible_position_cap_is_rejected_early():
    model = two_asset_model()
    with pytest.raises(ValueError, match="infeasible"):
        solve_mean_variance_cpu(model, PortfolioSpec(max_weight=0.10))


def test_spec_rejects_turnover_budget_without_previous_weights():
    with pytest.raises(ValueError, match="requires w_prev"):
        PortfolioSpec(turnover_budget=0.1)


def test_spec_rejects_group_labels_without_cap():
    with pytest.raises(ValueError, match="must be set together"):
        PortfolioSpec(group_labels=["a", "b"])


# ---------------------------------------------------------------------------
# GPU parity — skipped off-GPU
# ---------------------------------------------------------------------------

@requires_cuopt
def test_cuopt_matches_cvxpy_objective(model):
    from optimizer.mean_variance_cuopt import solve_mean_variance_cuopt

    spec = PortfolioSpec(risk_aversion=1.0, max_weight=0.10)
    cpu = solve_mean_variance_cpu(model, spec)
    gpu = solve_mean_variance_cuopt(model, spec)

    cov, mu = model.nearest_psd(), model.exp_returns
    obj_cpu = objective_value(cpu.weights, cov, mu, spec.risk_aversion)
    obj_gpu = objective_value(gpu.weights, cov, mu, spec.risk_aversion)
    assert abs(obj_cpu - obj_gpu) < 1e-8
    assert gpu.check(spec) == []


@requires_cuopt
def test_cuopt_matches_closed_form_min_variance(model):
    from optimizer.mean_variance_cuopt import solve_mean_variance_cuopt

    spec = PortfolioSpec(risk_aversion=0.0, max_weight=1.0, min_weight=-1.0)
    gpu = solve_mean_variance_cuopt(model, spec)
    np.testing.assert_allclose(gpu.weights, min_variance_closed_form(model.nearest_psd()), atol=1e-4)


@requires_cuopt
def test_quadratic_convention_probe_returns_known_value():
    from optimizer.cuopt_compat import quadratic_convention

    assert quadratic_convention() in (1.0, 2.0)
