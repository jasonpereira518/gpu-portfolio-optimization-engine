"""The problem this project hands to cuOpt, checked without a GPU.

GPU parity against CVXPY (tests/test_optimizer.py) is the ground truth for what
cuOpt returns; it only runs on a CUDA host. These tests pin down the model the
optimizer *builds*, against a stand-in that follows cuOpt's model-building
semantics (tests/fake_cuopt.py), so a formulation bug is caught on any machine.
"""

from __future__ import annotations

import numpy as np

from optimizer.mean_variance_cuopt import build_mean_variance_problem
from optimizer.spec import PortfolioSpec
from tests.fake_cuopt import FAKE_API

COV = np.array([
    [0.04, 0.01, 0.00],
    [0.01, 0.09, 0.02],
    [0.00, 0.02, 0.16],
])
MU = np.array([0.08, 0.10, 0.12])


def test_objective_keeps_the_expected_return_term():
    """minimize w'Σw - λμ'w with λ=2 → linear coefficients -2μ.

    cuOpt's setObjective zeroes coefficients set earlier through
    addVariable(obj=...), so a return term passed that way is silently
    dropped and every solve becomes minimum-variance.
    """
    spec = PortfolioSpec(risk_aversion=2.0, max_weight=0.6)
    prob, _, _ = build_mean_variance_problem(COV, MU, spec, api=FAKE_API, scale=1.0)

    np.testing.assert_allclose(prob.linear_objective, [-0.16, -0.20, -0.24])
    np.testing.assert_allclose(prob.quadratic_objective, COV)


def test_turnover_auxiliaries_carry_no_risk_and_no_return():
    """With a turnover budget the model has 3 weights + 3 |Δw| auxiliaries.

    cuOpt adds a matrix-form Q positionally over all variables, so the 3x3
    covariance must be zero-padded to 6x6 with the weights in the leading
    block; the auxiliaries get neither risk nor return.
    """
    spec = PortfolioSpec(risk_aversion=2.0, max_weight=0.6, turnover_budget=0.2,
                         w_prev=np.array([0.5, 0.3, 0.2]))
    prob, _, _ = build_mean_variance_problem(COV, MU, spec, api=FAKE_API, scale=1.0)

    expected_q = np.zeros((6, 6))
    expected_q[:3, :3] = COV
    np.testing.assert_allclose(prob.quadratic_objective, expected_q)
    np.testing.assert_allclose(prob.linear_objective, [-0.16, -0.20, -0.24, 0.0, 0.0, 0.0])


def test_convention_scale_multiplies_only_the_risk_term():
    """If the solver read Q as ½x'Qx, the probe's factor of 2 goes on Σ, not on μ."""
    spec = PortfolioSpec(risk_aversion=2.0, max_weight=0.6)
    prob, _, _ = build_mean_variance_problem(COV, MU, spec, api=FAKE_API, scale=2.0)

    np.testing.assert_allclose(prob.quadratic_objective, 2.0 * COV)
    np.testing.assert_allclose(prob.linear_objective, [-0.16, -0.20, -0.24])
