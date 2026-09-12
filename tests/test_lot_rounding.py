"""Stage 2 of the two-stage design: rounding weights to tradeable lots.

The CPU half runs the real ``solve_lot_rounding_cuopt`` against a stand-in
with cuOpt's model-building semantics whose ``solve()`` hands the model to
HiGHS (tests/fake_cuopt.py), so the MIP *formulation* is checked on any
machine. Expected answers come from hand derivation or brute-force
enumeration, never from a solver, so a formulation bug cannot hide behind two
solvers agreeing on the same wrong model.

The GPU half runs the whole two-stage pipeline — cuOpt QP, then cuOpt MIP —
and compares cuOpt's MIP against HiGHS on the identical model.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from optimizer.turnover_mip_cuopt import round_lots_greedy, solve_lot_rounding_cuopt
from tests.fake_cuopt import FAKE_API

# Three assets whose answers can be worked out with a pencil: one lot of each
# is 10%, 5% and 2% of a 1,000 portfolio, and the ideal holdings are 4.3, 6.2
# and 13 lots.
PRICES = np.array([100.0, 50.0, 20.0])
TARGET = np.array([0.43, 0.31, 0.26])
VALUE = 1_000.0


def _brute_force(target, prices, value, lot_size, allow_cash, prev_shares=None,
                 cost_per_share=0.0, max_trades=None):
    """Best objective over every lot vector, by enumeration.

    Searches up to the most lots each asset could hold on its own — a bound
    that owes nothing to the model's own position bounds.
    """
    lot_weight = prices * lot_size / value
    prev_lots = np.zeros(len(prices)) if prev_shares is None else prev_shares / lot_size
    limits = np.floor(1.0 / lot_weight).astype(int)
    lots = np.array(list(itertools.product(*(range(k + 1) for k in limits))), dtype=float)

    invested = lots @ lot_weight
    feasible = invested <= 1.0 + 1e-9
    if not allow_cash:
        feasible &= invested >= 1.0 - 1e-9
    traded = np.abs(lots - prev_lots)
    if max_trades is not None:
        feasible &= (traded > 1e-9).sum(axis=1) <= max_trades

    objective = (np.abs(lots * lot_weight - target).sum(axis=1)
                 + cost_per_share * lot_size * traded.sum(axis=1) / value)
    return float(objective[feasible].min())


def _objective(solution, value):
    """The MIP's objective re-evaluated from the returned holdings."""
    return solution.tracking_error + solution.transaction_cost / value


# ---------------------------------------------------------------------------
# Hand-derived answers
# ---------------------------------------------------------------------------

def test_fully_invested_rounding_matches_hand_derivation():
    """sum(w) == 1: 10a + 5b + 2c = 100 forces b even, and (4, 6, 15) is the
    closest such point: 0.03 + 0.01 under on A and B, 0.04 over on C."""
    solution = solve_lot_rounding_cuopt(TARGET, PRICES, VALUE, api=FAKE_API)

    np.testing.assert_array_equal(solution.shares, [4, 6, 15])
    assert solution.tracking_error == pytest.approx(0.08)
    assert solution.weights.sum() == pytest.approx(1.0)


def test_cash_allowed_rounding_matches_hand_derivation():
    """sum(w) <= 1: every asset can round to its nearest lot (4, 6, 13) and
    the 4% left over stays in cash."""
    solution = solve_lot_rounding_cuopt(TARGET, PRICES, VALUE, allow_cash=True, api=FAKE_API)

    np.testing.assert_array_equal(solution.shares, [4, 6, 13])
    assert solution.tracking_error == pytest.approx(0.04)
    assert solution.weights.sum() == pytest.approx(0.96)


def test_greedy_sits_between_the_two_budget_rules_on_the_hand_example():
    """Floor to (4, 6, 13), then only C's 2% lot fits the 4% remainder: (4, 6, 14).

    Its 0.06 beats the fully-invested MIP (0.08) — greedy is allowed to keep
    cash that the MIP is not — and loses to the MIP that may keep cash too (0.04).
    """
    greedy = round_lots_greedy(TARGET, PRICES, VALUE)

    np.testing.assert_array_equal(greedy.shares, [4, 6, 14])
    assert greedy.tracking_error == pytest.approx(0.06)


# ---------------------------------------------------------------------------
# The formulation against brute force
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("allow_cash", [False, True])
@pytest.mark.parametrize("case", ["plain", "costs", "trade_limit"])
def test_mip_optimum_matches_brute_force(allow_cash, case):
    """Every branch of the model — plain rounding, the transaction-cost term
    with existing holdings, and the binary trade-count limit — reaches the
    enumerated optimum."""
    prices = np.array([70.0, 45.0, 30.0, 25.0])
    target = np.array([0.35, 0.30, 0.20, 0.15])
    value, lot_size = 1_000.0, 1
    prev_shares, cost, max_trades = None, 0.0, None
    if case in ("costs", "trade_limit"):
        prev_shares = np.array([3.0, 8.0, 5.0, 4.0])  # 82% invested, far from target
        cost = 2.0 if case == "costs" else 0.0
        max_trades = 2 if case == "trade_limit" else None

    solution = solve_lot_rounding_cuopt(
        target, prices, value, prev_shares=prev_shares, lot_size=lot_size,
        cost_per_share=cost, max_trades=max_trades, allow_cash=allow_cash, api=FAKE_API,
    )
    want = _brute_force(target, prices, value, lot_size, allow_cash,
                        prev_shares=prev_shares, cost_per_share=cost, max_trades=max_trades)

    assert _objective(solution, value) == pytest.approx(want, abs=1e-6)
    if max_trades is not None:
        assert solution.n_trades <= max_trades


def test_a_mip_without_a_solution_raises_instead_of_returning_nan_holdings():
    """One asset whose lot is 30% of the book cannot be exactly 100% invested.

    cuOpt's getValue() returns NaN rather than raising when there is no
    incumbent, so only the termination status can say the solve failed.
    """
    with pytest.raises(RuntimeError, match="Infeasible"):
        solve_lot_rounding_cuopt(np.array([1.0]), np.array([300.0]), VALUE, api=FAKE_API)


@pytest.mark.parametrize("seed", range(5))
def test_cash_allowed_mip_is_never_worse_than_greedy(seed):
    """Greedy never overspends and never exceeds the model's position bounds,
    so its answer is always feasible for the cash-allowed MIP: the MIP losing
    on tracking error could only mean it stopped short of its optimum."""
    rng = np.random.default_rng(seed)
    n = 25
    target = rng.dirichlet(np.ones(n))
    prices = rng.uniform(5.0, 500.0, n)
    for lot_size in (1, 10):
        greedy = round_lots_greedy(target, prices, 250_000.0, lot_size=lot_size)
        mip = solve_lot_rounding_cuopt(target, prices, 250_000.0, lot_size=lot_size,
                                       allow_cash=True, api=FAKE_API)
        assert mip.tracking_error <= greedy.tracking_error + 1e-9
