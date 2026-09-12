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

from data.universe import synthetic_prices
from optimizer.cuopt_compat import cuopt_available
from optimizer.mean_variance_cpu import solve_mean_variance_cpu
from optimizer.spec import PortfolioSpec
from optimizer.turnover_mip_cuopt import round_lots_greedy, solve_lot_rounding_cuopt
from pipeline.cpu_baseline import build_risk_model
from tests.fake_cuopt import FAKE_API

requires_cuopt = pytest.mark.skipif(not cuopt_available(), reason="cuOpt requires a CUDA host")

# cuOpt stops a MIP once it is within mip_relative_gap (default 1e-4) of its
# bound; the HiGHS reference solves to a zero gap. Twice the gap covers both.
MIP_RTOL = 2e-4

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


BOOK = 1e6  # portfolio value for the realistic-scale cases


@pytest.fixture(scope="module")
def universe():
    """40 synthetic names (last prices $10-$5,000) and their risk model."""
    prices = synthetic_prices(40, n_days=1260, seed=3).prices
    return prices.iloc[-1].to_numpy(), build_risk_model(prices, estimator="ledoit_wolf")


def _spec(turnover_budget):
    w_prev = np.full(40, 1.0 / 40)  # the book being rebalanced: equal weight
    return PortfolioSpec(risk_aversion=2.0, max_weight=0.15, turnover_budget=turnover_budget,
                         w_prev=w_prev if turnover_budget is not None else None), w_prev


@pytest.fixture(scope="module")
def rebalance(universe):
    """A rebalance at the scale the MIP meets in practice: a $1M book held
    equal-weight in whole shares, moving to a turnover-capped QP target."""
    last, model = universe
    spec, w_prev = _spec(0.25)
    target = solve_mean_variance_cpu(model, spec).weights
    return target, last, np.floor(w_prev * BOOK / last)


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


def test_a_trade_limit_can_always_keep_the_current_book():
    """Trading nothing is a legal answer whenever cash is allowed. C and D are
    targeted at zero but held, so a position bound derived from targets alone
    would force a trade in both — two trades against a limit of one — and
    report a problem with an obvious answer as infeasible."""
    prices = np.array([70.0, 45.0, 30.0, 25.0])
    target = np.array([0.50, 0.50, 0.0, 0.0])
    prev_shares = np.array([3.0, 8.0, 5.0, 4.0])

    solution = solve_lot_rounding_cuopt(target, prices, VALUE, prev_shares=prev_shares, max_trades=1,
                                        allow_cash=True, api=FAKE_API)

    assert solution.n_trades <= 1
    want = _brute_force(target, prices, VALUE, 1, True, prev_shares=prev_shares, max_trades=1)
    assert _objective(solution, VALUE) == pytest.approx(want, abs=1e-6)


def test_a_realistic_per_share_fee_never_leaves_the_mip_worse_off_than_ignoring_it(rebalance):
    """Ignoring costs is a feasible answer for the cost-aware model, so the
    cost-aware answer can be no worse on tracking error plus cost — at a real
    fee ($0.01 a share on a $1M book), not only the brute-force toy's. A cost
    term priced in the wrong units fails this, and so did HiGHS with presolve
    on, which is why the stand-in solves with presolve off."""
    target, prices, prev_shares = rebalance
    fee = 0.01

    def with_fees(solution):
        return solution.tracking_error + fee * np.abs(solution.shares - prev_shares).sum() / BOOK

    blind = solve_lot_rounding_cuopt(target, prices, BOOK, prev_shares=prev_shares,
                                     allow_cash=True, api=FAKE_API)
    aware = solve_lot_rounding_cuopt(target, prices, BOOK, prev_shares=prev_shares, cost_per_share=fee,
                                     allow_cash=True, api=FAKE_API)

    assert with_fees(aware) <= with_fees(blind) + 1e-6  # HiGHS's default absolute MIP gap


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


# ---------------------------------------------------------------------------
# GPU — skipped off-GPU
# ---------------------------------------------------------------------------

@requires_cuopt
@pytest.mark.parametrize("allow_cash", [False, True], ids=["fully_invested", "cash_allowed"])
@pytest.mark.parametrize("lot_size", [1, 10, 100])
@pytest.mark.parametrize("turnover_budget", [None, 0.25], ids=["no_turnover_cap", "turnover_0.25"])
def test_two_stage_qp_then_mip_on_gpu(universe, turnover_budget, lot_size, allow_cash):
    """Stage 1 on cuOpt's QP, stage 2 on cuOpt's MIP, held to HiGHS's proven
    optimum of the cash-allowed model built by the same code."""
    from optimizer.mean_variance_cuopt import solve_mean_variance_cuopt

    last, model = universe
    spec, w_prev = _spec(turnover_budget)
    target = solve_mean_variance_cuopt(model, spec).weights

    gpu = solve_lot_rounding_cuopt(target, last, BOOK, lot_size=lot_size, allow_cash=allow_cash,
                                   time_limit=30.0)
    cash_optimum = solve_lot_rounding_cuopt(target, last, BOOK, lot_size=lot_size, allow_cash=True,
                                            api=FAKE_API)

    assert gpu.status in ("Optimal", "FeasibleFound")
    assert np.all(np.isfinite(gpu.shares)) and np.all(gpu.shares >= 0)
    invested = gpu.weights.sum()
    assert invested <= 1.0 + 1e-5  # cuOpt's feasibility tolerance is 1e-6; rounding adds a little
    if allow_cash:
        assert gpu.tracking_error == pytest.approx(cash_optimum.tracking_error, rel=MIP_RTOL, abs=1e-7)
        greedy = round_lots_greedy(target, last, BOOK, lot_size=lot_size)
        assert gpu.tracking_error <= greedy.tracking_error * (1 + MIP_RTOL) + 1e-9
    else:
        assert invested >= 1.0 - 1e-5
        # Fully invested is the cash-allowed model plus a constraint, so it can
        # match that optimum at best; doing better would mean overspending.
        assert gpu.tracking_error >= cash_optimum.tracking_error * (1 - MIP_RTOL) - 1e-7

    if turnover_budget is not None:
        # Stage 2 does not enforce stage 1's turnover cap. By the triangle
        # inequality, rounding can overshoot it by at most the rounding error.
        turnover = float(np.abs(gpu.weights - w_prev).sum())
        assert turnover <= turnover_budget + gpu.tracking_error + 1e-5


@requires_cuopt
@pytest.mark.parametrize("case", ["costs", "trade_limit"])
def test_mip_trading_terms_on_gpu_match_highs(rebalance, case):
    """The trade variables and binary trade indicators, on cuOpt against HiGHS.

    The fee is far above any real one on purpose. Moving a share toward its
    target cuts tracking error by the share's price over the book value, so a
    fee changes the answer only once it approaches the cheapest prices here
    (~$10); at $20 it moves 14 of the 40 names.
    """
    target, last, prev_shares = rebalance
    kwargs = {"prev_shares": prev_shares, "allow_cash": True,
              "cost_per_share": 20.0 if case == "costs" else 0.0,
              "max_trades": 10 if case == "trade_limit" else None}

    gpu = solve_lot_rounding_cuopt(target, last, BOOK, time_limit=30.0, **kwargs)
    ref = solve_lot_rounding_cuopt(target, last, BOOK, api=FAKE_API, **kwargs)

    assert gpu.status in ("Optimal", "FeasibleFound")
    assert _objective(gpu, BOOK) == pytest.approx(_objective(ref, BOOK), rel=MIP_RTOL, abs=1e-7)
    if case == "trade_limit":
        assert gpu.n_trades <= 10
