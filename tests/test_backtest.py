"""Backtest correctness, with lookahead bias as the headline test.

Lookahead is the failure mode that does not announce itself: the code runs, the
numbers are plausible, and the equity curve is simply too good. The test below
constructs a case where a lookahead bug would be unmistakable.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest

from backtest.engine import (
    equal_weight_benchmark,
    rebalance_dates,
    run_backtest,
)
from data.universe import synthetic_prices
from optimizer.mean_variance_cpu import solve_mean_variance_cpu
from optimizer.spec import PortfolioSpec
from pipeline.cpu_baseline import build_risk_model
from pipeline.risk_model import RiskModel


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return synthetic_prices(25, n_days=1600, seed=17).prices


RISK_FN = functools.partial(build_risk_model, estimator="ledoit_wolf")


def test_no_lookahead_the_risk_model_never_sees_future_prices(prices):
    """Record the last date handed to the risk model at each rebalance.

    If any risk model receives a price dated after its own rebalance date, the
    backtest is trading on information it could not have had.
    """
    seen: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def spying_risk_fn(trailing: pd.DataFrame) -> RiskModel:
        seen.append((trailing.index[0], trailing.index[-1]))
        return RISK_FN(trailing)

    result = run_backtest(
        prices, spying_risk_fn, solve_mean_variance_cpu,
        PortfolioSpec(risk_aversion=1.0, max_weight=0.15),
        frequency="QE", lookback_days=500,
    )

    assert len(seen) == len(result.rebalances)
    for (_, last_seen), record in zip(seen, result.rebalances):
        assert last_seen <= record.date, (
            f"lookahead: rebalance on {record.date} used data through {last_seen}"
        )


def test_perfect_foresight_would_be_detected(prices):
    """Sanity-check the detector above: a deliberately cheating run must beat honest.

    Without this, the lookahead test could be passing because the assertion is
    weak rather than because the engine is correct.
    """
    honest = run_backtest(
        prices, RISK_FN, solve_mean_variance_cpu,
        PortfolioSpec(risk_aversion=8.0, max_weight=0.15),
        frequency="QE", lookback_days=500, transaction_cost_bps=0.0, label="honest",
    )

    # A cheating risk model that estimates from the *next* 250 days.
    def cheating_risk_fn(trailing: pd.DataFrame) -> RiskModel:
        end = prices.index.get_loc(trailing.index[-1])
        future = prices.iloc[end : end + 250]
        return RISK_FN(future if len(future) > 60 else trailing)

    cheating = run_backtest(
        prices, cheating_risk_fn, solve_mean_variance_cpu,
        PortfolioSpec(risk_aversion=8.0, max_weight=0.15),
        frequency="QE", lookback_days=500, transaction_cost_bps=0.0, label="cheating",
    )

    assert cheating.sharpe() > honest.sharpe(), (
        "foresight did not help — the backtest may not be using the risk model at all"
    )


def test_weights_stay_fully_invested(prices):
    result = run_backtest(
        prices, RISK_FN, solve_mean_variance_cpu, PortfolioSpec(max_weight=0.20),
        frequency="QE", lookback_days=500,
    )
    for record in result.rebalances:
        assert abs(record.weights.sum() - 1.0) < 1e-6
        assert record.weights.min() >= -1e-9


def test_transaction_costs_reduce_returns(prices):
    spec = PortfolioSpec(risk_aversion=5.0, max_weight=0.20)
    free = run_backtest(prices, RISK_FN, solve_mean_variance_cpu, spec,
                        frequency="QE", lookback_days=500, transaction_cost_bps=0.0)
    costly = run_backtest(prices, RISK_FN, solve_mean_variance_cpu, spec,
                          frequency="QE", lookback_days=500, transaction_cost_bps=50.0)

    assert costly.total_return < free.total_return
    assert costly.total_cost_drag > 0
    # Gross curves ignore costs, so they must be identical across the two runs.
    np.testing.assert_allclose(
        costly.gross_equity.to_numpy(), free.gross_equity.to_numpy(), rtol=1e-9
    )


def test_rebalance_dates_are_real_trading_days(prices):
    marks = rebalance_dates(pd.DatetimeIndex(prices.index), "ME", lookback_days=500)
    assert marks, "expected at least one rebalance"
    assert all(d in prices.index for d in marks)
    assert all(d > prices.index[499] for d in marks)


def test_rebalance_dates_empty_when_history_too_short(prices):
    assert rebalance_dates(pd.DatetimeIndex(prices.index[:100]), "ME", lookback_days=500) == []


def test_backtest_raises_when_no_rebalance_possible(prices):
    with pytest.raises(ValueError, match="no rebalance dates"):
        run_backtest(prices.iloc[:100], RISK_FN, solve_mean_variance_cpu,
                     lookback_days=500)


def test_failed_solve_holds_previous_weights(prices):
    """A solver exception must not silently move the portfolio to cash."""
    calls = {"n": 0}

    def flaky_solver(model, spec):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated solver failure")
        return solve_mean_variance_cpu(model, spec)

    result = run_backtest(
        prices, RISK_FN, flaky_solver, PortfolioSpec(max_weight=0.20),
        frequency="QE", lookback_days=500,
    )
    assert calls["n"] > 2
    assert len(result.rebalances) == calls["n"] - 1  # one solve failed, rest recorded
    assert result.equity.iloc[-1] > 0


def test_equal_weight_benchmark_matches_mean_of_returns(prices):
    result = equal_weight_benchmark(prices)
    expected = prices.pct_change().fillna(0.0).mean(axis=1)
    np.testing.assert_allclose(result.returns.to_numpy(), expected.to_numpy(), rtol=1e-12)


def test_summary_statistics_are_self_consistent(prices):
    result = run_backtest(
        prices, RISK_FN, solve_mean_variance_cpu, PortfolioSpec(max_weight=0.20),
        frequency="QE", lookback_days=500,
    )
    assert result.max_drawdown <= 0
    assert result.annualized_vol > 0
    np.testing.assert_allclose(
        result.equity.iloc[-1], (1 + result.returns).prod(), rtol=1e-10
    )
