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
    compare_results,
    equal_weight_benchmark,
    rebalance_dates,
    run_backtest,
)
from data.universe import synthetic_prices
from optimizer.mean_variance_cpu import solve_mean_variance_cpu
from optimizer.spec import PortfolioSpec, Solution
from pipeline.cpu_baseline import build_risk_model
from pipeline.risk_model import RiskModel


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return synthetic_prices(25, n_days=1600, seed=17).prices


RISK_FN = functools.partial(build_risk_model, estimator="ledoit_wolf")


# ---------------------------------------------------------------------------
# Hand-checkable fixtures: 60 business days from Mon 2024-01-01. With
# lookback_days=5 and month-end rebalancing the rebalance dates are exactly
# 2024-01-31, 2024-02-29 and 2024-03-22 (the last row), so expected equity can
# be derived with a pencil.
# ---------------------------------------------------------------------------

def _two_assets_with_step(step_date: str, asset: str = "A", step: float = 0.10) -> pd.DataFrame:
    """Assets A and B flat at 100, except ``asset`` steps up by ``step`` on ``step_date``."""
    dates = pd.bdate_range("2024-01-01", periods=60)
    frame = pd.DataFrame(100.0, index=dates, columns=["A", "B"])
    frame.loc[pd.Timestamp(step_date):, asset] *= 1.0 + step
    return frame


def _tickers_only_risk_fn(trailing: pd.DataFrame) -> RiskModel:
    """The engine only needs the investable tickers from the risk model here."""
    n = trailing.shape[1]
    return RiskModel(np.zeros(n), np.eye(n), list(trailing.columns), "manual", "cpu")


def _scripted_solver(*weight_vectors):
    """Return the given weights on successive rebalances, repeating the last one."""
    remaining = [np.asarray(w, dtype=np.float64) for w in weight_vectors]

    def solver(model: RiskModel, spec: PortfolioSpec) -> Solution:
        weights = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return Solution(weights, 0.0, 0.0, 0.0, "scripted", "test", "scripted")

    return solver


def test_rebalance_day_return_is_earned_by_the_pre_rebalance_holdings():
    """A rebalance happens at the close, so that day's return belongs to the old weights.

    Hold A from 2024-01-31, A rises 10% on the 2024-02-29 rebalance day, then
    switch to B (flat). The 10% must be in the equity curve; dropping the
    rebalance day's P&L would leave terminal equity at exactly 1.0.
    """
    result = run_backtest(
        _two_assets_with_step("2024-02-29"), _tickers_only_risk_fn,
        _scripted_solver([1.0, 0.0], [0.0, 1.0]), PortfolioSpec(max_weight=1.0),
        frequency="ME", lookback_days=5, transaction_cost_bps=0.0,
    )
    assert [r.date for r in result.rebalances] == [
        pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29"), pd.Timestamp("2024-03-22"),
    ]
    assert result.equity.iloc[-1] == pytest.approx(1.10, rel=1e-12)


def test_trading_cost_is_charged_on_value_after_the_days_return():
    """Same path with 10 bps costs: 100% turnover on day one, 200% on the switch.

    Costs are a fraction of what the portfolio is worth when it trades — after
    that day's return — so the switch day compounds as (1 + 10%) * (1 - 0.2%).
    """
    result = run_backtest(
        _two_assets_with_step("2024-02-29"), _tickers_only_risk_fn,
        _scripted_solver([1.0, 0.0], [0.0, 1.0]), PortfolioSpec(max_weight=1.0),
        frequency="ME", lookback_days=5, transaction_cost_bps=10.0,
    )
    assert result.equity.iloc[-1] == pytest.approx(0.999 * 1.10 * 0.998, rel=1e-12)


def test_equal_weight_benchmark_rebalances_on_the_strategy_schedule_with_costs():
    """1/N bought on 2024-01-31, A rises 10% mid-February, back to 1/N on 2024-02-29.

    Drifted weights before the Feb rebalance are A=0.55/1.05, B=0.50/1.05, so
    that rebalance turns over 2*(0.55/1.05 - 0.5) of the book at 10 bps.
    """
    result = equal_weight_benchmark(
        _two_assets_with_step("2024-02-15"), _tickers_only_risk_fn,
        frequency="ME", lookback_days=5, transaction_cost_bps=10.0,
    )
    expected = 0.999 * 1.05 * (1.0 - 0.001 * 2.0 * (0.55 / 1.05 - 0.5))
    assert result.equity.iloc[-1] == pytest.approx(expected, rel=1e-12)
    assert [r.date for r in result.rebalances][:2] == [
        pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29"),
    ]


def test_strategy_and_benchmark_are_scored_over_the_same_window(prices):
    """Statistics start at the first rebalance, for the strategy and its benchmark alike.

    Scoring the strategy over its lookback period (all zero returns, before it
    holds anything) deflates its annualized return and Sharpe; scoring the
    benchmark from day one compares two different periods.
    """
    kwargs = {"frequency": "QE", "lookback_days": 500, "transaction_cost_bps": 10.0}
    strategy = run_backtest(prices, RISK_FN, solve_mean_variance_cpu,
                            PortfolioSpec(max_weight=0.20), **kwargs)
    benchmark = equal_weight_benchmark(prices, RISK_FN, **kwargs)

    assert strategy.returns.index[0] == strategy.rebalances[0].date
    assert strategy.returns.index.equals(benchmark.returns.index)


def test_compare_results_refuses_results_scored_over_different_windows(prices):
    kwargs = {"frequency": "QE", "transaction_cost_bps": 0.0}
    early = equal_weight_benchmark(prices, _tickers_only_risk_fn, lookback_days=500, **kwargs)
    late = equal_weight_benchmark(prices, _tickers_only_risk_fn, lookback_days=750, **kwargs)
    with pytest.raises(ValueError, match="different windows"):
        compare_results([early, late])


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
