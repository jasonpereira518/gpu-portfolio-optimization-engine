"""Rolling-window backtest.

The one property this file exists to guarantee: **no lookahead**. A rebalance
dated t is decided using only price observations at or before t, and the
resulting weights earn returns strictly after t. That invariant is enforced
structurally (by slicing with ``prices.loc[:date]`` before the risk model ever
sees the data) and asserted in ``tests/test_backtest.py``, because a lookahead
bug does not crash — it just produces a beautiful equity curve.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from optimizer.spec import PortfolioSpec, Solution
from pipeline.risk_model import TRADING_DAYS, RiskModel

log = logging.getLogger(__name__)

RiskModelFn = Callable[[pd.DataFrame], RiskModel]
SolverFn = Callable[[RiskModel, PortfolioSpec], Solution]


@dataclass
class RebalanceRecord:
    date: pd.Timestamp
    weights: np.ndarray
    turnover: float
    transaction_cost: float
    objective: float
    solve_time: float
    n_assets: int
    status: str


@dataclass
class BacktestResult:
    equity: pd.Series  # cumulative growth of 1.0, net of costs
    gross_equity: pd.Series  # same, ignoring transaction costs
    returns: pd.Series  # daily net portfolio returns
    rebalances: list[RebalanceRecord] = field(default_factory=list)
    label: str = ""

    # ---- performance statistics ----

    @property
    def total_return(self) -> float:
        return float(self.equity.iloc[-1] - 1.0)

    @property
    def annualized_return(self) -> float:
        years = len(self.returns) / TRADING_DAYS
        if years <= 0:
            return 0.0
        return float(self.equity.iloc[-1] ** (1.0 / years) - 1.0)

    @property
    def annualized_vol(self) -> float:
        return float(self.returns.std() * np.sqrt(TRADING_DAYS))

    def sharpe(self, risk_free: float = 0.0) -> float:
        """Annualized Sharpe. ``risk_free`` is an annual rate."""
        excess = self.returns - risk_free / TRADING_DAYS
        sd = excess.std()
        if sd == 0:
            return 0.0
        return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))

    @property
    def max_drawdown(self) -> float:
        running_max = self.equity.cummax()
        return float((self.equity / running_max - 1.0).min())

    @property
    def avg_turnover(self) -> float:
        if not self.rebalances:
            return 0.0
        # Skip the first rebalance: going from all-cash to fully-invested is a
        # 100% turnover by definition and says nothing about strategy churn.
        later = [r.turnover for r in self.rebalances[1:]]
        return float(np.mean(later)) if later else 0.0

    @property
    def total_cost_drag(self) -> float:
        """Return given up to transaction costs, in return units."""
        return float(self.gross_equity.iloc[-1] - self.equity.iloc[-1])

    @property
    def total_solve_time(self) -> float:
        return float(sum(r.solve_time for r in self.rebalances))

    def summary(self) -> dict[str, float]:
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "annualized_vol": self.annualized_vol,
            "sharpe": self.sharpe(),
            "max_drawdown": self.max_drawdown,
            "avg_turnover": self.avg_turnover,
            "cost_drag": self.total_cost_drag,
            "n_rebalances": float(len(self.rebalances)),
            "total_solve_time": self.total_solve_time,
        }

    def render(self) -> str:
        s = self.summary()
        return (
            f"{self.label or 'backtest'}\n"
            f"  total return       {s['total_return']:>9.2%}\n"
            f"  annualized return  {s['annualized_return']:>9.2%}\n"
            f"  annualized vol     {s['annualized_vol']:>9.2%}\n"
            f"  Sharpe             {s['sharpe']:>9.2f}\n"
            f"  max drawdown       {s['max_drawdown']:>9.2%}\n"
            f"  avg turnover       {s['avg_turnover']:>9.2%}\n"
            f"  cost drag          {s['cost_drag']:>9.4f}\n"
            f"  rebalances         {int(s['n_rebalances']):>9d}\n"
            f"  total solve time   {s['total_solve_time']:>9.3f}s"
        )


def rebalance_dates(
    index: pd.DatetimeIndex, frequency: str = "ME", lookback_days: int = 756
) -> list[pd.Timestamp]:
    """Period-end trading dates that have at least ``lookback_days`` of history.

    ``frequency`` uses pandas offset aliases: "ME" month-end, "QE" quarter-end.
    The returned dates are actual index members, not calendar period ends, so
    holidays never produce a rebalance on a day the market was shut.
    """
    if len(index) <= lookback_days:
        return []
    eligible = index[lookback_days:]
    marks = pd.Series(eligible, index=eligible).resample(frequency).last().dropna()
    return [pd.Timestamp(d) for d in marks.to_numpy()]


def run_backtest(
    prices: pd.DataFrame,
    risk_model_fn: RiskModelFn,
    solver_fn: SolverFn,
    spec: PortfolioSpec | None = None,
    frequency: str = "ME",
    lookback_days: int = 756,
    transaction_cost_bps: float = 10.0,
    apply_turnover_budget: bool = False,
    label: str = "",
) -> BacktestResult:
    """Run a rolling rebalance backtest.

    Args:
        prices: dates x tickers adjusted closes.
        risk_model_fn: trailing prices -> RiskModel. Swap this to change CPU/GPU.
        solver_fn: (RiskModel, PortfolioSpec) -> Solution. Likewise.
        lookback_days: trailing window, in trading days, used to estimate risk.
        transaction_cost_bps: charged on one-way notional traded.
        apply_turnover_budget: carry ``spec.turnover_budget`` into each solve by
            injecting the previous weights. Off by default because it makes each
            rebalance depend on the last, which is realistic but makes CPU/GPU
            comparison sensitive to any single divergent solve.

    Returns:
        BacktestResult with net and gross equity curves and per-rebalance records.
    """
    base_spec = spec or PortfolioSpec()
    returns = prices.pct_change().fillna(0.0)
    dates = pd.DatetimeIndex(prices.index)

    marks = rebalance_dates(dates, frequency, lookback_days)
    if not marks:
        raise ValueError(
            f"no rebalance dates: {len(dates)} rows is not more than "
            f"lookback_days={lookback_days}"
        )

    cost_rate = transaction_cost_bps / 1e4
    records: list[RebalanceRecord] = []

    # Weight vector aligned to the full ticker list; the investable subset can
    # change between rebalances as names gain or lose sufficient history.
    all_tickers = list(prices.columns)
    ticker_pos = {t: i for i, t in enumerate(all_tickers)}
    current = np.zeros(len(all_tickers))

    net_returns = np.zeros(len(dates))
    gross_returns = np.zeros(len(dates))

    next_mark = 0
    for day_idx, date in enumerate(dates):
        # --- rebalance decision, using only data up to and including `date` ---
        if next_mark < len(marks) and date == marks[next_mark]:
            trailing = prices.loc[:date].iloc[-lookback_days:]
            try:
                model = risk_model_fn(trailing)
                spec_now = base_spec
                if apply_turnover_budget and base_spec.turnover_budget is not None:
                    prev_aligned = np.array([current[ticker_pos[t]] for t in model.tickers])
                    spec_now = PortfolioSpec(
                        risk_aversion=base_spec.risk_aversion,
                        max_weight=base_spec.max_weight,
                        min_weight=base_spec.min_weight,
                        turnover_budget=base_spec.turnover_budget,
                        w_prev=prev_aligned,
                        group_labels=base_spec.group_labels,
                        group_max_weight=base_spec.group_max_weight,
                    )
                solution = solver_fn(model, spec_now)
            except Exception as exc:
                # A failed solve holds the previous portfolio rather than going
                # to cash — dropping to cash on a solver hiccup would put a
                # numerical artifact straight into the equity curve.
                log.warning("rebalance %s failed (%s); holding previous weights", date.date(), exc)
                next_mark += 1
                continue

            target = np.zeros(len(all_tickers))
            for w, ticker in zip(solution.weights, model.tickers):
                target[ticker_pos[ticker]] = w

            turnover = float(np.abs(target - current).sum())
            cost = turnover * cost_rate
            net_returns[day_idx] -= cost

            records.append(
                RebalanceRecord(
                    date=date, weights=target, turnover=turnover, transaction_cost=cost,
                    objective=solution.objective, solve_time=solution.solve_time,
                    n_assets=model.n_assets, status=solution.status,
                )
            )
            current = target
            next_mark += 1

        # --- P&L: today's weights were set strictly before today's return ---
        elif day_idx > 0 and current.any():
            day_return = float(returns.iloc[day_idx].to_numpy() @ current)
            net_returns[day_idx] += day_return
            gross_returns[day_idx] += day_return

            # Weights drift with prices between rebalances; not renormalizing
            # would silently model a daily rebalance back to target.
            grown = current * (1.0 + returns.iloc[day_idx].to_numpy())
            total = grown.sum()
            if total > 0:
                current = grown / total

    net = pd.Series(net_returns, index=dates, name="net_return")
    gross = pd.Series(gross_returns, index=dates, name="gross_return")

    return BacktestResult(
        equity=(1.0 + net).cumprod(),
        gross_equity=(1.0 + gross).cumprod(),
        returns=net,
        rebalances=records,
        label=label,
    )


def equal_weight_benchmark(prices: pd.DataFrame, label: str = "equal-weight") -> BacktestResult:
    """1/N buy-and-hold, the benchmark mean-variance actually has to beat.

    Included because "the optimizer made money" is not a result on its own —
    equal weighting is a famously hard baseline for mean-variance to beat out
    of sample, and omitting it would overstate the strategy.
    """
    returns = prices.pct_change().fillna(0.0)
    n = prices.shape[1]
    port = returns.mean(axis=1) if n else returns.sum(axis=1)
    return BacktestResult(
        equity=(1.0 + port).cumprod(),
        gross_equity=(1.0 + port).cumprod(),
        returns=port.rename("net_return"),
        rebalances=[],
        label=label,
    )


def compare_results(results: list[BacktestResult]) -> pd.DataFrame:
    """Side-by-side summary table — the solution-quality half of the benchmark."""
    return pd.DataFrame({r.label or f"run{i}": r.summary() for i, r in enumerate(results)}).T
