"""Stage 2 of the two-stage design: round continuous weights to tradeable lots.

Why two stages. cuOpt's MIP solver is, as of the 26.x releases, explicitly beta
and aimed at finding good *feasible* solutions to problems with **linear**
objectives — not mixed-integer *quadratic* programs. Forcing the mean-variance
risk term and the integrality constraints into a single MIQP is therefore
fighting the tool. The standard production pattern, used here, is:

    Stage 1 (QP, continuous)  -> ideal weights w*
    Stage 2 (MIP, linear)     -> tradeable lot counts closest to w*

Stage 2's objective is L1 tracking error to w* plus explicit transaction cost,
both linear once the absolute values are expanded with auxiliary variables. The
cost of the split is that the rounding is not risk-aware: it minimizes weight
distance, not variance distance. For lot sizes small relative to position sizes
that difference is negligible; ``report_drift`` quantifies it per rebalance so
the assumption is measured rather than asserted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from optimizer.cuopt_compat import is_optimal, load_cuopt, status_name


@dataclass(frozen=True)
class LotSolution:
    shares: np.ndarray  # integer share counts
    weights: np.ndarray  # realized weights after rounding
    tracking_error: float  # L1 distance to target weights
    n_trades: int
    transaction_cost: float
    solve_time: float
    build_time: float
    status: str


def solve_lot_rounding_cuopt(
    target_weights: np.ndarray,
    prices: np.ndarray,
    portfolio_value: float,
    prev_shares: np.ndarray | None = None,
    lot_size: int = 1,
    cost_per_share: float = 0.0,
    max_trades: int | None = None,
    time_limit: float = 60.0,
) -> LotSolution:
    """Round ``target_weights`` to integer lots with a cuOpt MIP.

    Variables
        n_i    integer lot count per asset (>= 0)
        d_i    continuous, >= |realized_weight_i - target_weight_i|
        t_i    continuous, >= |shares traded_i|  (only if costs/limits are on)
        y_i    binary trade indicator            (only if max_trades is set)

    Objective
        minimize  sum(d_i) + (cost_per_share / portfolio_value) * sum(t_i)

    Transaction cost is divided by portfolio value so both terms are in weight
    units and the sum is meaningful; mixing dollars and weights in one linear
    objective would make the relative weighting arbitrary.
    """
    api = load_cuopt()
    n = len(target_weights)
    prices = np.asarray(prices, dtype=np.float64)
    if len(prices) != n:
        raise ValueError(f"{len(prices)} prices for {n} target weights")
    if np.any(prices <= 0):
        raise ValueError("all prices must be positive")

    prev_shares = np.zeros(n) if prev_shares is None else np.asarray(prev_shares, dtype=np.float64)

    build_t0 = time.perf_counter()
    prob = api.Problem("Lot Rounding")

    # Weight contributed by one lot of asset i.
    lot_weight = (prices * lot_size) / portfolio_value
    # Upper bound: no position may exceed twice its target (plus a lot of slack
    # for tiny targets), which keeps the integer search space bounded.
    max_lots = np.maximum(np.ceil(2.0 * target_weights / np.maximum(lot_weight, 1e-12)), 1.0)

    lots = [
        prob.addVariable(lb=0.0, ub=float(max_lots[i]), vtype=api.VType.INTEGER, name=f"n_{i}")
        for i in range(n)
    ]
    # Objective coefficients are set once in setObjective below, never also via
    # addVariable(obj=...) — cuOpt would add both, silently double-weighting.
    dev = [prob.addVariable(lb=0.0, ub=1.0, name=f"d_{i}") for i in range(n)]

    # d_i >= +/- (lot_weight_i * n_i - target_i)
    for i in range(n):
        lw = float(lot_weight[i])
        tgt = float(target_weights[i])
        prob.addConstraint(dev[i] - lw * lots[i] >= -tgt, name=f"dev_pos_{i}")
        prob.addConstraint(dev[i] + lw * lots[i] >= tgt, name=f"dev_neg_{i}")

    # Budget: realized weights sum to 1.
    budget = api.LinearExpression(lots, [float(w) for w in lot_weight], 0.0)
    prob.addConstraint(budget == 1.0, name="budget")

    trade_vars: list = []
    if cost_per_share > 0.0 or max_trades is not None:
        prev_lots = prev_shares / lot_size
        for i in range(n):
            t = prob.addVariable(
                lb=0.0, ub=float(max_lots[i]) + abs(float(prev_lots[i])), name=f"t_{i}"
            )
            prob.addConstraint(t - lots[i] >= -float(prev_lots[i]), name=f"trd_pos_{i}")
            prob.addConstraint(t + lots[i] >= float(prev_lots[i]), name=f"trd_neg_{i}")
            trade_vars.append(t)

    if max_trades is not None:
        # y_i binary; t_i <= M_i * y_i forces y_i = 1 whenever asset i trades.
        indicators = []
        for i in range(n):
            y = prob.addVariable(lb=0.0, ub=1.0, vtype=api.VType.INTEGER, name=f"y_{i}")
            big_m = float(max_lots[i]) + abs(float(prev_shares[i] / lot_size)) + 1.0
            prob.addConstraint(trade_vars[i] - big_m * y <= 0.0, name=f"ind_{i}")
            indicators.append(y)
        count = api.LinearExpression(indicators, [1.0] * n, 0.0)
        prob.addConstraint(count <= float(max_trades), name="max_trades")

    prob.setObjective(
        api.LinearExpression(
            dev + trade_vars,
            [1.0] * n + [cost_per_share * lot_size / portfolio_value] * len(trade_vars),
            0.0,
        ),
        sense=api.MINIMIZE,
    )

    settings = api.SolverSettings()
    settings.set_parameter("time_limit", float(time_limit))
    build_time = time.perf_counter() - build_t0

    prob.solve(settings)

    if not is_optimal(prob):
        # A MIP that hit its time limit with an incumbent is still usable here —
        # any feasible lot vector is a tradeable portfolio. Surface it rather
        # than discarding the work, but say so in the status.
        try:
            _ = lots[0].getValue()
        except Exception as exc:
            raise RuntimeError(f"cuOpt MIP produced no feasible solution: {status_name(prob)}") from exc

    lot_counts = np.array([v.getValue() for v in lots], dtype=np.float64)
    shares = np.round(lot_counts) * lot_size
    realized = shares * prices / portfolio_value

    traded = np.abs(shares - prev_shares)
    return LotSolution(
        shares=shares,
        weights=realized,
        tracking_error=float(np.abs(realized - target_weights).sum()),
        n_trades=int((traded > 0).sum()),
        transaction_cost=float(traded.sum() * cost_per_share),
        solve_time=float(prob.SolveTime),
        build_time=build_time,
        status=status_name(prob),
    )


def round_lots_greedy(
    target_weights: np.ndarray,
    prices: np.ndarray,
    portfolio_value: float,
    lot_size: int = 1,
) -> LotSolution:
    """CPU reference for stage 2: floor to lots, then distribute the remainder.

    Largest-remainder allocation. This is the baseline the MIP must beat on
    tracking error — if it does not, the MIP layer is not earning its
    complexity, and reporting that honestly is more useful than shipping it.
    """
    build_t0 = time.perf_counter()
    prices = np.asarray(prices, dtype=np.float64)
    lot_weight = (prices * lot_size) / portfolio_value

    exact_lots = target_weights / lot_weight
    lots = np.floor(exact_lots)

    # Distribute leftover budget to the largest fractional remainders.
    remaining = 1.0 - float((lots * lot_weight).sum())
    order = np.argsort(-(exact_lots - lots))
    for i in order:
        if remaining < lot_weight[i]:
            continue
        lots[i] += 1
        remaining -= lot_weight[i]

    shares = lots * lot_size
    realized = shares * prices / portfolio_value
    return LotSolution(
        shares=shares,
        weights=realized,
        tracking_error=float(np.abs(realized - target_weights).sum()),
        n_trades=int((shares > 0).sum()),
        transaction_cost=0.0,
        solve_time=0.0,
        build_time=time.perf_counter() - build_t0,
        status="Greedy",
    )


def report_drift(
    lot_solution: LotSolution, target_weights: np.ndarray, cov: np.ndarray
) -> dict[str, float]:
    """Quantify what stage 2's linear objective costs in *risk* terms.

    The MIP minimizes weight distance; what actually matters is the change in
    portfolio variance. This measures the gap so the two-stage approximation is
    reported with a number attached instead of a hand-wave.
    """
    w_target = np.asarray(target_weights, dtype=np.float64)
    w_real = lot_solution.weights
    var_target = float(w_target @ cov @ w_target)
    var_real = float(w_real @ cov @ w_real)
    return {
        "l1_weight_drift": lot_solution.tracking_error,
        "target_vol": float(np.sqrt(max(var_target, 0.0))),
        "realized_vol": float(np.sqrt(max(var_real, 0.0))),
        "vol_drift_bps": float((np.sqrt(max(var_real, 0.0)) - np.sqrt(max(var_target, 0.0))) * 1e4),
    }
