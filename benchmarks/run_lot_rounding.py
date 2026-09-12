"""Stage 2 on the GPU: does the lot-rounding MIP beat greedy rounding?

    python -m benchmarks.run_lot_rounding --sizes 50 200 500 --out benchmarks/results/<gpu>-lots

For each universe size and turnover cap, stage 1 (the mean-variance QP) runs
once on CVXPY, so every host rounds the same target weights; the full
two-stage path on GPU — cuOpt QP into cuOpt MIP — is what
``tests/test_lot_rounding.py`` exercises. Each target is then rounded to
every lot size four ways:

  greedy               floor, then the largest remainders while cash lasts
  mip_fully_invested   cuOpt MIP, realized weights sum to exactly 1
  mip_cash_band        cuOpt MIP, between 1 - DEFAULT_MAX_CASH and 1 (the default)
  mip_cash_allowed     cuOpt MIP, sum <= 1 — greedy's own budget rule

Each row records tracking error (L1 to target), the cash left uninvested,
realized turnover against the equal-weight book being rebalanced, the
volatility drift the rounding caused (``report_drift``) and time. On a host
without cuOpt only the greedy rows exist: a baseline, not an answer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.harness import environment_metadata
from data.universe import synthetic_prices
from optimizer.cuopt_compat import CuOptApi, cuopt_available, load_cuopt
from optimizer.mean_variance_cpu import solve_mean_variance_cpu
from optimizer.spec import PortfolioSpec
from optimizer.turnover_mip_cuopt import (
    DEFAULT_MAX_CASH,
    LotSolution,
    report_drift,
    round_lots_greedy,
    solve_lot_rounding_cuopt,
)
from pipeline.cpu_baseline import build_risk_model

# Method -> max_cash: the two rules measured first, and the band between them.
MIP_BUDGETS = {"mip_fully_invested": 0.0, "mip_cash_band": DEFAULT_MAX_CASH, "mip_cash_allowed": 1.0}


def _row(case: dict, method: str, solver: str, solution: LotSolution, target: np.ndarray,
         w_prev: np.ndarray, cov: np.ndarray) -> dict:
    return {
        **case,
        "method": method,
        "solver": solver,
        "status": solution.status,
        "tracking_error": solution.tracking_error,
        "cash": 1.0 - float(solution.weights.sum()),
        "n_positions": int((solution.shares > 0).sum()),
        "target_turnover": float(np.abs(target - w_prev).sum()),
        "realized_turnover": float(np.abs(solution.weights - w_prev).sum()),
        "vol_drift_bps": report_drift(solution, target, cov)["vol_drift_bps"],
        "solve_s": solution.solve_time,
        "build_s": solution.build_time,
    }


def run_lot_rounding(
    sizes: list[int],
    lot_sizes: list[int],
    turnover_budgets: list[float | None],
    portfolio_value: float = 1e6,
    n_days: int = 1260,
    seed: int = 3,
    time_limit: float = 60.0,
    api: CuOptApi | None = None,
) -> pd.DataFrame:
    """One row per (size, turnover cap, lot size, rounding method).

    ``api`` defaults to the installed cuOpt; tests pass a stand-in.
    """
    with_mip = api is not None or cuopt_available()
    solver = f"cuopt-{(api or load_cuopt()).version}" if with_mip else None
    rows = []
    for n in sizes:
        prices = synthetic_prices(n, n_days=n_days, seed=seed).prices
        model = build_risk_model(prices, estimator="ledoit_wolf")
        cov, last = model.nearest_psd(), prices.iloc[-1].to_numpy()
        w_prev = np.full(n, 1.0 / n)
        for budget in turnover_budgets:
            # Same position cap as the benchmark sweep: 4x equal weight.
            spec = PortfolioSpec(risk_aversion=1.0, max_weight=min(1.0, 4.0 / n),
                                 turnover_budget=budget, w_prev=w_prev if budget is not None else None)
            target = solve_mean_variance_cpu(model, spec).weights
            for lot_size in lot_sizes:
                case = {"n_assets": n, "turnover_budget": budget, "lot_size": lot_size,
                        "portfolio_value": portfolio_value}
                greedy = round_lots_greedy(target, last, portfolio_value, lot_size=lot_size)
                rows.append(_row(case, "greedy", "numpy", greedy, target, w_prev, cov))
                if not with_mip:
                    continue
                for method, max_cash in MIP_BUDGETS.items():
                    try:
                        solution = solve_lot_rounding_cuopt(
                            target, last, portfolio_value, lot_size=lot_size, max_cash=max_cash,
                            time_limit=time_limit, api=api,
                        )
                    except RuntimeError as exc:  # no incumbent: recorded, not skipped
                        rows.append({**case, "method": method, "max_cash": max_cash, "status": str(exc)})
                        continue
                    rows.append({**_row(case, method, solver, solution, target, w_prev, cov),
                                 "max_cash": max_cash})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[50, 200, 500])
    parser.add_argument("--lots", type=int, nargs="+", default=[1, 10, 100])
    parser.add_argument("--turnover", type=float, nargs="+", default=[0.25],
                        help="turnover caps for stage 1; the uncapped case always runs too")
    parser.add_argument("--value", type=float, default=1e6, help="portfolio value in dollars")
    parser.add_argument("--time-limit", type=float, default=60.0, help="per MIP solve, seconds")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    meta = environment_metadata()
    rows = run_lot_rounding(args.sizes, args.lots, [None, *args.turnover],
                            portfolio_value=args.value, time_limit=args.time_limit)

    args.out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out / "lot_rounding.csv", index=False)
    (args.out / "environment.json").write_text(json.dumps(meta, indent=2))

    cols = ["n_assets", "turnover_budget", "lot_size", "method", "status", "tracking_error",
            "cash", "realized_turnover", "vol_drift_bps", "solve_s"]
    print(rows[[c for c in cols if c in rows]].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    if set(rows["method"]) == {"greedy"}:
        print("\nNOTE: cuOpt is not available on this host; only the greedy baseline ran.")
    print(f"\nwrote {args.out}/lot_rounding.csv, environment.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
