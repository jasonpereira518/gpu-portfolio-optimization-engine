"""CLI: run the rolling backtest on CPU and, where available, GPU.

    python -m backtest.run_backtest --source synthetic --n 200 --frequency QE

Runs the identical logical strategy through both backends and prints a
side-by-side table. The point is solution-quality parity: if the GPU pipeline
is faster but produces a different Sharpe, the speed number is worthless.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
from pathlib import Path

import pandas as pd

from backtest.engine import compare_results, equal_weight_benchmark, run_backtest
from data.universe import load_universe
from optimizer.mean_variance_cpu import solve_mean_variance_cpu
from optimizer.spec import PortfolioSpec

log = logging.getLogger(__name__)
RESULTS_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "results"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["synthetic", "yfinance"], default="synthetic")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--days", type=int, default=2520, help="synthetic source only")
    parser.add_argument("--frequency", default="QE", help="ME month-end, QE quarter-end")
    parser.add_argument("--lookback", type=int, default=756, help="trading days of history")
    parser.add_argument("--estimator", default="ledoit_wolf",
                        choices=["sample", "ledoit_wolf", "pca_factor"])
    parser.add_argument("--risk-aversion", type=float, default=2.0)
    parser.add_argument("--max-weight", type=float, default=None)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    data = load_universe(args.source, args.n, start=args.start)
    prices = data.prices
    if args.source == "synthetic" and len(prices) > args.days:
        prices = prices.iloc[-args.days :]

    cap = args.max_weight if args.max_weight is not None else min(1.0, 8.0 / prices.shape[1])
    spec = PortfolioSpec(risk_aversion=args.risk_aversion, max_weight=cap)
    log.info("universe=%s shape=%s max_weight=%.4f", data.label, prices.shape, cap)

    from pipeline.cpu_baseline import build_risk_model

    results = [
        run_backtest(
            prices,
            functools.partial(build_risk_model, estimator=args.estimator),
            solve_mean_variance_cpu,
            spec,
            frequency=args.frequency,
            lookback_days=args.lookback,
            transaction_cost_bps=args.cost_bps,
            label="cpu (pandas + cvxpy)",
        )
    ]

    from optimizer.cuopt_compat import cuopt_available
    from pipeline.gpu_pipeline import rapids_available

    if rapids_available() and cuopt_available():
        from optimizer.mean_variance_cuopt import solve_mean_variance_cuopt
        from pipeline.gpu_pipeline import build_risk_model_gpu

        results.append(
            run_backtest(
                prices,
                functools.partial(build_risk_model_gpu, estimator=args.estimator),
                solve_mean_variance_cuopt,
                spec,
                frequency=args.frequency,
                lookback_days=args.lookback,
                transaction_cost_bps=args.cost_bps,
                label="gpu (cudf + cuopt)",
            )
        )
    else:
        log.warning("GPU stack unavailable; running the CPU backtest only")

    results.append(equal_weight_benchmark(prices, label="equal-weight 1/N"))

    for result in results:
        print()
        print(result.render())

    table = compare_results(results)
    print("\n" + table.to_string(float_format=lambda v: f"{v:.4f}"))

    if len(results) == 3:  # cpu, gpu, benchmark
        cpu, gpu = results[0], results[1]
        sharpe_gap = abs(cpu.sharpe() - gpu.sharpe())
        print(
            f"\nSolution-quality parity: Sharpe gap {sharpe_gap:.6f}, "
            f"terminal-equity gap {abs(cpu.equity.iloc[-1] - gpu.equity.iloc[-1]):.6f}"
        )
        print(
            f"Solve-time total: cpu {cpu.total_solve_time:.3f}s vs "
            f"gpu {gpu.total_solve_time:.3f}s "
            f"({cpu.total_solve_time / max(gpu.total_solve_time, 1e-9):.2f}x)"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "backtest_summary.csv")
    equity = pd.DataFrame({r.label: r.equity for r in results})
    equity.to_csv(args.out / "equity_curves.csv")
    (args.out / "backtest_config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    print(f"\nwrote {args.out}/backtest_summary.csv, equity_curves.csv, backtest_config.json")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
