"""CLI: run the rolling backtest on CPU and, where available, GPU.

    python -m backtest.run_backtest --source synthetic --n 200 --frequency QE
    python -m backtest.run_backtest --prices data/cache/prices_120_<key>.parquet

Runs the identical logical strategy through both backends and prints a
side-by-side table. The point is solution-quality parity: if the GPU pipeline
is faster but produces a different Sharpe, the speed number is worthless.

``--prices`` pins a price snapshot. yfinance revises adjusted closes as
dividends are paid, so a published real-data number is only reproducible
against the exact file it came from; its SHA-256 is recorded in
``backtest_config.json``.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    parser.add_argument("--prices", type=Path, default=None,
                        help="pinned price snapshot (Parquet, dates x tickers); overrides --source")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    if args.prices is not None:
        prices = pd.read_parquet(args.prices)
        universe = f"snapshot-{prices.shape[1]}"
        provenance = {"prices_file": args.prices.name, "prices_sha256": _sha256(args.prices)}
    else:
        data = load_universe(args.source, args.n, start=args.start)
        prices = data.prices
        if args.source == "synthetic" and len(prices) > args.days:
            prices = prices.iloc[-args.days :]
        universe = data.label
        provenance = {}

    cap = args.max_weight if args.max_weight is not None else min(1.0, 8.0 / prices.shape[1])
    spec = PortfolioSpec(risk_aversion=args.risk_aversion, max_weight=cap)
    log.info("universe=%s shape=%s max_weight=%.4f", universe, prices.shape, cap)

    from pipeline.cpu_baseline import build_risk_model

    cpu_risk_fn = functools.partial(build_risk_model, estimator=args.estimator)
    schedule = {
        "frequency": args.frequency,
        "lookback_days": args.lookback,
        "transaction_cost_bps": args.cost_bps,
    }
    results = [
        run_backtest(prices, cpu_risk_fn, solve_mean_variance_cpu, spec,
                     label="cpu (pandas + cvxpy)", **schedule)
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
                label="gpu (cudf + cuopt)",
                **schedule,
            )
        )
    else:
        log.warning("GPU stack unavailable; running the CPU backtest only")

    # Same schedule, costs and investable universe as the strategy; only the
    # weights differ.
    results.append(equal_weight_benchmark(prices, cpu_risk_fn, label="equal-weight 1/N", **schedule))

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
    # Paths are where files happened to live, not parameters of the experiment
    # (and as absolute paths they leak the local filesystem); the snapshot is
    # identified by name + hash instead. With a snapshot, the --source options
    # were never used, so recording them would misdescribe the run.
    unused = {"out", "prices"} | ({"source", "n", "start", "days"} if args.prices else set())
    config = {k: v for k, v in vars(args).items() if k not in unused}
    config.update(provenance)
    config.update({
        "universe": universe,
        "n_assets": prices.shape[1],
        "price_rows": f"{prices.index[0].date()}..{prices.index[-1].date()}",
        # compare_results above has already enforced one shared window.
        "scored_window": f"{results[0].returns.index[0].date()}..{results[0].returns.index[-1].date()}",
        "max_weight_applied": cap,
    })
    (args.out / "backtest_config.json").write_text(json.dumps(config, indent=2, default=str))
    print(f"\nwrote {args.out}/backtest_summary.csv, equity_curves.csv, backtest_config.json")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
