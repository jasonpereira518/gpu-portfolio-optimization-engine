"""Full benchmark sweep: CPU vs GPU, per pipeline stage, across universe sizes.

    python -m benchmarks.run_benchmarks --sizes 50 500 3000 --runs 5

Runs whatever is available on the host. On a CPU-only machine it produces the
complete CPU column and marks the GPU column as unavailable — which is a
legitimate partial result, not a failure, and is labeled as such in the output
so it can never be mistaken for a speedup claim.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.harness import Timing, benchmark_stage, environment_metadata, speedup_table
from data.universe import synthetic_prices
from optimizer.spec import PortfolioSpec

log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def bench_cpu(prices: pd.DataFrame, estimator: str, spec: PortfolioSpec,
              n_runs: int) -> list[Timing]:
    from optimizer.mean_variance_cpu import solve_mean_variance_cpu
    from pipeline.cpu_baseline import build_risk_model, rolling_features

    n_days, n_assets = prices.shape
    common = {"n_assets": n_assets, "n_days": n_days, "n_runs": n_runs, "backend": "cpu"}
    timings = []

    _, t = benchmark_stage(rolling_features, prices, stage="features", **common)
    timings.append(t)

    model, t = benchmark_stage(
        build_risk_model, prices, estimator=estimator, stage="risk_model",
        extra={"estimator": estimator}, **common,
    )
    timings.append(t)

    _, t = benchmark_stage(
        solve_mean_variance_cpu, model, spec, stage="solve",
        extra={"solver": "cvxpy"}, **common,
    )
    timings.append(t)
    return timings


def bench_gpu(prices: pd.DataFrame, estimator: str, spec: PortfolioSpec,
              n_runs: int) -> list[Timing]:
    from pipeline.gpu_pipeline import (
        build_risk_model_gpu, rapids_available, rolling_features_gpu, to_gpu,
    )

    n_days, n_assets = prices.shape
    common = {"n_assets": n_assets, "n_days": n_days, "n_runs": n_runs, "backend": "gpu"}
    timings = []

    if rapids_available():
        # Host->device transfer is timed as its own stage. Excluding it would
        # be the single biggest way to overstate the GPU win on a pipeline that
        # starts from a pandas frame; folding it into every stage would
        # understate it. Reported separately, once.
        gdf, t = benchmark_stage(to_gpu, prices, stage="h2d_transfer", **common)
        timings.append(t)

        _, t = benchmark_stage(rolling_features_gpu, gdf, stage="features", **common)
        timings.append(t)

        model, t = benchmark_stage(
            build_risk_model_gpu, gdf, estimator=estimator, stage="risk_model",
            extra={"estimator": estimator}, **common,
        )
        timings.append(t)
    else:
        log.warning("RAPIDS unavailable; skipping GPU feature/risk stages")
        model = None

    from optimizer.cuopt_compat import cuopt_available

    if cuopt_available():
        from optimizer.mean_variance_cuopt import solve_mean_variance_cuopt

        if model is None:
            from pipeline.cpu_baseline import build_risk_model

            # cuOpt can still be benchmarked without RAPIDS; the risk model is
            # then CPU-built, which is noted so the row is not read as a
            # fully-GPU pipeline.
            model = build_risk_model(prices, estimator=estimator)

        _, t = benchmark_stage(
            solve_mean_variance_cuopt, model, spec, stage="solve",
            extra={"solver": "cuopt"}, **common,
        )
        timings.append(t)
    else:
        log.warning("cuOpt unavailable; skipping GPU solve stage")

    return timings


def run_sweep(
    sizes: list[int], n_days: int, estimator: str, n_runs: int,
    max_weight: float | None = None, risk_aversion: float = 1.0, seed: int = 11,
) -> tuple[pd.DataFrame, list[Timing]]:
    all_timings: list[Timing] = []

    for n in sizes:
        # max_weight must scale with n or the budget constraint is infeasible:
        # a 10% cap cannot fill a portfolio of 5 names. Default to 4x equal
        # weight, which keeps the constraint binding (and the problem
        # non-trivial) at every size.
        cap = max_weight if max_weight is not None else min(1.0, 4.0 / n)
        spec = PortfolioSpec(risk_aversion=risk_aversion, max_weight=cap)

        prices = synthetic_prices(n, n_days=n_days, seed=seed).prices
        log.info("=== n=%d assets x %d days (max_weight=%.4f) ===", n, n_days, cap)

        all_timings += bench_cpu(prices, estimator, spec, n_runs)
        all_timings += bench_gpu(prices, estimator, spec, n_runs)

    return pd.DataFrame([t.as_row() for t in all_timings]), all_timings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[50, 500, 3000])
    parser.add_argument("--days", type=int, default=2520, help="~10 years of trading days")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--estimator", default="ledoit_wolf",
                        choices=["sample", "ledoit_wolf", "pca_factor"])
    parser.add_argument("--max-weight", type=float, default=None)
    parser.add_argument("--risk-aversion", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    meta = environment_metadata()
    print("Environment:")
    for k, v in meta.items():
        print(f"  {k:<10} {v}")
    print()

    raw, timings = run_sweep(
        args.sizes, args.days, args.estimator, args.runs,
        max_weight=args.max_weight, risk_aversion=args.risk_aversion,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.out / "timings_raw.csv", index=False)
    (args.out / "environment.json").write_text(json.dumps(meta, indent=2))

    table = speedup_table(timings)
    table.to_csv(args.out / "speedup_table.csv", index=False)

    print("\nPer-stage timings (seconds; median of %d runs after one warm-up):" % args.runs)
    cols = ["stage", "backend", "n_assets", "median_s", "std_s", "cv", "warmup_s"]
    print(raw[cols].to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    # Title the summary table for what it actually contains. Calling a
    # single-column CPU table "Speedup" invites a screenshot of it being read
    # as a GPU result, which is the one mistake this whole harness exists to
    # prevent.
    backends = set(raw["backend"])
    if "gpu" in backends and "cpu" in backends:
        print("\nSpeedup (CPU median / GPU median):")
    else:
        present = "GPU" if "gpu" in backends else "CPU"
        print(f"\nPer-stage summary ({present}-only — no speedup, nothing to compare against):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    if "gpu" not in backends:
        print(
            "\nNOTE: no GPU backend was available on this host. The table above is a "
            "CPU-only baseline and contains no speedup claim."
        )
    print(f"\nwrote {args.out}/timings_raw.csv, speedup_table.csv, environment.json")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
