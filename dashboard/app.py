"""Streamlit dashboard: weights, equity curves, benchmark table.

    streamlit run dashboard/app.py

Runs on CPU-only hosts and labels the backend it actually used, so a screenshot
of this page can never be mistaken for a GPU result when it was not one.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from backtest.engine import compare_results, equal_weight_benchmark, run_backtest
from data.universe import synthetic_prices
from optimizer.cuopt_compat import cuopt_available
from optimizer.mean_variance_cpu import solve_mean_variance_cpu
from optimizer.spec import PortfolioSpec
from pipeline.cpu_baseline import build_risk_model
from pipeline.gpu_pipeline import rapids_available

RESULTS_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "results"

st.set_page_config(page_title="GPU Portfolio & Risk Engine", layout="wide")
st.title("GPU Portfolio & Risk Decision Engine")

# --------------------------------------------------------------------------
# Backend status — shown first, deliberately
# --------------------------------------------------------------------------

gpu_data, gpu_solver = rapids_available(), cuopt_available()
cols = st.columns(3)
cols[0].metric("RAPIDS (cuDF/cuML)", "available" if gpu_data else "not available")
cols[1].metric("cuOpt", "available" if gpu_solver else "not available")
cols[2].metric("Active backend", "GPU" if (gpu_data and gpu_solver) else "CPU")

if not (gpu_data and gpu_solver):
    st.info(
        "No CUDA GPU detected on this host, so everything below is the CPU baseline "
        "(pandas + CVXPY). Numbers on this page are therefore not a speedup claim."
    )

# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Universe")
    n_assets = st.slider("Assets", 20, 1000, 200, step=20)
    n_days = st.slider("Trading days", 500, 3000, 2000, step=250)
    seed = st.number_input("Seed", value=17, step=1)

    st.header("Risk model")
    estimator = st.selectbox("Covariance estimator", ["ledoit_wolf", "sample", "pca_factor"])
    n_factors = st.slider("PCA factors", 2, 50, 10, disabled=estimator != "pca_factor")

    st.header("Optimization")
    risk_aversion = st.slider("Risk aversion", 0.0, 20.0, 2.0, step=0.5)
    max_weight_mult = st.slider("Position cap (x equal weight)", 1.0, 20.0, 8.0, step=0.5)

    st.header("Backtest")
    frequency = st.selectbox("Rebalance", ["QE", "ME"], format_func=
                             lambda f: {"QE": "Quarterly", "ME": "Monthly"}[f])
    lookback = st.slider("Lookback (trading days)", 250, 1260, 756, step=63)
    cost_bps = st.slider("Transaction cost (bps)", 0.0, 50.0, 10.0, step=1.0)

max_weight = min(1.0, max_weight_mult / n_assets)


@st.cache_data(show_spinner=False)
def load_prices(n: int, days: int, seed: int) -> pd.DataFrame:
    return synthetic_prices(n, n_days=days, seed=int(seed)).prices


prices = load_prices(n_assets, n_days, seed)
spec = PortfolioSpec(risk_aversion=risk_aversion, max_weight=max_weight)
risk_fn = functools.partial(build_risk_model, estimator=estimator, n_factors=n_factors)

# --------------------------------------------------------------------------
# Current portfolio
# --------------------------------------------------------------------------

st.header("Current optimal portfolio")

with st.spinner("Solving..."):
    model = risk_fn(prices.iloc[-lookback:])
    solution = solve_mean_variance_cpu(model, spec)

cov = model.nearest_psd()
port_vol = float(np.sqrt(solution.weights @ cov @ solution.weights))
port_ret = float(model.exp_returns @ solution.weights)

cols = st.columns(5)
cols[0].metric("Expected return", f"{port_ret:.2%}")
cols[1].metric("Expected volatility", f"{port_vol:.2%}")
cols[2].metric("Expected Sharpe", f"{port_ret / port_vol:.2f}" if port_vol else "—")
cols[3].metric("Positions held", int((solution.weights > 1e-6).sum()))
cols[4].metric("Solve time", f"{solution.solve_time * 1000:.1f} ms")

violations = solution.check(spec)
if violations:
    st.error("Constraint violations: " + "; ".join(violations))
else:
    st.success(f"All constraints satisfied ({solution.solver}, status={solution.status})")

top = (
    pd.Series(solution.weights, index=model.tickers)
    .sort_values(ascending=False)
    .head(25)
)
st.bar_chart(top)

# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------

st.header("Rolling backtest")

if st.button("Run backtest", type="primary"):
    with st.spinner("Backtesting..."):
        strategy = run_backtest(
            prices, risk_fn, solve_mean_variance_cpu, spec,
            frequency=frequency, lookback_days=lookback,
            transaction_cost_bps=cost_bps,
            label="mean-variance (CPU)" if not (gpu_data and gpu_solver) else "mean-variance (GPU)",
        )
        benchmark = equal_weight_benchmark(prices, label="equal-weight 1/N")

    st.line_chart(pd.DataFrame({r.label: r.equity for r in (strategy, benchmark)}))
    st.dataframe(compare_results([strategy, benchmark]).style.format("{:.4f}"))

    st.subheader("Per-rebalance detail")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "date": r.date.date(), "turnover": r.turnover, "cost": r.transaction_cost,
                    "objective": r.objective, "solve_time_s": r.solve_time, "status": r.status,
                }
                for r in strategy.rebalances
            ]
        )
    )

# --------------------------------------------------------------------------
# Benchmark table, if a sweep has been run
# --------------------------------------------------------------------------

st.header("Benchmark results")

speedup_csv = RESULTS_DIR / "speedup_table.csv"
if speedup_csv.exists():
    st.dataframe(pd.read_csv(speedup_csv))
    env_json = RESULTS_DIR / "environment.json"
    if env_json.exists():
        with st.expander("Environment the numbers were produced on"):
            st.json(env_json.read_text())
else:
    st.caption("No benchmark results yet — run `make bench` to generate them.")
