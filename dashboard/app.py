"""Streamlit dashboard: weights, equity curves, benchmark table.

    streamlit run dashboard/app.py

Every solve on this page runs the CPU baseline (pandas + CVXPY), on any host.
Labels are taken from the backend fields of the objects each solve actually
returned — never from which GPU libraries happen to be importable — so a
screenshot of this page can never be mistaken for a GPU result. GPU numbers
come from the benchmark suite, shown at the bottom.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import streamlit as st

from backtest.engine import compare_results, equal_weight_benchmark, run_backtest
from benchmarks.render_tables import RESULTS, gpu_sweeps
from data.universe import synthetic_prices
from optimizer.cuopt_compat import cuopt_available
from optimizer.mean_variance_cpu import solve_mean_variance_cpu
from optimizer.spec import PortfolioSpec
from pipeline.cpu_baseline import build_risk_model
from pipeline.gpu_pipeline import rapids_available

st.set_page_config(page_title="GPU Portfolio & Risk Engine", layout="wide")
st.title("GPU Portfolio & Risk Decision Engine")

# --------------------------------------------------------------------------
# Backend status — shown first, deliberately
# --------------------------------------------------------------------------

gpu_data, gpu_solver = rapids_available(), cuopt_available()
cols = st.columns(2)
cols[0].metric("cuDF/cuML on this host", "installed" if gpu_data else "not installed")
cols[1].metric("cuOpt on this host", "installed" if gpu_solver else "not installed")

st.info(
    "The solves on this page run the CPU baseline (pandas + CVXPY) on every host; "
    "the label under each result names the backend that produced it. Numbers here "
    "are not a speedup claim — GPU timings come from the benchmark suite below."
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

# Named from what ran, e.g. "cpu risk model + cvxpy/CLARABEL".
ran_on = f"{model.backend} risk model + {solution.backend}/{solution.solver}"

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
    st.success(f"All constraints satisfied ({ran_on}, status={solution.status})")

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
        schedule = {"frequency": frequency, "lookback_days": lookback,
                    "transaction_cost_bps": cost_bps}
        strategy = run_backtest(prices, risk_fn, solve_mean_variance_cpu, spec,
                                label=f"mean-variance ({ran_on})", **schedule)
        benchmark = equal_weight_benchmark(prices, risk_fn, **schedule)

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
# Benchmark tables, one per GPU sweep under benchmarks/results/<host>/
# --------------------------------------------------------------------------

st.header("Benchmark results")

# The sweeps the README renders, under the README's labels: the GPU and
# covariance estimator each sweep's own files recorded. A CPU-only sweep is a
# baseline, not a speedup result, so it is not listed.
sweeps = gpu_sweeps(RESULTS)
for sweep in sweeps:
    st.markdown(sweep.label)
    st.dataframe(sweep.table, hide_index=True)
    with st.expander("Environment the numbers were produced on"):
        st.json(sweep.environment)
if not sweeps:
    st.caption(
        "No GPU sweeps under `benchmarks/results/` yet — run one on a GPU host with "
        "`--out benchmarks/results/<host>` (see docs/setup-wsl2.md). A CPU-only sweep "
        "is a baseline, not a speedup result, so it is not listed here."
    )
