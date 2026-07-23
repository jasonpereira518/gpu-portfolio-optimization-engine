# GPU Portfolio & Risk Decision Engine

Mean-variance portfolio optimization built twice — once on CPU (pandas + CVXPY)
and once on GPU (RAPIDS cuDF/cuML + NVIDIA cuOpt) — with a benchmark harness
that measures where the GPU actually wins, and a validation suite that proves
both paths produce the same answer first.

The CPU path is not a strawman. It is the correctness oracle for the GPU path
and the baseline for every timing claim, and it is written with the same care
(vectorized estimators, no accidental O(T·n²) loops) so that any speedup
measured is attributable to hardware rather than to a deliberately slow
reference.

---

## Status

| Component                                                                     | State                                                        |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------ |
| Data pipeline (yfinance + synthetic generator, Parquet cache, quality checks) | Complete, tested                                             |
| CPU baseline (returns, rolling features, 3 covariance estimators)             | Complete, tested                                             |
| CPU optimizer (CVXPY QP: box, budget, turnover, group caps)                   | Complete, tested                                             |
| Backtest engine (rolling rebalance, costs, no-lookahead enforcement)          | Complete, tested                                             |
| Benchmark harness (per-stage, warm-up separated, variance reported)           | Complete                                                     |
| GPU pipeline (cuDF/CuPy/cuML)                                                 | Written, **not yet executed** — no CUDA host available       |
| cuOpt QP + MIP layer                                                          | Written against the verified 26.02 API, **not yet executed** |
| NIM explainer (stretch)                                                       | Written with an offline fallback, **not yet executed**       |

**There are no speedup numbers in this README yet, and there will not be until
the GPU code has actually run.** Everything above marked "not yet executed" is
code written against NVIDIA's published API and guarded so that it fails with an
actionable message rather than silently falling back to CPU. Section
"[Reproducing the GPU results](#reproducing-the-gpu-results)" is the exact
sequence to fill in the missing column on a rented L4/A10.

---

## Quick start (CPU, no GPU required)

```bash
make venv
make test
```

```bash
make backtest
```

```bash
make bench
```

The full CPU pipeline — data, risk models, optimizer, backtest, benchmark
harness, dashboard — runs on any machine. Only the GPU column needs CUDA.

---

## Architecture

```
 Price data           cuDF / pandas         cuML / NumPy          cuOpt / CVXPY
┌──────────────┐     ┌───────────────┐    ┌────────────────┐    ┌───────────────┐
│ yfinance     │────▶│ returns,      │───▶│ covariance:    │───▶│ QP:           │
│ (Parquet     │     │ rolling vol,  │    │ sample /       │    │ mean-variance │
│  cache)      │     │ momentum      │    │ Ledoit-Wolf /  │    │ weights       │
│ synthetic    │     │               │    │ PCA factor     │    │               │
│ (k-factor)   │     └───────────────┘    └────────────────┘    └───────┬───────┘
└──────────────┘                                                        │
                                                                        ▼
┌──────────────┐     ┌───────────────┐    ┌────────────────┐    ┌───────────────┐
│ dashboard    │◀────│ benchmark     │◀───│ backtest:      │◀───│ MIP (opt.):   │
│ (Streamlit)  │     │ table +       │    │ rolling        │    │ lot rounding, │
│ + NIM        │     │ parity report │    │ rebalance, P&L │    │ turnover      │
│  explainer   │     │               │    │ vs 1/N         │    │ limits        │
└──────────────┘     └───────────────┘    └────────────────┘    └───────────────┘
```

Both backends implement the same two interfaces — `RiskModel` in
[`pipeline/risk_model.py`](pipeline/risk_model.py) and `PortfolioSpec` /
`Solution` in [`optimizer/spec.py`](optimizer/spec.py). The backtest and the
benchmark take the risk-model function and solver function as arguments, so
switching CPU→GPU changes two callables and nothing else. That is what makes
"same problem, different hardware" a structural guarantee rather than a claim.

---

## What is actually engineered here

Five decisions carry most of the technical weight:

**1. Ledoit-Wolf shrinkage, in closed form, on both sides.**
At realistic universe sizes the number of assets is comparable to the number of
trading days in the estimation window, so the sample covariance matrix is
ill-conditioned or singular — and mean-variance optimization is maximally
sensitive to exactly those small eigenvalues. The textbook formulation of the
shrinkage intensity loops over T observations forming n×n outer products: about
2.3 × 10¹⁰ operations at n = 3,000 over ten years. The identity

```
Σ_t ‖x_t x_tᵀ − S‖²_F  =  Σ_t (x_tᵀ x_t)²  −  T‖S‖²_F
```

collapses that to a single reduction over rows. Both
[`pipeline/cpu_baseline.py`](pipeline/cpu_baseline.py) and
[`pipeline/gpu_pipeline.py`](pipeline/gpu_pipeline.py) use it, and
`test_ledoit_wolf_matches_naive_loop_implementation` pins the vectorized version
to the literal loop form. Writing the CPU baseline the naive way would have
manufactured a large fake speedup.

**2. The quadratic-objective convention is probed, not assumed.**
Solvers disagree on whether a quadratic matrix means `xᵀQx` or `½xᵀQx`. Getting
it wrong silently halves or doubles the effective risk aversion — the resulting
portfolio still looks entirely plausible, so the bug survives inspection.
[`optimizer/cuopt_compat.py`](optimizer/cuopt_compat.py) determines the
convention at runtime by solving a one-variable problem whose answer is 0.5
under one convention and 1.0 under the other.

**3. Model construction is O(n), not O(n²).**
The widely-circulated cuOpt portfolio snippet builds the budget constraint and
the return term by chaining `expr = expr + term` across n variables, allocating
n intermediate expression objects. At 3,000 assets that Python-side cost swamps
the GPU solve it exists to feed. This implementation carries linear objective
coefficients on the variables (`addVariable(obj=...)`) and builds the budget
constraint as a single `LinearExpression`. Model build time is still reported
separately from solve time in every benchmark, because the dense n² covariance
hand-off is a real cost and hiding it inside "GPU time" would cut both ways.

**4. Two-stage QP → MIP, because cuOpt's MIP solver is linear-objective and beta.**
Forcing integrality and the quadratic risk term into one MIQP fights the tool.
Stage 1 solves the continuous QP for ideal weights; stage 2
([`optimizer/turnover_mip_cuopt.py`](optimizer/turnover_mip_cuopt.py)) solves a
linear MIP that rounds them to tradeable lots, minimizing L1 tracking error plus
explicit transaction cost. The cost of the split is that rounding is not
risk-aware — so `report_drift` measures the realized volatility gap per
rebalance, and a greedy largest-remainder rounder is included as the baseline
the MIP has to beat. If it does not beat it, that gets reported.

**5. No-lookahead is enforced structurally and tested adversarially.**
A lookahead bug does not crash; it just produces a beautiful equity curve. The
backtest slices `prices.loc[:date]` before the risk model sees anything, and
[`tests/test_backtest.py`](tests/test_backtest.py) both records the last date
every risk model was handed (asserting it never exceeds its own rebalance date)
_and_ runs a deliberately cheating variant to confirm that foresight would in
fact show up — so the test cannot pass by being vacuous.

---

## Benchmark methodology

`benchmarks/harness.py` encodes four rules, each corresponding to a common way
GPU benchmarks are unintentionally inflated:

- **Warm-up is reported, not dropped.** The first call pays CUDA context
  creation, kernel JIT and memory-pool growth. It is a separate `warmup_s`
  column, never averaged in and never quietly discarded.
- **Device synchronization before the clock stops.** CUDA is asynchronous;
  timing without a sync measures how fast Python enqueues kernels.
- **Variance is reported.** Five runs, with median, std and coefficient of
  variation. GPU timings are frequently bimodal and a bare mean hides that.
- **Stages are timed separately** — host→device transfer, feature engineering,
  covariance, solve — so a fast stage cannot carry a slow one, and the
  crossover point can be located per stage rather than in aggregate.

Universe sizes 50 / 500 / 3,000 over ~10 years of daily data. At n = 50 the GPU
is expected to _lose_ to the CPU: kernel launch and transfer overhead dominate a
problem that small. That crossover is a finding, not an embarrassment, and it
gets reported.

**Fairness constraints:** same physical machine for both columns, identical
random seeds, identical algorithms on both sides (not sklearn-vs-cuML), and
float64 on both sides. Float32 would give cuDF a ~2× bandwidth advantage but
loses roughly seven significant digits in the covariance — enough to change the
optimizer's answer. `dtype` is exposed so that tradeoff can be measured
separately rather than silently taken.

---

## Validation

Correctness is established before any timing is trusted:

```bash
make test      # 47 tests; GPU tests skip cleanly off-GPU
make parity    # CPU vs GPU numerical comparison
```

The checks that matter most have **no solver on the reference side**, so a bug
shared by CVXPY and cuOpt could not hide behind their agreement:

- Two uncorrelated assets: minimum-variance weights must equal `(1/σ²) / Σ(1/σ²)`.
- General minimum variance: must equal `Σ⁻¹1 / 1ᵀΣ⁻¹1`.
- The solver's objective must not be beaten by the analytic solution.
- Monotonicity along the efficient frontier (a sign-error canary).

For CPU-vs-GPU solution comparison, the **objective value** is held to 1e-8 and
per-name **weights** only to 1e-4 — deliberately. Mean-variance problems with
many near-substitutable assets have a flat optimum, so two solvers can land on
visibly different weight vectors whose objectives agree to ten digits. Asserting
tight weight equality produces false failures, and a test that cries wolf is a
test that gets ignored.

---

## Reproducing the GPU results

On a rented L4 / A10 / L40S (an H100 is unnecessary at this problem size):

```bash
docker build -t gpu-portfolio-engine . && docker run --gpus all -it --rm -v $(pwd):/workspace gpu-portfolio-engine
```

Or without Docker:

```bash
pip install --extra-index-url=https://pypi.nvidia.com -r requirements-gpu.txt
```

Then, **in this order** — parity before speed, always:

```bash
python -m pipeline.parity_tests --n 500 --days 2520
```

```bash
python -m benchmarks.run_benchmarks --sizes 50 500 3000 --days 2520 --runs 5
```

```bash
python -m backtest.run_backtest --source synthetic --n 500 --frequency QE
```

Results land in `benchmarks/results/` as CSVs plus an `environment.json`
recording GPU model, driver, and every library version. Estimated cost for the
full sweep: 10–20 GPU-hours at $0.50–1.50/hr, so roughly **$10–30**.

If cuOpt's API has moved again by then, `optimizer/cuopt_compat.py` is the only
file that should need editing — every version-sensitive lookup is isolated
there, and it raises a message naming the docs page rather than an
`AttributeError` deep in the optimizer.

---

## Known limitations

Stated here rather than discovered by a reader:

- **Survivorship bias.** The yfinance path uses the _current_ S&P 500 membership,
  so the backtest never holds a company that was delisted or acquired. This
  inflates returns. Fixing it properly needs point-in-time constituent data
  (CRSP/Compustat), which is not freely available.
- **Expected returns are historical means.** This is the standard textbook
  choice and also the standard reason mean-variance underperforms out of
  sample: sample means are a famously noisy return forecast. The included
  equal-weight benchmark frequently beats the optimizer on synthetic data, which
  is the expected result (cf. DeMiguel, Garlappi & Uppal, 2009) and is reported
  rather than tuned away.
- **The synthetic generator is a k-factor model**, so it is generous to
  factor-based covariance estimators by construction. It exists to reach
  universe sizes free data sources will not serve, and every result records
  which source produced it; synthetic and real numbers are never mixed in one
  table.
- **No transaction-cost model beyond linear bps.** No market impact, no bid-ask
  spread modeling, no borrow costs.
- **Daily data only.** Nothing intraday.

---

## Repository layout

```
data/          universe.py (yfinance + synthetic), validate_data.py, download_universe.py
pipeline/      risk_model.py (shared contract), cpu_baseline.py, gpu_pipeline.py, parity_tests.py
optimizer/     spec.py (shared contract), mean_variance_cpu.py, mean_variance_cuopt.py,
               turnover_mip_cuopt.py, cuopt_compat.py (version shim)
backtest/      engine.py, run_backtest.py
benchmarks/    harness.py, run_benchmarks.py, results/
explainer/     nim_explainer.py
dashboard/     app.py
tests/         test_data.py, test_risk_models.py, test_optimizer.py, test_backtest.py
```

---

## References

- NVIDIA cuOpt Python API — `Problem` / `QuadraticExpression` / `SolverSettings`
  signatures verified against the [26.02 LP/QP/MILP API reference](https://archive.docs.nvidia.com/cuopt/user-guide/26.02.00/cuopt-python/lp-qp-milp/lp-qp-milp-api.html)
  and [examples](https://archive.docs.nvidia.com/cuopt/user-guide/26.02.00/cuopt-python/lp-qp-milp/lp-qp-milp-examples.html) (July 2026).
- [NVIDIA cuOpt docs](https://docs.nvidia.com/cuopt/user-guide/latest/) · [github.com/NVIDIA/cuopt](https://github.com/NVIDIA/cuopt)
- [RAPIDS cuDF](https://docs.rapids.ai/api/cudf/stable/) — pandas-compatible GPU dataframes.
- Ledoit, O. & Wolf, M. (2004), "A well-conditioned estimator for large-dimensional
  covariance matrices," _Journal of Multivariate Analysis_ 88(2).
- DeMiguel, V., Garlappi, L. & Uppal, R. (2009), "Optimal Versus Naive
  Diversification," _Review of Financial Studies_ 22(5).
