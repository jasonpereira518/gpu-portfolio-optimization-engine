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

| Component                                                                     | State                                                                                                    |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Data pipeline (yfinance + synthetic generator, Parquet cache, quality checks) | Complete, tested                                                                                         |
| CPU baseline (returns, rolling features, 3 covariance estimators)             | Complete, tested                                                                                         |
| CPU optimizer (CVXPY + Clarabel QP: box, budget, turnover, group caps)        | Complete, tested                                                                                         |
| Backtest engine (rolling rebalance, costs, no-lookahead, invested-window scoring) | Complete, tested                                                                                     |
| Benchmark harness (per-stage, warm-up separated, variance reported)           | Complete                                                                                                 |
| GPU pipeline (cuDF/CuPy for returns, rolling features, Ledoit-Wolf covariance) | Executed on an RTX 4060 Laptop GPU (WSL2); parity and benchmark results committed                       |
| cuOpt QP layer (box, budget, turnover, group caps)                           | Executed on an RTX 4060 Laptop GPU (WSL2); parity-checked against CVXPY, results committed               |
| cuOpt MIP layer (turnover lot-rounding, [`turnover_mip_cuopt.py`](optimizer/turnover_mip_cuopt.py)) | Model assembly tested off-GPU; **not yet executed on a GPU** — no test or benchmark exercises it yet |
| cuML PCA factor covariance estimator                                         | Tested off-GPU (`--estimator pca_factor`); **not yet executed on a GPU** — parity/benchmark runs so far used Ledoit-Wolf only |
| NIM explainer (stretch)                                                       | Written with an offline fallback, **not yet executed**                                                   |

GPU numbers below are from a single RTX 4060 Laptop GPU under WSL2 — see
[docs/setup-wsl2.md](docs/setup-wsl2.md) for the bring-up sequence. Rows still
marked "not yet executed" are code written against NVIDIA's published API and
guarded so that it fails with an actionable message rather than silently
falling back to CPU. Section
"[Reproducing the GPU results](#reproducing-the-gpu-results)" is the exact
sequence to fill in the missing rows.

### GPU results

Generated from the committed files in `benchmarks/results/` by
`python -m benchmarks.render_tables` — never typed — and only after the
parity checks pass on the same machine.

<!-- BEGIN GENERATED: gpu-speedup -->
**NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 592.82** — `ledoit_wolf` covariance — `benchmarks/results/rtx4060-wsl2/`

| stage | assets | CPU | GPU | speedup |
|---|---|---|---|---|
| h2d_transfer | 50 | — | 47.9 ms | — |
| h2d_transfer | 500 | — | 543.1 ms | — |
| h2d_transfer | 3000 | — | 3.72 s | — |
| features | 50 | 10.8 ms | 378.3 ms | 0.03× |
| features | 500 | 93.2 ms | 4.98 s | 0.02× |
| features | 3000 | 630.6 ms | 30.54 s | 0.02× |
| risk_model | 50 | 1.7 ms | 157.8 ms | 0.01× |
| risk_model | 500 | 20.4 ms | 2.14 s | 0.01× |
| risk_model | 3000 | 310.8 ms | 12.01 s | 0.03× |
| psd_repair | 50 | 0.0 ms | 0.0 ms | 0.87× |
| psd_repair | 500 | 0.6 ms | 0.3 ms | 1.90× |
| psd_repair | 3000 | 57.7 ms | 59.4 ms | 0.97× |
| solve | 50 | 3.9 ms | 81.5 ms | 0.05× |
| solve | 500 | 142.3 ms | 242.3 ms | 0.59× |
| solve | 3000 | 5.96 s | 2.41 s | 2.48× |
| solve_cvxpy | 50 | 3.9 ms | 61.1 ms | 0.06× |
| solve_cvxpy | 500 | 142.3 ms | 288.2 ms | 0.49× |
| solve_cvxpy | 3000 | 5.96 s | 4.77 s | 1.25× |

**NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 592.82** — `pca_factor` covariance — `benchmarks/results/rtx4060-wsl2-pca/`

| stage | assets | CPU | GPU | speedup |
|---|---|---|---|---|
| h2d_transfer | 50 | — | 50.3 ms | — |
| h2d_transfer | 500 | — | 551.1 ms | — |
| h2d_transfer | 3000 | — | 3.68 s | — |
| features | 50 | 11.8 ms | 350.5 ms | 0.03× |
| features | 500 | 91.7 ms | 4.76 s | 0.02× |
| features | 3000 | 626.4 ms | 32.06 s | 0.02× |
| risk_model | 50 | 9.6 ms | 159.8 ms | 0.06× |
| risk_model | 500 | 157.9 ms | 1.96 s | 0.08× |
| risk_model | 3000 | 4.59 s | 12.39 s | 0.37× |
| psd_repair | 50 | 0.0 ms | 0.0 ms | 3.09× |
| psd_repair | 500 | 0.5 ms | 0.3 ms | 1.81× |
| psd_repair | 3000 | 58.6 ms | 58.4 ms | 1.00× |
| solve | 50 | 7.3 ms | 82.0 ms | 0.09× |
| solve | 500 | 133.4 ms | 256.1 ms | 0.52× |
| solve | 3000 | 6.40 s | 2.41 s | 2.66× |
| solve_cvxpy | 50 | 7.3 ms | 62.5 ms | 0.12× |
| solve_cvxpy | 500 | 133.4 ms | 282.0 ms | 0.47× |
| solve_cvxpy | 3000 | 6.40 s | 4.76 s | 1.34× |
<!-- END GENERATED: gpu-speedup -->

### Lot rounding: MIP vs greedy

Stage 2 rounds the stage-1 weights to whole lots. Each row rounds one QP target
three ways and reports L1 tracking error to it, as a share of the book; the MIP
columns add, in parentheses, how that compares with greedy's. "Fully invested"
is the MIP's default budget rule (realized weights sum to exactly 1); "cash
allowed" is greedy's own rule (they sum to at most 1), under which greedy's
answer is always feasible for the MIP.

<!-- BEGIN GENERATED: lot-rounding -->
**NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 592.82** — $1M book — `benchmarks/results/rtx4060-wsl2-lots/`

| assets | turnover cap | lot | greedy | MIP, fully invested | MIP, cash allowed | cash left: greedy / MIP, cash allowed | MIP solve: fully invested / cash allowed |
|---|---|---|---|---|---|---|---|
| 50 | none | 1 | 0.55% | 0.55% (0.2% worse) | 0.54% (2.2% better) | 0.0011% / 0.013% | 2.61 s / 7.65 s |
| 50 | none | 10 | 6.1% | 6.1% (0.14% worse) | 5.1% (16% better) | 0.0083% / 2.5% | 4.23 s / 8.23 s |
| 50 | none | 100 | 40% | 41% (2.7% worse) | 38% (5.8% better) | 0.13% / 3.4% | 2.77 s / 3.10 s |
| 50 | 0.25 | 1 | 0.84% | 0.53% (36% better) | 0.52% (37% better) | 0% / 0.002% | 5.30 s / 7.95 s |
| 50 | 0.25 | 10 | 7% | 7% (0.061% worse) | 6.3% (10% better) | 0.0041% / 4.4% | 4.30 s / 4.67 s |
| 50 | 0.25 | 100 | 55% | 56% (1.1% worse) | 50% (9.7% better) | 0.034% / 32% | 9.56 s / 9.94 s |
| 200 | none | 1 | 0.94% | 0.95% (1.6% worse) | 0.93% (0.47% better) | 0% / 0.0048% | 3.92 s / 5.00 s |
| 200 | none | 10 | 11% | 11% (0.39% worse) | 11% (2.5% better) | 0% / 4.5% | 23.49 s / 4.65 s |
| 200 | none | 100 | 83% | 84% (1.7% worse) | 79% (4.9% better) | 0.076% / 34% | 12.97 s / 2.67 s |
| 200 | 0.25 | 1 | 1.3% | 1.3% (1.1% worse) † | 1.3% (0.22% better) † | 0% / 0.0029% | 60.05 s / 60.02 s |
| 200 | 0.25 | 10 | 14% | 14% (0.5% worse) † | 14% (0.83% better) | 0.0044% / 4.4% | 60.08 s / 5.09 s |
| 200 | 0.25 | 100 | 104% | 104% (0.0075% worse) | 88% (15% better) | 0.0082% / 70% | 36.38 s / 3.74 s |
| 500 | none | 1 | 4.7% | 4.7% (0.27% worse) † | 4.7% (0.25% better) | 0% / 0.04% | 60.18 s / 8.71 s |
| 500 | none | 10 | 52% | 52% (0.56% worse) † | 50% (3.7% better) | 0.007% / 18% | 60.03 s / 5.24 s |
| 500 | none | 100 | 172% | 172% (0.032% worse) † | 100% (42% better) | 0.055% / 100% | 60.03 s / 434.3 ms |
| 500 | 0.25 | 1 | 7.1% | 7.2% (1.3% worse) † | 7.1% (0.012% better) † | 0% / 0.0073% | 60.11 s / 60.05 s |
| 500 | 0.25 | 10 | 75% | 75% (0.16% worse) † | 68% (8.9% better) | 0.011% / 39% | 60.05 s / 3.50 s |
| 500 | 0.25 | 100 | 171% | 171% (0.061% worse) † | 100% (41% better) | 0.11% / 100% | 62.05 s / 1.06 s |

† hit the time limit: the best solution found is shown, not a proven optimum.
<!-- END GENERATED: lot-rounding -->

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

**3. The objective is built the way cuOpt actually reads it.**
cuOpt's `Problem.setObjective` zeroes every linear objective coefficient before
applying the expression it is given, so a return term passed as
`addVariable(obj=...)` is silently dropped and the solve becomes minimum-variance.
This project shipped exactly that bug until reading the solver source turned it
up ([details](docs/case-study.md#reading-the-solver-source-not-just-its-reference)).
The return term now travels inside the objective expression. The covariance goes
in as one sparse matrix, zero-padded over any auxiliary variables because cuOpt
adds a matrix-form quadratic positionally. Each linear term is a single
`LinearExpression` rather than an `expr = expr + term` chain that allocates n
intermediate objects. `tests/test_cuopt_formulation.py` checks the assembled
model against a stand-in with cuOpt's model-building semantics, so the
formulation is tested on machines without a GPU. Model build time is still
reported separately from solve time in every benchmark, because the dense n²
covariance hand-off is a real cost and hiding it inside "GPU time" would cut
both ways.

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
fact show up — so the test cannot pass by being vacuous. The same structural
approach covers scoring: every result is scored from its first rebalance,
the equal-weight benchmark runs through the same engine on the same schedule,
and `compare_results` refuses results scored over different dates — the fix
for an evaluation bug an earlier version of this project had
([erratum](docs/case-study.md#erratum-2026-09-12)).

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
make test      # GPU tests skip cleanly off-GPU
make parity    # CPU vs GPU numerical comparison
```

The checks that matter most have **no solver on the reference side**, so a bug
shared by CVXPY and cuOpt could not hide behind their agreement:

- Two uncorrelated assets: minimum-variance weights must equal `(1/σ²) / Σ(1/σ²)`.
- General minimum variance: must equal `Σ⁻¹1 / 1ᵀΣ⁻¹1`.
- The solver's objective must not be beaten by the analytic solution.
- Monotonicity along the efficient frontier (a sign-error canary).

For CPU-vs-GPU solution comparison, the **objective value** is held to 1e-6
(cuOpt's barrier solver converges to 1e-8 *relative* accuracy, so the absolute
gap grows with the objective) and per-name **weights** only to 1e-4 —
deliberately. Mean-variance problems with
many near-substitutable assets have a flat optimum, so two solvers can land on
visibly different weight vectors whose objectives agree to ten digits. Asserting
tight weight equality produces false failures, and a test that cries wolf is a
test that gets ignored.

---

## Reproducing the GPU results

Any CUDA GPU from Volta onward with at least 16 GB of system RAM (cuOpt's
minimum). On Windows with a GeForce RTX card, follow
[`docs/setup-wsl2.md`](docs/setup-wsl2.md), which also covers what WSL2 can and
cannot measure. The GPU stack is pinned to a single release (cuDF, cuML and
cuOpt 26.08) in `requirements-gpu.txt`:

```bash
pip install --extra-index-url=https://pypi.nvidia.com -r requirements-gpu.txt
```

Or in the container (base image pinned by digest):

```bash
docker build -t gpu-portfolio-engine . && docker run --gpus all -it --rm -v $(pwd):/workspace gpu-portfolio-engine
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

The PCA factor estimator — the one stage that runs cuML rather than cuDF/CuPy —
gets its own parity check and sweep, and stage 2 its MIP-vs-greedy comparison:

```bash
python -m pipeline.parity_tests --n 500 --days 2520 --estimator pca_factor
```

```bash
python -m benchmarks.run_benchmarks --sizes 50 500 3000 --days 2520 --runs 5 --estimator pca_factor --out benchmarks/results/<host>-pca
```

```bash
python -m benchmarks.run_lot_rounding --sizes 50 200 500 --out benchmarks/results/<host>-lots
```

Results land in `benchmarks/results/` as CSVs plus an `environment.json`
recording GPU model, driver, and every library version. Estimated cost for the
full sweep: 10–20 GPU-hours at $0.50–1.50/hr, so roughly **$10–30**.

Version-sensitive name lookups are isolated in `optimizer/cuopt_compat.py`,
which raises a message naming the docs page rather than an `AttributeError`
deep in the optimizer. Behavioral changes are a different matter — the
objective bug above was one — which is why parity runs before any timing.

---

## Known limitations

Stated here rather than discovered by a reader:

- **Survivorship bias.** The yfinance path uses the _current_ S&P 500 membership,
  so the backtest never holds a company that was delisted or acquired. This
  inflates returns. Fixing it properly needs point-in-time constituent data
  (CRSP/Compustat), which is not freely available.
- **Expected returns are historical means.** This is the standard textbook
  choice and also the standard reason mean-variance disappoints out of sample:
  sample means are a famously noisy return forecast. On the real-data snapshot
  the optimizer and a same-schedule equal-weight benchmark are level on Sharpe
  (0.93 vs 0.90, well inside the ~0.4 standard error of a 9-year estimate),
  with the optimizer taking about 8 points more volatility and six times the
  turnover — consistent with DeMiguel, Garlappi & Uppal (2009), and reported
  rather than tuned away.
- **The synthetic generator is a k-factor model with constant per-asset drift**,
  so it is generous to factor-based covariance estimators, and it makes
  historical means informative by construction — mean-variance "wins" there
  for that reason alone. It exists to reach universe sizes free data sources
  will not serve, and every result records which source produced it; synthetic
  and real numbers are never mixed in one table.
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
benchmarks/    harness.py, run_benchmarks.py, render_tables.py (docs tables from results/), results/
explainer/     nim_explainer.py
dashboard/     app.py
notebooks/     colab_gpu_runner.ipynb (second GPU data point on a free T4)
docs/          case-study.md, setup-wsl2.md (Windows 11 + WSL2 GPU bring-up)
tests/         test_data.py, test_risk_models.py, test_optimizer.py, test_backtest.py,
               test_cuopt_formulation.py + fake_cuopt.py, test_benchmarks.py, test_render_tables.py
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
