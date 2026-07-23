# GPU Portfolio & Risk Decision Engine
### Full Build Guide — cuOpt + RAPIDS (cuDF/cuML) Portfolio Optimization

*A standalone implementation guide. Grounded in NVIDIA's actual current cuOpt Python API (v26.x), RAPIDS installation docs, and NVIDIA's own `cuFOLIO` reference direction, verified July 22, 2026.*

---

## 0. What You're Building, In One Paragraph

A pipeline that takes historical price data for a large universe of stocks, computes returns/risk statistics on GPU (RAPIDS `cuDF`/`cuML`), solves a mean-variance portfolio optimization problem on GPU (NVIDIA `cuOpt`), backtests the resulting portfolios against a CPU-only baseline (pandas + SciPy/CVXPY), and reports a rigorous, reproducible speed and solution-quality comparison. This mirrors NVIDIA's own `cuFOLIO` toolkit direction ("GPU-accelerated portfolio optimization toolkit for building, backtesting, and scaling modern investment workflows with NVIDIA cuOpt and CUDA-X Data Science") — you're building a smaller, from-scratch version of a real, current NVIDIA product direction, not inventing a use case to justify the NVIDIA name.

**Why this is credible and not a "wrapper":** `cuOpt` is a GPU-native numerical solver (primal-dual hybrid gradient / interior-point methods running as CUDA kernels), not a hosted model API. You are formulating the optimization problem yourself — the engineering is in the constraint modeling, the data pipeline, and the benchmark rigor, exactly like a real quant-infrastructure project.

---

## 1. Prerequisites

- Python 3.11+ (cuOpt's Python wheels currently target 3.11–3.14; check the install selector at the URL in Section 2 before you commit to a Python version).
- A CUDA-capable NVIDIA GPU. For this problem size (up to a few thousand assets), a single consumer GPU (RTX 3060+ ) or a rented cloud GPU (L4, A10, or L40S — no need for H100/A100 at this scale) is sufficient.
- CUDA 12.x or 13.x driver installed (cuOpt ships separate wheels per CUDA major version: `cuopt-cu12` vs `cuopt-cu13`).
- Docker + NVIDIA Container Toolkit, if you go the container route (recommended for a clean, reproducible environment).
- Basic familiarity with pandas and NumPy — you'll be translating pandas code to cuDF, which is a near-drop-in API.
- A free account with a market-data source. `yfinance` (free, no key required) is the simplest starting point for daily OHLCV data; if you already have Bloomberg/Refinitiv/WRDS access through Carolina Investment Group, that's an even stronger, more credible data source to cite in your README.

---

## 2. Environment Setup

### Option A — Local/rented GPU with pip (recommended for iterating quickly)

```bash
# 1. Create a clean environment
python -m venv nvidia-portfolio-env
source nvidia-portfolio-env/bin/activate

# 2. Install RAPIDS (cuDF + cuML) — adjust cu12/cu13 to match your CUDA version
pip install --extra-index-url=https://pypi.nvidia.com \
    cudf-cu13==26.6.* cuml-cu13==26.6.* dask-cudf-cu13==26.6.*

# 3. Install cuOpt — adjust cu12/cu13 to match your CUDA version
pip install --extra-index-url=https://pypi.nvidia.com 'cuopt-cu13==26.2.*'

# 4. Supporting libraries
pip install pandas numpy scipy cvxpy yfinance matplotlib fastapi uvicorn pytest
```

> **Check before you run this:** cuOpt's install command and even its Python module layout have changed release-to-release in 2026 (the LP/QP/MILP module was recently reorganized into a "Convex Optimization" section in newer releases). Before writing any code, visit `https://docs.nvidia.com/cuopt/user-guide/latest/cuopt-python/quick-start.html` and copy the exact current install command and API import paths for the version you install. Pin your `cuopt-cuXX==` version once you've confirmed the API surface, so your code doesn't silently break on a later upgrade.

### Option B — Container (recommended for the final reproducible benchmark)

```bash
docker pull nvidia/cuopt:latest-cuda12.9-py3.13
docker run --gpus all -it --rm -v $(pwd):/workspace nvidia/cuopt:latest-cuda12.9-py3.13
```

Then `pip install` cuDF/cuML and the supporting libraries inside the container, or use a RAPIDS base container (`rapidsai/notebooks`) and add cuOpt on top — either order works, but building your own combined image and pinning versions in a `Dockerfile` is what makes your benchmark numbers reproducible by someone else, which is exactly what you want in the final GitHub repo.

### Verify the install (smoke test)

```python
python -c "
import cudf
from cuopt import routing
cost_matrix = cudf.DataFrame([[0,2,2,2],[2,0,2,2],[2,2,0,2],[2,2,2,0]], dtype='float32')
dm = routing.DataModel(cost_matrix.shape[0], 2, 3)
dm.add_cost_matrix(cost_matrix)
dm.add_transit_time_matrix(cost_matrix.copy(deep=True))
sol = routing.Solve(dm, routing.SolverSettings())
print(sol.get_route())
"
```

If this prints a route table, your GPU, CUDA, cuDF, and cuOpt install are all working together. Do this **before** writing any project code — debugging environment issues mid-project is the single biggest time sink on GPU projects.

---

## 3. Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  Historical      │────▶│  cuDF Feature     │────▶│  cuML Risk Model    │
│  Price Data      │     │  Pipeline         │     │  (covariance, PCA   │
│  (OHLCV, 10yr,   │     │  (returns, vol,   │     │  factor exposures)  │
│  N tickers)      │     │  factor signals)  │     │                     │
└─────────────────┘     └──────────────────┘     └──────────┬──────────┘
                                                              │
                                                              ▼
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  Dashboard /     │◀────│  Backtest Engine  │◀────│  cuOpt Solver       │
│  Report          │     │  (P&L, turnover,  │     │  (QP: mean-variance │
│  (+ optional NIM │     │  drawdown, vs.    │     │  weights; MIP:      │
│  Nemotron         │     │  CPU baseline)    │     │  lot-size/turnover  │
│  explainer)       │     │                   │     │  constraints)       │
└─────────────────┘     └──────────────────┘     └───────────────────┘
```

**Data flow:** raw OHLCV/fundamentals → cuDF cleaning and feature engineering (returns, rolling volatility, factor exposures) → cuML covariance/factor model estimation → cuOpt QP solve (portfolio weights) → optional cuOpt MIP post-processing (lot sizes, turnover limits) → backtest simulator → dashboard + benchmark report.

**Parallel CPU baseline:** the exact same logical pipeline, implemented in pandas + NumPy + SciPy/CVXPY, run on the same machine's CPU. This is not optional — without it, you have no speedup claim, only a demo.

---

## 4. Step-by-Step Build Instructions

### Phase 1 — Data Acquisition (Days 1-2)

1. Pick a universe: start with the S&P 500 constituent list (freely available) or a smaller 50-100 ticker subset while you're debugging, then scale up once the pipeline works.
2. Pull 10+ years of daily OHLCV data via `yfinance` (or your CIG data access, if available) and cache it locally as Parquet — don't re-download on every run.
3. Pull or construct a simple fundamentals/factor dataset if you want factor-based risk modeling (market cap, sector, P/E) — this can be as simple as sector classification from the ticker list if you want to keep scope tight.
4. Write a data-quality check script: flag missing days, delistings, and stock splits/adjustments — a benchmark built on bad data isn't credible.

### Phase 2 — CPU Baseline First (Days 3-5)

Build the **entire logical pipeline in pandas + SciPy/CVXPY before touching a GPU.** This de-risks the project (correctness first) and gives you the exact baseline you'll need for every benchmark claim later.

```python
import pandas as pd
import numpy as np
import cvxpy as cp

def compute_returns_and_cov(price_df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """price_df: dates x tickers, adjusted close."""
    returns = price_df.pct_change().dropna()
    exp_returns = returns.mean() * 252          # annualized
    cov_matrix = returns.cov() * 252            # annualized
    return exp_returns, cov_matrix

def solve_mean_variance_cpu(exp_returns, cov_matrix, risk_aversion=1.0, max_weight=0.10):
    n = len(exp_returns)
    w = cp.Variable(n)
    risk = cp.quad_form(w, cov_matrix.values)
    ret = exp_returns.values @ w
    objective = cp.Minimize(risk - risk_aversion * ret)
    constraints = [cp.sum(w) == 1, w >= 0, w <= max_weight]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return w.value, problem.solver_stats.solve_time
```

Confirm this produces sane portfolios (weights sum to 1, no single position above your cap, reasonable risk/return tradeoff) on a small universe before moving on.

### Phase 3 — Port to cuDF, Validate Numerical Parity (Days 6-8)

The fastest path: use **cuDF's pandas Accelerator Mode**, which requires close to zero code changes.

```python
%load_ext cudf.pandas   # in a notebook; use `python -m cudf.pandas your_script.py` from the CLI
import pandas as pd     # this `pd` is now GPU-accelerated, with automatic CPU fallback

# Your existing pandas code from Phase 2 runs largely unchanged here.
returns = price_df.pct_change().dropna()
exp_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252
```

For the "classic" (non-accelerator) cuDF mode, which gives you more control and a cleaner story for your README ("I used native cuDF, not just the automatic mode"), the equivalent is:

```python
import cudf

price_gdf = cudf.from_pandas(price_df)   # or read Parquet directly with cudf.read_parquet
returns_gdf = price_gdf.pct_change().dropna()
exp_returns_gdf = returns_gdf.mean() * 252
```

**Validation step (do not skip):** for a small universe, compute `exp_returns` and `cov_matrix` both ways and assert they match within floating-point tolerance (`numpy.allclose`). Only after this passes should you trust any speed comparison — a fast wrong answer is worse than a slow right one, and this is exactly the kind of check an NVIDIA interviewer will ask whether you did.

### Phase 4 — cuML Risk Model (Days 9-10)

For a factor-model approach to covariance estimation (more realistic than a raw sample covariance matrix at scale, and a stronger technical story):

```python
from cuml.decomposition import PCA

def factor_covariance_cuml(returns_gdf, n_factors=10):
    pca = PCA(n_components=n_factors)
    factor_returns = pca.fit_transform(returns_gdf)
    # Reconstruct a factor-model covariance estimate:
    # Sigma ≈ B * F * B^T + D  (B = factor loadings, F = factor covariance, D = idiosyncratic variance)
    loadings = pca.components_
    factor_cov = cudf.DataFrame(factor_returns).cov()
    # ... combine into a full covariance matrix (this is the "engineering" part —
    # work through the linear algebra yourself rather than copying a snippet blindly)
    return loadings, factor_cov
```

This step is where you can go deeper technically if you have time: a Ledoit-Wolf shrinkage estimator or a proper Barra-style factor model is a genuinely more sophisticated risk model than a raw sample covariance matrix, and cuML gives you the PCA/regression primitives to build it.

### Phase 5 — cuOpt Solver Formulation (Days 11-15, the technical core)

This is the highest-value section of the whole project. Below is a **reference starting formulation** using cuOpt's confirmed current Python API (`cuopt.linear_programming.problem`). Validate every piece against the official examples first — cuOpt's Python API has moved fast in 2026, so treat this as a pattern to adapt, not a guaranteed drop-in.

**Start here — confirm these two official examples run on your machine before writing your own formulation:**
- `simple_qp_example.py` (minimize `x² + y²` subject to linear constraints)
- `qp_matrix_example.py` (a matrix-form QP using `QuadraticExpression`, structurally identical to a mean-variance problem)

Both are in NVIDIA's docs at `docs.nvidia.com/cuopt/user-guide/latest/cuopt-python/lp-qp-milp/lp-qp-milp-examples.html` (or the renamed "Convex Optimization" section in newer releases) and in the `NVIDIA/cuopt-examples` GitHub repo.

**Your mean-variance formulation, adapted from the matrix QP pattern:**

```python
from cuopt.linear_programming.problem import Problem, MINIMIZE, QuadraticExpression
from cuopt.linear_programming.solver_settings import SolverSettings

def build_mean_variance_problem(cov_matrix: np.ndarray, exp_returns: np.ndarray,
                                  risk_aversion: float = 1.0, max_weight: float = 0.10):
    n = len(exp_returns)
    prob = Problem("Mean-Variance Portfolio")

    # Decision variables: one weight per asset, long-only, capped position size
    weights = [prob.addVariable(lb=0.0, ub=max_weight, name=f"w_{i}") for i in range(n)]

    # Budget constraint: weights sum to 1 (fully invested, no leverage)
    budget_expr = weights[0]
    for w in weights[1:]:
        budget_expr = budget_expr + w
    prob.addConstraint(budget_expr == 1.0, name="budget")

    # Quadratic risk term: w^T * Sigma * w
    quad_risk = QuadraticExpression(cov_matrix.tolist(), weights)

    # Linear expected-return term
    return_expr = exp_returns[0] * weights[0]
    for mu_i, w_i in zip(exp_returns[1:], weights[1:]):
        return_expr = return_expr + mu_i * w_i

    # Minimize risk, penalized by (negative) expected return: standard mean-variance tradeoff
    prob.setObjective(quad_risk - risk_aversion * return_expr, sense=MINIMIZE)

    settings = SolverSettings()
    settings.set_parameter("time_limit", 60)
    return prob, weights, settings


def solve_and_extract(prob, weights, settings):
    prob.solve(settings)
    if prob.Status.name != "Optimal":
        raise RuntimeError(f"Solver did not find optimal solution: {prob.Status.name}")
    return np.array([w.getValue() for w in weights]), prob.SolveTime, prob.ObjValue
```

**Adding discrete/turnover constraints (MIP layer, optional stretch):** `cuOpt`'s MIP solver (as of mid-2026) is explicitly in beta and currently optimized for finding fast, high-quality *feasible* solutions on linear objectives rather than combined mixed-integer-quadratic problems. The practical, honest approach for a student project is a **two-stage formulation**, which is also a legitimate real-world technique:
1. Solve the continuous QP above for target weights.
2. Formulate a follow-on MIP (linear objective — e.g., minimize number of trades or transaction cost) that rounds those target weights to tradeable lot sizes / limits turnover from the current portfolio, using `INTEGER` variables exactly as in NVIDIA's own `production_planning_example.py` pattern (`prob.addVariable(vtype=INTEGER, ...)`).

Document this two-stage design decision explicitly in your README — it shows you understood a real current solver limitation rather than hitting an error and giving up, which is a better interview story than pretending the tool has no limits.

**Validate correctness before trusting speed:** solve a 3-5 asset toy version by hand (or against `cvxpy`'s output on the identical problem) and confirm `cuOpt` matches. Only then scale to your full universe.

### Phase 6 — Backtest Engine (Days 16-18)

A rolling-window backtest: re-optimize on a monthly or quarterly basis using only data available up to that point (no lookahead), hold the resulting portfolio until the next rebalance, and track cumulative P&L, turnover, and drawdown. Run this identically against both the cuOpt-derived weights and the CPU-solver-derived weights so you can report **solution quality parity**, not just speed.

### Phase 7 — Benchmark Suite (Days 19-21)

Structure your benchmark script to isolate each pipeline stage separately — this is what turns "it's faster" into a credible engineering result:

```python
import time

def benchmark_stage(fn, *args, n_runs=5, **kwargs):
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return result, {"mean": np.mean(times), "std": np.std(times), "min": min(times)}
```

Run this for: (a) feature engineering (pandas vs cuDF), (b) covariance/risk model (NumPy/sklearn vs cuML), (c) solve step (SciPy/CVXPY vs cuOpt), at each of your universe sizes (e.g., 50 / 500 / 3,000 tickers). Report a table, not a single headline number — and report variance across runs, since GPU warm-up effects on the first call are real and you should either exclude or explicitly note them.

### Phase 8 — Dashboard (Days 22-24)

A lightweight Streamlit or FastAPI+React dashboard showing: current portfolio weights, backtest equity curve (GPU-derived vs CPU-derived, overlaid), the benchmark timing table, and (if you build it) the natural-language rebalance explanation.

### Phase 9 (Stretch) — NIM-Served Explainer Agent (Days 25-30+)

Serve a Nemotron 3 Nano model via NIM (self-hosted, single GPU) and prompt it with the structured output of a rebalance event (old weights, new weights, the constraints that were binding, the top contributors to risk) to generate a plain-English explanation. Keep this component deliberately small — it should read as "I added an explanation layer on top of a real optimization engine," not "I built a chatbot that happens to mention portfolios." If you build this, log actual latency/cost for the NIM call too, and include it in your benchmark table so the whole project stays consistent in its "measure everything" ethos.

---

## 5. Benchmark Methodology (Full Detail)

- **Universe sizes:** at minimum, three points — small (50 tickers, where GPU overhead may erase any advantage — report this honestly), medium (500), large (3,000+, S&P 500 + Russell 2000 subset or similar).
- **Time period:** 10+ years of daily data, so the feature-engineering step operates on a genuinely large array (this is where cuDF's advantage is real, not a toy dataset).
- **Repetitions:** run each configuration 5 times, report mean ± std, and separately note the first-run time (GPU context/kernel warm-up is a real effect and hiding it would be misleading).
- **Fairness checks:** same machine for CPU and GPU runs (don't compare a laptop CPU to a rented GPU instance's CPU cores), single-threaded CPU baseline unless you also report a multi-threaded comparison explicitly labeled as such, identical random seeds where applicable.
- **What to report:** wall-clock time per pipeline stage (not just total), solution quality (objective value, resulting Sharpe ratio in backtest) at every universe size to prove parity, and the crossover point where GPU acceleration starts winning.

---

## 6. Repository Structure

```
gpu-portfolio-engine/
├── README.md                      # architecture diagram, how to reproduce, results table
├── Dockerfile                     # pinned cuopt/cuDF/cuML versions
├── requirements.txt
├── data/
│   ├── download_universe.py       # yfinance pull + Parquet cache
│   └── validate_data.py           # missing-day / split checks
├── pipeline/
│   ├── cpu_baseline.py            # pandas + SciPy/CVXPY
│   ├── gpu_pipeline.py            # cuDF + cuML
│   └── parity_tests.py            # numerical validation between the two
├── optimizer/
│   ├── mean_variance_cuopt.py     # cuOpt QP formulation
│   ├── turnover_mip_cuopt.py      # optional MIP post-processing layer
│   └── mean_variance_cpu.py       # CVXPY equivalent, for benchmarking
├── backtest/
│   └── engine.py                  # rolling rebalance, P&L, turnover, drawdown
├── benchmarks/
│   ├── run_benchmarks.py
│   └── results/                   # raw timing CSVs + generated charts
├── explainer/                     # stretch: NIM-served Nemotron agent
│   └── nim_explainer.py
├── dashboard/
│   └── app.py
└── docs/
    ├── architecture-diagram.png
    └── case-study.md              # the technical blog post version
```

---

## 7. Testing & Validation Checklist

- [ ] `cuDF` feature outputs numerically match `pandas` outputs (`np.allclose`) on the same small dataset.
- [ ] `cuOpt` QP solution matches `cvxpy` solution (objective value and weights) on a small, tractable universe.
- [ ] `cuOpt` matches a hand-derived closed-form solution on a toy 2-3 asset unconstrained case.
- [ ] Backtest has no lookahead bias (rebalance decisions only use data available as of that date).
- [ ] Benchmark runs are repeated (≥5x) and report variance, not single timings.
- [ ] Data quality checks pass (no unexplained gaps, splits/dividends handled).
- [ ] Every reported number in the README is reproducible by rerunning a script in the repo — no hand-edited results.

---

## 8. Cost Management

- Develop and debug the CPU baseline and data pipeline **entirely without a GPU** — this is most of the code and all of the correctness risk.
- Only spin up a rented GPU (RunPod/Lambda; an L4 or A10 instance is enough for this problem size, no need for H100/A100) for the actual cuDF/cuML/cuOpt runs and the final benchmark sweep.
- Estimate: 10-20 total GPU-hours across the whole project at ~$0.50-1.50/hr for an L4/A10-class instance → roughly **$10-30 total cloud spend**, making this one of the cheapest projects to execute on the full portfolio list.
- Shut down instances between work sessions; don't leave a GPU idling overnight.

---

## 9. Suggested Timeline

| Week | Focus |
|---|---|
| 1 | Data acquisition + CPU baseline (pandas/CVXPY) working end-to-end on a small universe |
| 2 | Port to cuDF/cuML; validate numerical parity against the CPU baseline |
| 3 | cuOpt QP formulation; validate against CVXPY and closed-form cases; scale to full universe sizes |
| 4 | Backtest engine; full benchmark sweep across universe sizes; dashboard |
| 5-6 (stretch) | Turnover MIP layer, NIM-served explainer agent, polish visualizations, write case study, record demo |

**MVP scope (if time-constrained):** CPU baseline + cuDF/cuOpt port + one universe size + a benchmark table. That alone is a legitimate, demonstrable project.
**Full/exceptional scope:** all of the above plus multiple universe sizes, the MIP turnover layer, factor-model risk estimation via cuML, and the NIM explainer agent.

---

## 10. Demo Script (for interviews or a recorded video)

1. Open with the business problem in one sentence: "Quant desks need to re-optimize large portfolios often, but CPU solvers don't scale — I built a GPU-accelerated version and measured exactly where and how much it helps."
2. Show the CPU baseline running on your largest universe — let it visibly take time.
3. Show the GPU pipeline running the identical problem, side by side if possible, with a live timer.
4. Show your parity validation output — "same answer, faster" — this is the credibility moment.
5. Walk through the benchmark table across universe sizes, and be upfront about the crossover point where GPU stops being worth it at small scale.
6. If built, close with the backtest equity curve and (optionally) the natural-language rebalance explanation.

---

## 11. Resume Bullets & Interview Talking Points

**Resume bullets:**
- "Built a GPU-accelerated portfolio optimization engine using NVIDIA cuOpt and RAPIDS cuDF/cuML, measuring a [X]x speedup over a CPU (SciPy/CVXPY) baseline across a [N]-asset universe and 10 years of daily data."
- "Designed and numerically validated QP portfolio formulations in cuOpt against a CVXPY reference implementation before benchmarking at production-scale universe sizes."
- "Extended the pipeline with a two-stage QP+MIP design to handle turnover and lot-size constraints within cuOpt's current solver capabilities."

**Interview talking points to rehearse:**
- Why mean-variance optimization is a QP and how `cuOpt` represents the quadratic term (`QuadraticExpression` over a covariance matrix).
- Where GPU acceleration helps (large dense linear algebra: feature engineering, covariance estimation, large QP solves) and where it doesn't (small problems, where kernel launch/data-transfer overhead can dominate) — and that you *measured* this crossover rather than assuming it.
- How you validated correctness independently of speed (CVXPY parity, closed-form toy cases).
- The two-stage QP+MIP design choice, and why you didn't force a single mixed-integer-quadratic formulation given cuOpt's current beta status for that combination.
- How this connects to NVIDIA's own `cuFOLIO` direction, showing you understand where their roadmap is heading, not just their existing tools.

---

## 12. Key Risks and How to Handle Them

- **Risk: "fast but wrong" perception.** Mitigate with the parity-validation checklist in Section 7 — always show correctness before speed in your demo and README.
- **Risk: cuOpt API changes underneath you mid-project.** Pin your installed version once your code works, and note the pinned version explicitly in `requirements.txt`/`Dockerfile`. If you upgrade later, re-run your test suite before trusting new results.
- **Risk: small-universe GPU overhead makes your speedup story weak at first.** This is fine — report it honestly as the crossover point, which is itself a legitimate finding a real engineer would report.
- **Risk: MIP/turnover layer is genuinely hard given cuOpt's beta MIP status.** Treat it as a stretch goal, not core scope — the QP layer alone is a complete, credible MVP.
- **Risk: data quality issues (splits, delistings, survivorship bias) undermine the backtest.** Use adjusted close prices, and explicitly note survivorship bias as a known limitation in your README rather than ignoring it — acknowledging a limitation you understand is stronger than a hidden flaw.

---

## 13. Sources

- NVIDIA cuOpt Python API and examples (Problem/QuadraticExpression/SolverSettings syntax, QP/MIP/LP patterns): `docs.nvidia.com/cuopt/user-guide/latest/cuopt-python/` (quick-start, LP-QP-MILP API reference and examples pages).
- NVIDIA cuOpt installation/pip commands across recent releases: `docs.nvidia.com/cuopt/user-guide/{25.10.00,26.02.00}/cuopt-python/quick-start.html`.
- NVIDIA cuOpt GitHub: `github.com/NVIDIA/cuopt`, `github.com/NVIDIA/cuopt-examples`.
- NVIDIA's own cuFOLIO reference toolkit: `github.com/NVIDIA-AI-Blueprints/` (cuFOLIO description).
- RAPIDS cuDF pandas Accelerator Mode and install commands: `rapids.ai/cudf-pandas`, `docs.rapids.ai/api/cudf/stable/cudf_pandas/`, `rapids.ai` (install selector).
- Cloud GPU pricing for cost estimates: see the earlier research report's Section 12 sources (Spheron, Thunder Compute, BuildMVPFast GPU pricing comparisons, 2026).

*Note: cuOpt's Python API module organization changed between the 26.02/26.04 and 26.06 releases (LP/QP/MILP was reorganized into "Convex Optimization" and "MIP" sections). Always check the current docs at the URL above before finalizing your code — treat every code snippet in this guide as a verified-as-of-July-2026 pattern to adapt, not a guaranteed final API.*
