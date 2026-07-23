# Case study: building a GPU portfolio optimizer without a GPU

*Working notes. This is the technical-blog-post version of the README —
what was decided, what was measured, and what is still open.*

---

## The premise

Quant desks re-optimize large portfolios frequently, and the two expensive
stages are the same every time: estimating a covariance matrix over a few
thousand assets, and solving a constrained quadratic program over those same
assets. Both are dense linear algebra. Both are, in principle, exactly what a
GPU is for.

"In principle" is the interesting part. The question worth answering is not
*does a GPU help* but *where does it start helping, by how much, and what does
it cost in complexity to get there.* That requires a CPU baseline good enough
that beating it means something.

## Building the baseline honestly

The first real decision was how much care to put into the CPU side. The
tempting version writes the GPU path carefully and the CPU path quickly, and
reports the ratio. That produces a large number and no information.

The concrete instance: Ledoit-Wolf shrinkage. The estimator needs a dispersion
term

```
β² = (1/T²) Σ_t ‖x_t x_tᵀ − S‖²_F / n
```

which the literature and most reference implementations write as a loop over
the T observations, forming an n×n outer product each time. At n = 3,000 assets
and T = 2,520 days that is roughly 2.3 × 10¹⁰ floating-point operations —
minutes on CPU, and a spectacular-looking GPU speedup.

But the loop is unnecessary on either device. Expanding the Frobenius norm:

```
Σ_t ‖x_t x_tᵀ − S‖²_F = Σ_t ‖x_t x_tᵀ‖²_F − 2Σ_t ⟨x_t x_tᵀ, S⟩ + T‖S‖²_F
                      = Σ_t (x_tᵀx_t)² − 2⟨T·S, S⟩ + T‖S‖²_F
                      = Σ_t (x_tᵀx_t)² − T‖S‖²_F
```

which is one reduction over rows and no n×n intermediates at all. Both paths
use this form. `test_ledoit_wolf_matches_naive_loop_implementation` pins the
result to the literal loop version so the optimization cannot drift into a
different estimator.

The speedup this project eventually reports will be smaller because of that
decision. It will also be real.

## Reading the solver API rather than trusting a snippet

The widely-circulated cuOpt portfolio-optimization snippet has three problems
that only show up when you check it against the actual API reference:

1. `prob.Status.name` — `Status` is a plain `int` in the 26.02 release. This
   raises `AttributeError` at the exact moment you are trying to find out
   whether the solve succeeded.
2. Building the budget constraint and the return term by chaining
   `expr = expr + term` over n variables. Each `+` allocates a new expression
   object, so model construction is quadratic in n. At 3,000 assets this
   dominates the GPU solve it exists to feed — you would be benchmarking Python
   object allocation and attributing the result to CUDA.
3. Passing the covariance to `QuadraticExpression` with no statement of what
   the matrix means.

Point 3 is the subtle one. Solvers split roughly evenly on whether a quadratic
matrix denotes `xᵀQx` or `½xᵀQx`. CVXPY's `quad_form` is the former. If cuOpt
were the latter, every cuOpt portfolio in this project would be solved at half
the intended risk aversion — and would still look completely reasonable. Sum to
one, respect the position cap, sensible-looking diversification. Nothing about
the output announces the error.

So the convention is not assumed. `optimizer/cuopt_compat.py` solves

```
minimize  q·x² − c·x    with q = c = 1,  x ∈ [0, 10]
```

whose optimum is x = 0.5 under one convention and x = 1.0 under the other, and
scales the covariance accordingly. Three lines of setup to convert a silent
factor-of-two error into a startup assertion.

## What "the same problem" has to mean

The comparison is only meaningful if both backends solve an identical problem.
Enforcing that by discipline does not survive contact with a refactor, so it is
enforced by types: `RiskModel` and `PortfolioSpec` are the only things that
cross between stages, and the backtest and benchmark both take the risk-model
function and the solver function as arguments. Switching CPU→GPU changes two
callables.

The same reasoning drove using the same *algorithm* on both sides rather than
sklearn's `LedoitWolf` on CPU and cuML's on GPU. Those differ in their shrinkage
target and their handling of the mean, so comparing them would measure two
estimators and credit the difference to hardware.

Float64 on both sides, for the same reason. Float32 would hand cuDF roughly a 2×
memory-bandwidth advantage, but a covariance matrix accumulated in float32 loses
about seven significant digits — enough to push a near-singular matrix indefinite
and change the optimizer's answer. That tradeoff is worth measuring; it is not
worth taking silently, so `dtype` is a parameter and the default matches CPU.

## Where the two-stage design came from

cuOpt's MIP solver is, as of the 26.x releases, explicitly beta and targeted at
finding good feasible solutions to problems with **linear** objectives. Real
portfolios need integrality — you cannot buy 43.7 shares — so the textbook move
is a mixed-integer quadratic program, which is precisely what the tool does not
do well.

Rather than force it, stage 1 solves the continuous QP for ideal weights and
stage 2 solves a linear MIP that rounds them to tradeable lots, minimizing L1
tracking error plus explicit transaction cost. This is a standard production
pattern, not a workaround.

It does have a genuine cost: the rounding minimizes *weight* distance, not
*variance* distance, so it is not risk-aware. Rather than assert that the
difference is small, `report_drift` computes the realized volatility gap per
rebalance, and a greedy largest-remainder rounder is included as the baseline
the MIP has to beat. If the MIP does not beat it, the MIP layer is not earning
its complexity — and that is the finding.

## The test that had to be adversarial

Lookahead bias is the failure mode that does not announce itself. The code runs,
the numbers are plausible, and the equity curve is simply too good. Structural
prevention (`prices.loc[:date]` before the risk model sees anything) is
necessary but not sufficient, because the natural test for it — "assert no risk
model saw data past its rebalance date" — passes trivially if the backtest
never calls the risk model in the first place, or if the assertion is subtly
weak.

So there are two tests. The first records the last date every risk model was
handed and asserts it never exceeds that rebalance's date. The second builds a
deliberately cheating risk model that estimates from the *next* 250 days and
asserts that it produces a higher Sharpe — confirming that foresight, if
present, would in fact show up in the metric being watched. A test that cannot
fail is not a test.

## What the CPU-only results already say

Mean-variance against an equal-weight benchmark, quarterly rebalance, 10 bps
one-way costs, 756-day lookback, Ledoit-Wolf covariance. Two universes: 120 real
S&P 500 names over 2014–2026 (`--source yfinance`), and 150 synthetic assets
over ten years (`--source synthetic`).

| | MV (real) | 1/N (real) | MV (synthetic) | 1/N (synthetic) |
|---|---|---|---|---|
| annualized return | 16.2% | 16.4% | 10.4% | 15.9% |
| annualized vol | 22.8% | 17.3% | 18.0% | 17.9% |
| Sharpe | 0.77 | 0.96 | 0.64 | 0.91 |
| max drawdown | −36.9% | −35.8% | −25.1% | −22.2% |
| avg turnover | 57.9% | 0% | 62.7% | 0% |

The optimizer loses on both, and on real data it loses in the specific way the
literature predicts: it matches 1/N on return while running ~5 points *more*
volatility, despite volatility being the thing it minimizes. That is estimation
error in the inputs propagating straight through an optimizer that treats them
as certain.

(Real-data caveat: these 120 names are current S&P 500 members, so the run
carries survivorship bias and both columns are flattered. The comparison between
them is still fair — both hold the same universe.) This is the expected result and it is not a bug: expected
returns are estimated as historical sample means, which are a famously noisy
forecast, and DeMiguel, Garlappi & Uppal (2009) is a well-known paper
demonstrating exactly this against a naive 1/N benchmark.

It is worth stating plainly because it separates two claims the project is
making. The *engineering* claim is that the GPU pipeline computes the same
answer faster. The *investment* claim would be that the answer is a good one —
and this project does not make that claim. Reporting the loss is what keeps the
first claim credible.

## Open items

- Every GPU number. The code is written and version-shimmed; none of it has run.
  Parity first, then timings — in that order, on the same physical machine.
- The n = 50 crossover point. Expected to favor CPU; worth knowing precisely
  where it flips.
- Whether the dense n² covariance hand-off to cuOpt's Python layer becomes the
  binding constraint at n = 3,000. If model construction dominates solve time,
  the interesting engineering moves to the MPS path or a sparse formulation.
- Whether the MIP rounding layer beats greedy largest-remainder at all.
