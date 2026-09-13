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

## Reading the solver source, not just its reference

The failure mode worth designing against in solver code is the silent one: a
model that is wrong in a way that still produces a plausible portfolio. Three
places where a naive cuOpt formulation goes wrong that way — one of which this
project itself got wrong, and found only by reading cuOpt's source.

**The return term has to travel inside the objective.** The first version of
`optimizer/mean_variance_cuopt.py` set each variable's linear coefficient at
creation — `addVariable(obj=-λμᵢ)` — to avoid building a long expression, then
passed only the quadratic risk term to `setObjective`. In cuOpt
(`linear_programming/problem.py`, v26.02 through v26.08), `setObjective` first
zeroes every variable's linear coefficient and then applies the expression it
is given. The return term was discarded, so every cuOpt solve in this project
was a minimum-variance solve. The output would have summed to one, respected
the cap and looked diversified; nothing about it announces the error. The
design did guard against it — the GPU parity test compares objective values at
λ = 1, where this bug is unmissable — but that test needs a GPU and had never
run. The fix puts the return term in the objective expression, and
`tests/test_cuopt_formulation.py` now checks the assembled model on any machine
against a stand-in that reproduces cuOpt's model-building semantics.

The same reading turned up a second trap: a matrix-form `QuadraticExpression`
is added *positionally* onto a matrix over all of the problem's variables, so
once turnover auxiliaries exist the n×n covariance has to be zero-padded to
cover them — otherwise the solve fails on a shape mismatch.

**Linear terms as one expression, not a chain.** Building the budget constraint
or the return term as `expr = expr + term` over n variables allocates a new
expression per step, so model construction is quadratic in n. At 3,000 assets
that would dominate the GPU solve it exists to feed — benchmarking Python
object allocation and attributing it to CUDA. Each linear term is instead one
`LinearExpression` holding all n coefficients, and the covariance goes in as a
single sparse matrix rather than 9M nested Python floats.

**The quadratic convention is asserted, not assumed.** Solvers split on whether
a quadratic matrix denotes `xᵀQx` or `½xᵀQx`; getting it wrong silently halves
or doubles the effective risk aversion. NVIDIA's documentation states that cuOpt
takes Q "without the 1/2 factor", i.e. `xᵀQx`, matching CVXPY's `quad_form`.
`optimizer/cuopt_compat.py` still checks it at startup by solving

```
minimize  q·x² − c·x    with q = c = 1,  x ∈ [0, 10]
```

whose optimum is x = 0.5 under one convention and x = 1.0 under the other: one
tiny solve turns a silent factor-of-two error into a startup assertion.

*Correction:* an earlier version of this section also claimed that
`prob.Status.name` raises `AttributeError` because `Status` is a plain `int`.
That is wrong. After a solve, `Status` holds cuOpt's termination status, an
`IntEnum` with a `.name`; only the placeholder before a solve is a bare `-1`.

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

The CPU solver is pinned for the same reason. Left alone, CVXPY picked OSQP for
this QP. Polished OSQP turned out to be just as accurate here (objectives within
about 1e-9 of Clarabel's), but Clarabel was about three times faster at n = 500,
and it is an interior-point method like cuOpt's barrier QP solver. Pinning it
makes the CPU baseline the stronger of the two options, and it stops the
reference from changing whenever CVXPY changes its default.

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

That was the finding. Run on an RTX 4060, the MIP as designed — fully
invested — lost to greedy in 17 of 18 cases, and the fault was the budget
rule: greedy may keep cash and the MIP may not, and a sum of integer lots at
market prices meets 1 only to within a solver's feasibility tolerance, which
then picks the answer. Under greedy's own rule the MIP wins, but at coarse lot
sizes it does so mostly by holding cash.

The default budget is now a cash band, invested between 99.5% and 100%. That
problem is well-posed, and greedy's answer is feasible for it whenever greedy
keeps 0.5% cash or less, so the comparison is fair. Under the band the MIP beat
greedy in all 18 cases, but by under 1% in 13 of them. cuOpt also needed its
full 60 s in 7 cases, where CPU HiGHS proved 5 of those 7 optimal in seconds.
The honest summary is that the MIP layer earns a small, consistent edge, and
that a GPU does not pay off at this size. The README's lot-rounding table and
item 4 of "What is actually engineered here" have the numbers.

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

## What the CPU-only results say

Mean-variance against equal-weight 1/N, both run through the same backtest
engine: quarterly rebalance at the close, 10 bps one-way costs, 756-day
lookback, Ledoit-Wolf covariance, risk aversion 2, position cap 8/n. Both are
scored from the first rebalance, over identical dates. Two universes:

- **Real:** 120 current S&P 500 names, from a pinned yfinance snapshot
  (2014-01-02 to 2026-07-22, scored from 2017-03-31; the file's SHA-256 is in
  `benchmarks/results/backtest/sp500-120-snapshot/backtest_config.json`).
- **Synthetic:** 150 assets from the k-factor generator (2014-01-01 to
  2023-08-29, scored from 2016-12-30).

<!-- BEGIN GENERATED: backtest-summary -->
| | MV (real) | 1/N (real) | MV (synthetic) | 1/N (synthetic) |
|---|---|---|---|---|
| annualized return | 23.6% | 16.4% | 24.5% | 16.0% |
| annualized vol | 26.6% | 18.7% | 23.9% | 17.9% |
| Sharpe | 0.93 | 0.90 | 1.04 | 0.92 |
| max drawdown | −36.9% | −36.5% | −27.5% | −21.8% |
| avg turnover per rebalance | 57.9% | 9.9% | 58.3% | 17.4% |
<!-- END GENERATED: backtest-summary -->

(Generated from the committed `backtest_summary.csv` files by
`python -m benchmarks.render_tables`.)

On real data the optimizer earns about 7 points more return for about 8 points
more volatility, and ends up level with 1/N on Sharpe. The 0.03 gap is noise:
over 9.3 scored years the standard error of a Sharpe estimate is roughly 0.4.
It gets there with six times the turnover. With risk aversion 2 applied to
annualized means, the return term dominates this objective, so this is closer to
return-chasing than to variance minimization. Its inputs are historical sample
means, a famously noisy forecast, and failing to beat 1/N reliably out of sample
is what DeMiguel, Garlappi & Uppal (2009) found for mean-variance generally.

Two caveats. The 120 names are *current* S&P 500 members, so survivorship bias
flatters both real columns, though the comparison between them is fair because
both hold the same universe. And the synthetic win should not be read as
evidence of anything: the generator gives each asset a constant drift, so
historical means are informative there by construction. Synthetic data exists
to reach universe sizes free data will not serve, not to judge the strategy.

It is worth stating plainly because it separates two claims the project is
making. The *engineering* claim is that the GPU pipeline computes the same
answer faster. The *investment* claim would be that the answer is a good one —
and this project does not make that claim.

### Erratum (2026-09-12)

An earlier version of this section reported that the optimizer lost to 1/N on
both universes. Two backtest bugs produced that comparison:

1. **Different scoring windows.** The strategy's statistics included its
   756-day lookback, about a quarter of the sample, spent holding nothing and
   earning exactly zero. That deflated its annualized return and Sharpe. The
   benchmark was a costless daily-rebalanced 1/N scored from the first price
   row: a different period under different trading assumptions.
2. **Missing rebalance-day returns.** On each rebalance day the engine swapped
   in the new holdings before booking that day's return, so neither the old nor
   the new portfolio earned it.

On the same real-data snapshot with the same settings, the old engine
reproduces the previously published figures exactly: MV 16.2% annualized,
Sharpe 0.77, against 1/N 16.4%, Sharpe 0.96. The corrected engine gives the
table above.

The previously published synthetic column (MV Sharpe 0.64 against 0.91) could
not be reproduced from the committed code and configuration, so it is withdrawn
rather than explained.

The fixes are structural, so the bug cannot quietly come back:

- Results are scored from the first rebalance.
- The rebalance-day return accrues to the pre-rebalance holdings, and costs are
  charged on post-return value.
- 1/N runs through the same engine on the same schedule, costs and universe.
- `compare_results` refuses to put results scored over different dates side by
  side.
- Hand-derived tests in `tests/test_backtest.py` pin each case.

## Open items

- A second GPU. Every GPU component has now run, parity first, on one RTX
  4060 Laptop GPU under WSL2 (see the README). A data-center GPU on native
  Linux is the obvious next data point, and it is also where a local NIM
  could be measured: no LLM NIM is validated on an 8 GB card, so the
  explainer has run only against NVIDIA's hosted API.
- Whether the explainer earns its place. On the hosted run, two of three
  replies summed risk shares they were told only to quote, and got the sum
  wrong. The template fallback has no such failure mode. The explainer would
  need a model that keeps to the facts before it is worth more than the
  template.
- The n = 50 crossover point. Expected to favor CPU; worth knowing precisely
  where it flips.
- Whether the dense n² covariance hand-off to cuOpt's Python layer becomes the
  binding constraint at n = 3,000. If model construction dominates solve time,
  the interesting engineering moves to the MPS path or a sparse formulation.
- A turnover cap enforced in stage 2 itself, which rounding currently
  overshoots by up to its rounding error.
