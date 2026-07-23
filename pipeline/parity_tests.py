"""Numerical parity between the CPU and GPU paths.

Run this before trusting any timing number. A fast wrong answer is worse than
a slow right one, and on a CPU-only machine this module reports SKIPPED rather
than silently passing.

    python -m pipeline.parity_tests --n 200 --days 2000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from optimizer.mean_variance_cpu import solve_mean_variance_cpu
from optimizer.spec import PortfolioSpec, objective_value
from pipeline.cpu_baseline import build_risk_model
from pipeline.risk_model import RiskModel


@dataclass
class ParityResult:
    name: str
    max_abs_diff: float
    max_rel_diff: float
    tolerance: float
    passed: bool
    note: str = ""

    def render(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        extra = f"  ({self.note})" if self.note else ""
        return (
            f"[{mark}] {self.name:<34} max_abs={self.max_abs_diff:.3e} "
            f"max_rel={self.max_rel_diff:.3e} tol={self.tolerance:.1e}{extra}"
        )


def _compare(name: str, a: np.ndarray, b: np.ndarray, tol: float, note: str = "") -> ParityResult:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        return ParityResult(name, np.inf, np.inf, tol, False, f"shape {a.shape} vs {b.shape}")
    abs_diff = np.abs(a - b)
    scale = np.maximum(np.abs(a), np.abs(b))
    # Relative difference is undefined where both values are ~0; those entries
    # are covered by the absolute check instead of producing spurious 0/0.
    mask = scale > 1e-12
    rel_diff = np.zeros_like(abs_diff)
    rel_diff[mask] = abs_diff[mask] / scale[mask]
    return ParityResult(
        name, float(abs_diff.max()), float(rel_diff.max()), tol, bool(abs_diff.max() <= tol), note
    )


def compare_risk_models(cpu: RiskModel, gpu: RiskModel, tol: float = 1e-9) -> list[ParityResult]:
    results = [
        _compare("exp_returns", cpu.exp_returns, gpu.exp_returns, tol),
        _compare("covariance", cpu.cov, gpu.cov, tol),
    ]
    if cpu.tickers != gpu.tickers:
        results.append(
            ParityResult("ticker ordering", np.inf, np.inf, 0.0, False, "column order differs")
        )
    # Eigenvalue agreement is the check that matters for the optimizer: two
    # covariance matrices can agree entrywise and still condition differently.
    ev_cpu = np.linalg.eigvalsh(cpu.symmetrized())
    ev_gpu = np.linalg.eigvalsh(gpu.symmetrized())
    results.append(_compare("eigenvalues", ev_cpu, ev_gpu, tol * 10))
    return results


def compare_solutions(
    cpu_sol, gpu_sol, cov: np.ndarray, mu: np.ndarray, spec: PortfolioSpec,
    weight_tol: float = 1e-4, objective_tol: float = 1e-8,
) -> list[ParityResult]:
    """Compare two solvers' answers to the same QP.

    Weights get a looser tolerance than the objective on purpose. Mean-variance
    problems with many near-substitutable assets have a flat optimum: two
    solvers can land on visibly different weight vectors whose objective values
    agree to 10 digits. The objective is the invariant; per-name weights are
    not, and asserting tight weight equality would produce false failures that
    train you to ignore the test.
    """
    obj_cpu = objective_value(cpu_sol.weights, cov, mu, spec.risk_aversion)
    obj_gpu = objective_value(gpu_sol.weights, cov, mu, spec.risk_aversion)

    return [
        _compare(
            "objective (common evaluation)",
            np.array([obj_cpu]), np.array([obj_gpu]), objective_tol,
            note="the invariant — must match tightly",
        ),
        _compare(
            "weights", cpu_sol.weights, gpu_sol.weights, weight_tol,
            note="loose by design: flat optimum",
        ),
        _compare(
            "portfolio vol",
            np.array([np.sqrt(cpu_sol.weights @ cov @ cpu_sol.weights)]),
            np.array([np.sqrt(gpu_sol.weights @ cov @ gpu_sol.weights)]),
            1e-6,
        ),
    ]


def run_parity(n_assets: int = 200, n_days: int = 2000, estimator: str = "ledoit_wolf",
               seed: int = 7) -> tuple[list[ParityResult], bool]:
    """Full parity sweep. Skips GPU comparisons cleanly on a CPU-only host."""
    from data.universe import synthetic_prices

    prices = synthetic_prices(n_assets, n_days=n_days, seed=seed).prices
    spec = PortfolioSpec(risk_aversion=1.0, max_weight=0.10)

    cpu_model = build_risk_model(prices, estimator=estimator)
    cpu_sol = solve_mean_variance_cpu(cpu_model, spec)

    results: list[ParityResult] = []
    skipped = False

    from pipeline.gpu_pipeline import rapids_available

    if rapids_available():
        from pipeline.gpu_pipeline import build_risk_model_gpu

        gpu_model = build_risk_model_gpu(prices, estimator=estimator)
        results += compare_risk_models(cpu_model, gpu_model)
    else:
        skipped = True
        print("SKIP: RAPIDS not available — cuDF/cuML parity not checked on this host.")

    from optimizer.cuopt_compat import cuopt_available

    if cuopt_available():
        from optimizer.mean_variance_cuopt import solve_mean_variance_cuopt

        gpu_sol = solve_mean_variance_cuopt(cpu_model, spec)
        cov = cpu_model.nearest_psd()
        results += compare_solutions(cpu_sol, gpu_sol, cov, cpu_model.exp_returns, spec)
        violations = gpu_sol.check(spec)
        results.append(
            ParityResult(
                "cuOpt constraint feasibility", 0.0, 0.0, 0.0, not violations,
                note="; ".join(violations) if violations else "all constraints satisfied",
            )
        )
    else:
        skipped = True
        print("SKIP: cuOpt not available — QP solver parity not checked on this host.")

    # Solver-independent ground truth, which runs everywhere and would catch a
    # bug that CVXPY and cuOpt happened to share.
    from optimizer.mean_variance_cpu import min_variance_closed_form

    cov = cpu_model.nearest_psd()
    closed_form = min_variance_closed_form(cov)
    loose_spec = PortfolioSpec(risk_aversion=0.0, max_weight=1.0, min_weight=-1.0)
    unconstrained = solve_mean_variance_cpu(cpu_model, loose_spec)
    results.append(
        _compare(
            "min-variance vs closed form", closed_form, unconstrained.weights, 1e-4,
            note="no solver involved on the reference side",
        )
    )

    return results, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--days", type=int, default=2000)
    parser.add_argument("--estimator", default="ledoit_wolf")
    args = parser.parse_args()

    results, skipped = run_parity(args.n, args.days, args.estimator)
    print(f"\nParity: {args.n} assets x {args.days} days, estimator={args.estimator}\n")
    for r in results:
        print(r.render())

    failures = [r for r in results if not r.passed]
    print(f"\n{len(results) - len(failures)}/{len(results)} checks passed"
          + ("  (some comparisons SKIPPED — CPU-only host)" if skipped else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
