"""Does cuOpt's MIP call a solution Optimal after its own postsolve found it infeasible?

    python -m benchmarks.cuopt_postsolve_check --out benchmarks/results/<gpu>-postsolve/<cuopt version>

cuOpt checks the post-solved solution against the original problem (in
cpp/src/mip_heuristics/solve.cu, 26.08 through main as of 2026-11). When the
check fails it logs "The solution is not feasible after post solve", then
reports Optimal or FeasibleFound anyway: that status comes from the MIP gap
alone.

This sweep re-solves the lot-rounding MIP over the lot-rounding grid. It runs
each budget rule with presolve on and off, optionally across seeds in
deterministic mode. For every solve it records:

- the status, and whether that warning was logged
- cuOpt's own violation stats
- the same violations recomputed here from the raw, unrounded values
- the reported objective next to the one those values imply

A flagged solve keeps its MPS file and solver log under ``flagged/``, so the
smallest can be minimized and sent upstream.

It also re-checks one model-building behavior. ``setObjective`` discards
coefficients passed as ``addVariable(obj=...)``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.harness import environment_metadata
from benchmarks.run_lot_rounding import MIP_BUDGETS, stage1_targets
from optimizer.cuopt_compat import CuOptApi, load_cuopt, status_name
from optimizer.turnover_mip_cuopt import solve_lot_rounding_cuopt

WARNING = "not feasible after post solve"
# cuOpt's default mip_absolute_tolerance; a violation past it is a real one.
TOL = 1e-6
# cpp/include/cuopt/mathematical_optimization/constants.h, same in 26.08 and 26.10.
PRESOLVE_NAMES = {-1: "default", 0: "off", 1: "papilo", 2: "pslp"}


def _recording(api: CuOptApi, params: dict, workdir: Path, solves: list) -> CuOptApi:
    """``api`` whose Problem.solve applies ``params``, writes the model as MPS,
    logs to its own file and appends itself to ``solves``."""

    class Recorded(api.Problem):
        def solve(self, settings=None):
            settings = settings or api.SolverSettings()
            stem = workdir / f"solve{len(solves)}"
            self.mps_path, self.log_path = stem.with_suffix(".mps"), stem.with_suffix(".log")
            for name, value in {**params, "log_file": str(self.log_path)}.items():
                settings.set_parameter(name, value)
            if hasattr(self, "writeMPS"):  # the CPU stand-in has none
                self.writeMPS(str(self.mps_path))
            solves.append(self)
            super().solve(settings)

    return dataclasses.replace(api, Problem=Recorded)


def _measure(prob, target, lot_weight, max_cash) -> dict:
    """Violations and objectives from the raw values, independent of the solver.

    The sweep's model has no trading terms, so its variables are the n lot
    counts then the n deviations, and its rows are the two deviation bounds
    per asset plus the budget band: everything is checked here.
    """
    n = len(target)
    values = np.array([v.getValue() for v in prob.getVariables()], dtype=np.float64)
    lots, dev = values[:n], values[n:2 * n]
    invested = float(lot_weight @ lots)
    stats = getattr(prob, "SolutionStats", None)
    log = prob.log_path.read_text(errors="replace") if prob.log_path.exists() else ""
    return {
        "warned": WARNING in log,
        "cuopt_max_constraint_violation": getattr(stats, "max_constraint_violation", np.nan),
        "cuopt_max_int_violation": getattr(stats, "max_int_violation", np.nan),
        "cuopt_max_bound_violation": getattr(stats, "max_variable_bound_violation", np.nan),
        "invested": invested,
        "budget_violation": max(invested - 1.0, (1.0 - max_cash) - invested, 0.0),
        "dev_violation": float(np.max(np.abs(lot_weight * lots - target) - dev, initial=0.0)),
        "int_violation": float(np.max(np.abs(lots - np.round(lots)), initial=0.0)),
        "bound_violation": float(max(-values.min(), 0.0)),
        "obj_reported": float(prob.ObjValue),
        "obj_raw": float(dev.sum()),  # what the solver minimized, at its own values
        # The tracking error of the portfolio actually traded (integer lots).
        "obj_true": float(np.abs(lot_weight * np.round(lots) - target).sum()),
    }


def check_post_solve(
    sizes: list[int],
    lot_sizes: list[int],
    turnover_budgets: list[float | None],
    presolve: list[int],
    seeds: list[int] | None = None,
    portfolio_value: float = 1e6,
    time_limit: float = 60.0,
    flagged_dir: Path | None = None,
    api: CuOptApi | None = None,
) -> pd.DataFrame:
    """One row per (case, budget rule, presolve setting, seed).

    ``seeds=None`` runs once in cuOpt's default opportunistic mode, as the
    project does; a list runs each seed in deterministic mode.
    """
    api = api or load_cuopt()
    runs = [(None, {})] if seeds is None else [
        (s, {"mip_determinism_mode": 1, "random_seed": s}) for s in seeds]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for n, budget, target, last, _, _ in stage1_targets(sizes, turnover_budgets):
            for lot_size in lot_sizes:
                lot_weight = last * lot_size / portfolio_value
                for method, max_cash in MIP_BUDGETS.items():
                    for pre in presolve:
                        for seed, params in runs:
                            solves: list = []
                            recorded = _recording(api, {"presolve": pre, **params}, Path(tmp), solves)
                            case = {"n_assets": n, "turnover_budget": budget, "lot_size": lot_size,
                                    "method": method, "max_cash": max_cash,
                                    "presolve": PRESOLVE_NAMES.get(pre, pre), "seed": seed}
                            try:
                                solve_lot_rounding_cuopt(target, last, portfolio_value, lot_size=lot_size,
                                                         max_cash=max_cash, time_limit=time_limit,
                                                         api=recorded)
                            except RuntimeError:  # no incumbent; the status still says why
                                pass
                            prob = solves[-1]
                            row = {**case, "status": status_name(prob), "solve_s": float(prob.SolveTime),
                                   **_measure(prob, target, lot_weight, max_cash)}
                            ours = max(row["budget_violation"], row["dev_violation"],
                                       row["int_violation"], row["bound_violation"])
                            row["flagged"] = bool(row["warned"] or ours > TOL
                                                  or row["cuopt_max_constraint_violation"] > TOL)
                            if row["flagged"] and flagged_dir is not None:
                                row["artifact"] = _keep(prob, case, flagged_dir)
                            rows.append(row)
    return pd.DataFrame(rows)


def _keep(prob, case: dict, flagged_dir: Path) -> str:
    """Copy a flagged solve's MPS and log next to the results; return the stem."""
    flagged_dir.mkdir(parents=True, exist_ok=True)
    stem = "n{n_assets}_t{turnover_budget}_lot{lot_size}_{method}_pre-{presolve}_seed{seed}".format(**case)
    for src, suffix in ((prob.mps_path, ".mps"), (prob.log_path, ".log")):
        if src.exists():
            shutil.copy(src, flagged_dir / f"{stem}{suffix}")
    return stem


def objective_wipe_check(api: CuOptApi | None = None) -> dict:
    """Does setObjective discard a coefficient given as addVariable(obj=...)?

    minimize -x + y over x, y in [0, 1] is -1. With -x's coefficient set by
    ``obj=`` and only y passed to setObjective, cuOpt solves minimize y,
    whose optimum is 0.
    """
    api = api or load_cuopt()
    prob = api.Problem("objective-wipe")
    x = prob.addVariable(lb=0.0, ub=1.0, obj=-1.0, name="x")
    y = prob.addVariable(lb=0.0, ub=1.0, name="y")
    prob.addConstraint(x + y <= 2.0, name="non_binding")  # cuOpt needs at least one row
    prob.setObjective(api.LinearExpression([y], [1.0], 0.0), sense=api.MINIMIZE)
    prob.solve(api.SolverSettings())
    return {"objective": float(prob.ObjValue), "obj_coefficient_kept": abs(prob.ObjValue + 1.0) < 1e-6}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[50, 200, 500])
    parser.add_argument("--lots", type=int, nargs="+", default=[1, 10, 100])
    parser.add_argument("--turnover", type=float, nargs="+", default=[0.25],
                        help="turnover caps for stage 1; the uncapped case always runs too")
    parser.add_argument("--presolve", type=int, nargs="+", default=[-1, 0],
                        help="cuOpt presolve settings: -1 default (Papilo for MIP), 0 off, 1 Papilo, 2 PSLP")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="run each seed in deterministic mode (default: one opportunistic run)")
    parser.add_argument("--time-limit", type=float, default=60.0, help="per MIP solve, seconds")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    meta = environment_metadata()
    wipe = objective_wipe_check()
    rows = check_post_solve(args.sizes, args.lots, [None, *args.turnover], args.presolve, args.seeds,
                            time_limit=args.time_limit, flagged_dir=args.out / "flagged")

    rows["false_status"] = rows["status"].isin(["Optimal", "FeasibleFound"]) & rows["warned"]
    args.out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out / "postsolve.csv", index=False)
    (args.out / "environment.json").write_text(json.dumps({**meta, "objective_wipe": wipe}, indent=2))

    summary = rows.groupby(["method", "presolve"]).agg(
        solves=("status", "size"), warned=("warned", "sum"), false_status=("false_status", "sum"),
        max_cuopt_violation=("cuopt_max_constraint_violation", "max"),
        max_budget_violation=("budget_violation", "max"),
        max_obj_understated=("obj_true", lambda s: (s - rows.loc[s.index, "obj_reported"]).max()))
    print(summary.to_string(float_format=lambda v: f"{v:.3g}"))
    print(f"\ncuOpt {load_cuopt().version}; setObjective kept addVariable(obj=): "
          f"{wipe['obj_coefficient_kept']} (objective {wipe['objective']:.6g}; -1 if kept, 0 if wiped)")
    print(f"{int(rows['false_status'].sum())} of {len(rows)} solves reported a solution "
          f"cuOpt's own postsolve had found infeasible; {int(rows['flagged'].sum())} flagged in all")
    print(f"wrote {args.out}/postsolve.csv, environment.json, flagged/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
