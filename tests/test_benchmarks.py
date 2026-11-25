"""Benchmark sweep contract: which stages are timed, and what each row records."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.cuopt_postsolve_check import _measure, check_post_solve, objective_wipe_check
from benchmarks.run_benchmarks import run_sweep
from benchmarks.run_lot_rounding import run_lot_rounding
from tests.fake_cuopt import FAKE_API


def test_cpu_sweep_times_psd_repair_as_its_own_stage():
    """Covariance repair is CPU linear algebra, not solver work.

    Timed inside "solve" it would dilute any GPU solver speedup with an O(n^3)
    CPU eigendecomposition on both sides; timed as its own stage it is visible.
    """
    raw, _ = run_sweep([30], n_days=300, estimator="sample", n_runs=2, backends=("cpu",))
    cpu = raw[raw["backend"] == "cpu"]

    assert list(cpu["stage"]) == ["features", "risk_model", "psd_repair", "solve"]


def test_solve_rows_record_the_objective_reached():
    """Speed is only comparable at matched quality, so each solve row carries its objective."""
    raw, _ = run_sweep([30], n_days=300, estimator="ledoit_wolf", n_runs=2, backends=("cpu",))
    solves = raw[raw["stage"] == "solve"]

    assert len(solves) == 1
    assert np.isfinite(solves["extra_objective"]).all()


def test_lot_rounding_sweep_puts_each_budget_rule_next_to_greedy():
    """Every case gets greedy and both MIP budget rules, each row labeled with
    the rule it actually ran under — the comparison is meaningless otherwise."""
    # A fully-invested solve can take minutes to prove optimal; its incumbent
    # at the time limit is still fully invested, which is all this checks.
    rows = run_lot_rounding([20], lot_sizes=[1], turnover_budgets=[None, 0.25], time_limit=3.0,
                            api=FAKE_API)

    cases = rows.groupby(["turnover_budget", "lot_size"], dropna=False)
    assert len(cases) == 2
    for _, case in cases:
        by = case.set_index("method")
        assert set(by.index) == {"greedy", "mip_fully_invested", "mip_cash_band", "mip_cash_allowed"}
        assert abs(by.loc["mip_fully_invested", "cash"]) < 1e-5
        assert by.loc["mip_cash_band", "cash"] <= by.loc["mip_cash_band", "max_cash"] + 1e-9
        assert by.loc["mip_cash_allowed", "tracking_error"] <= by.loc["greedy", "tracking_error"] + 1e-9


def test_postsolve_check_catches_what_an_optimal_status_hides(tmp_path):
    """cuOpt can log a failed postsolve and still say Optimal, so the warning in
    the log and violations recomputed from the raw values each count on their own."""
    log = tmp_path / "solve.log"
    log.write_text("Post-solve status: ...\nThe solution is not feasible after post solve\n")
    lots = [2.0, 2.00001]  # 1e-5 off integral; invests 1.0000025 against an exact budget of 1
    prob = SimpleNamespace(
        getVariables=lambda: [SimpleNamespace(getValue=lambda v=v: v) for v in [*lots, 0.0, 0.0]],
        log_path=log, ObjValue=0.0,
    )

    m = _measure(prob, target=np.array([0.5, 0.5]), lot_weight=np.array([0.25, 0.25]), max_cash=0.0)

    assert m["warned"]
    assert m["budget_violation"] == pytest.approx(2.5e-6)
    assert m["dev_violation"] == pytest.approx(2.5e-6)
    assert m["int_violation"] == pytest.approx(1e-5)
    assert m["obj_true"] == pytest.approx(0.0)  # the integer lots it rounds to track exactly


def test_postsolve_check_passes_a_proven_optimum_and_sees_the_objective_wipe():
    """End to end on the HiGHS stand-in: a proven optimum is never flagged, and the
    setObjective check reports the wipe the stand-in reproduces from cuOpt."""
    rows = check_post_solve([20], [1], [None], presolve=[-1], time_limit=3.0, api=FAKE_API)

    assert set(rows["method"]) == {"mip_fully_invested", "mip_cash_band", "mip_cash_allowed"}
    assert rows["status"].isin(["Optimal", "FeasibleFound"]).all()
    assert not rows["flagged"].any()
    assert objective_wipe_check(FAKE_API) == {"objective": 0.0, "obj_coefficient_kept": False}


def test_lot_rounding_sweep_without_cuopt_reports_greedy_only(monkeypatch):
    """Off-GPU there is no MIP to compare, so no MIP rows — never a CPU solver
    quietly standing in for cuOpt."""
    import benchmarks.run_lot_rounding as sweep

    monkeypatch.setattr(sweep, "cuopt_available", lambda: False)
    rows = sweep.run_lot_rounding([30], lot_sizes=[1], turnover_budgets=[None])

    assert set(rows["method"]) == {"greedy"}
