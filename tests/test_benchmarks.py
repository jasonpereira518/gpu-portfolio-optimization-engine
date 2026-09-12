"""Benchmark sweep contract: which stages are timed, and what each row records."""

from __future__ import annotations

import numpy as np

from benchmarks.run_benchmarks import run_sweep


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
