"""Docs tables are generated from committed result files, never typed."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from benchmarks.render_tables import (
    StaleBlocks,
    backtest_table,
    replace_block,
    speedup_tables,
    sync_file,
)

DOC = """intro
<!-- BEGIN GENERATED: demo -->
old numbers
<!-- END GENERATED: demo -->
outro
"""


def test_only_the_marked_block_is_replaced():
    out = replace_block(DOC, "demo", "new numbers")
    assert out == DOC.replace("old numbers", "new numbers")


def test_missing_markers_are_an_error_not_a_silent_skip():
    with pytest.raises(KeyError, match="demo"):
        replace_block("no markers here\n", "demo", "x")


def test_check_mode_reports_a_stale_file_without_writing_it(tmp_path):
    doc = tmp_path / "README.md"
    doc.write_text(DOC)
    with pytest.raises(StaleBlocks, match="README.md"):
        sync_file(doc, {"demo": "new numbers"}, check=True)
    assert doc.read_text() == DOC


def _write_backtest(results, name, strategy, benchmark):
    run = results / "backtest" / name
    run.mkdir(parents=True)
    cols = ["annualized_return", "annualized_vol", "sharpe", "max_drawdown", "avg_turnover"]
    pd.DataFrame(
        [strategy, benchmark], columns=cols,
        index=["cpu (pandas + cvxpy)", "equal-weight 1/N"],
    ).to_csv(run / "backtest_summary.csv")


def test_backtest_table_formats_committed_summaries(tmp_path):
    _write_backtest(tmp_path, "real", [0.2356, 0.266, 0.9294, -0.3687, 0.5788],
                    [0.1636, 0.187, 0.9043, -0.3654, 0.099])
    # Values chosen off rounding boundaries so the expected strings are unambiguous.
    _write_backtest(tmp_path, "synthetic", [0.2453, 0.2397, 1.036, -0.2753, 0.5833],
                    [0.1601, 0.1789, 0.92, -0.2187, 0.1742])

    table = backtest_table(tmp_path, [("real", "real"), ("synthetic", "synthetic")])

    assert table.splitlines()[:4] == [
        "| | MV (real) | 1/N (real) | MV (synthetic) | 1/N (synthetic) |",
        "|---|---|---|---|---|",
        "| annualized return | 23.6% | 16.4% | 24.5% | 16.0% |",
        "| annualized vol | 26.6% | 18.7% | 24.0% | 17.9% |",
    ]
    assert "| max drawdown | −36.9% | −36.5% | −27.5% | −21.9% |" in table


def test_speedup_tables_say_so_when_no_gpu_results_exist(tmp_path):
    cpu_only = tmp_path / "mac-cpu"
    cpu_only.mkdir()
    (cpu_only / "environment.json").write_text(json.dumps({"gpu": "none detected"}))
    pd.DataFrame({"stage": ["solve"], "n_assets": [50], "n_days": [252], "cpu": [0.01]}).to_csv(
        cpu_only / "speedup_table.csv", index=False)

    assert "No GPU results" in speedup_tables(tmp_path)


def test_speedup_tables_render_each_gpu_host_with_its_environment(tmp_path):
    host = tmp_path / "rtx-wsl2"
    host.mkdir()
    (host / "environment.json").write_text(json.dumps({"gpu": "NVIDIA GeForce RTX 4090, 24564 MiB, 580.10"}))
    pd.DataFrame({
        "stage": ["solve", "solve"], "n_assets": [50, 3000], "n_days": [2520, 2520],
        "cpu": [0.004, 9.0], "gpu": [0.02, 0.9], "speedup": [0.2, 10.0],
    }).to_csv(host / "speedup_table.csv", index=False)

    text = speedup_tables(tmp_path)

    assert "NVIDIA GeForce RTX 4090" in text
    assert "| solve | 50 | 4.0 ms | 20.0 ms | 0.20× |" in text
    assert "| solve | 3000 | 9.00 s | 900.0 ms | 10.00× |" in text
