"""Render the docs' result tables from committed result files.

    python -m benchmarks.render_tables           # rewrite generated blocks in place
    python -m benchmarks.render_tables --check   # exit 1 if any block is stale

Every number in a generated block is read from a CSV committed next to the
environment or config file that produced it; none is typed by hand. Blocks
are delimited by

    <!-- BEGIN GENERATED: <name> -->
    <!-- END GENERATED: <name> -->

and everything between a pair of markers is replaced. ``--check`` is the
guard for CI: a results change without a regenerated doc fails the build.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results"
DOCS = [ROOT / "README.md", ROOT / "docs" / "case-study.md"]

# (benchmarks/results/backtest/<dir>, column label), in table order.
BACKTESTS = [("sp500-120-snapshot", "real"), ("synthetic-150", "synthetic")]
STRATEGY, BENCHMARK = "cpu (pandas + cvxpy)", "equal-weight 1/N"
BACKTEST_ROWS = [
    ("annualized return", "annualized_return", "pct"),
    ("annualized vol", "annualized_vol", "pct"),
    ("Sharpe", "sharpe", "ratio"),
    ("max drawdown", "max_drawdown", "pct"),
    ("avg turnover per rebalance", "avg_turnover", "pct"),
]
STAGE_ORDER = ["h2d_transfer", "features", "risk_model", "psd_repair", "solve", "solve_cvxpy"]


class StaleBlocks(RuntimeError):
    """A doc's generated blocks do not match the committed results."""


def _markers(name: str) -> tuple[str, str]:
    return f"<!-- BEGIN GENERATED: {name} -->", f"<!-- END GENERATED: {name} -->"


def replace_block(text: str, name: str, content: str) -> str:
    """Replace everything between ``name``'s markers with ``content``."""
    begin, end = _markers(name)
    pattern = re.compile(re.escape(begin) + r"\n.*?" + re.escape(end), re.S)
    if not pattern.search(text):
        raise KeyError(f"no generated block named {name!r}")
    return pattern.sub(lambda _: f"{begin}\n{content.rstrip()}\n{end}", text)


def sync_file(path: Path, blocks: dict[str, str], check: bool = False) -> bool:
    """Regenerate the blocks ``path`` contains. Returns True if it changed.

    In ``check`` mode nothing is written; a stale file raises ``StaleBlocks``.
    """
    old = path.read_text()
    new = old
    for name, content in blocks.items():
        if _markers(name)[0] in new:
            new = replace_block(new, name, content)
    if new == old:
        return False
    if check:
        raise StaleBlocks(f"{path.name} has stale generated blocks; run python -m benchmarks.render_tables")
    path.write_text(new)
    return True


def _fmt(value: float, kind: str) -> str:
    if pd.isna(value):
        return "—"
    if kind == "pct":
        return f"{value:.1%}".replace("-", "−")
    return f"{value:.2f}"


def backtest_table(results: Path, runs: list[tuple[str, str]]) -> str:
    """Strategy vs 1/N for each backtest run, one column pair per run."""
    summaries = {label: pd.read_csv(results / "backtest" / name / "backtest_summary.csv", index_col=0)
                 for name, label in runs}
    header = "| | " + " | ".join(f"MV ({label}) | 1/N ({label})" for _, label in runs) + " |"
    lines = [header, "|" + "---|" * (1 + 2 * len(runs))]
    for title, key, kind in BACKTEST_ROWS:
        cells = []
        for _, label in runs:
            table = summaries[label]
            cells += [_fmt(table.loc[STRATEGY, key], kind), _fmt(table.loc[BENCHMARK, key], kind)]
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _seconds(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:.2f} s" if value >= 1.0 else f"{value * 1e3:.1f} ms"


def _estimators(host: Path) -> list[str]:
    """Covariance estimator(s) a sweep ran with, as recorded in its raw timings."""
    raw_csv = host / "timings_raw.csv"
    if not raw_csv.exists():
        return []
    raw = pd.read_csv(raw_csv)
    if "extra_estimator" not in raw:
        return []
    return sorted(raw["extra_estimator"].dropna().unique())


def speedup_tables(results: Path) -> str:
    """One table per GPU sweep whose results are committed."""
    sections = []
    for host in sorted(p for p in results.iterdir() if p.is_dir() and p.name != "backtest"):
        table_csv, env_json = host / "speedup_table.csv", host / "environment.json"
        if not (table_csv.exists() and env_json.exists()):
            continue
        table = pd.read_csv(table_csv)
        if "gpu" not in table:
            continue  # a CPU-only sweep is a baseline, not a speedup result
        env = json.loads(env_json.read_text())
        table["order"] = table["stage"].map(
            {s: i for i, s in enumerate(STAGE_ORDER)}).fillna(len(STAGE_ORDER))
        table = table.sort_values(["order", "n_assets"])
        # One GPU can have several sweeps (e.g. per estimator); say which this is.
        estimator = "".join(f" `{e}` covariance —" for e in _estimators(host))
        lines = [
            f"**{env.get('gpu', 'unknown GPU')}** —{estimator} `benchmarks/results/{host.name}/`",
            "",
            "| stage | assets | CPU | GPU | speedup |",
            "|---|---|---|---|---|",
        ]
        for _, row in table.iterrows():
            speedup = "—" if pd.isna(row.get("speedup")) else f"{row['speedup']:.2f}×"
            lines.append(f"| {row['stage']} | {int(row['n_assets'])} | {_seconds(row.get('cpu'))} "
                         f"| {_seconds(row['gpu'])} | {speedup} |")
        sections.append("\n".join(lines))
    if not sections:
        return ("_No GPU results committed yet — see [docs/setup-wsl2.md](docs/setup-wsl2.md). "
                "This table is generated from `benchmarks/results/`, so it fills in when they are._")
    return "\n\n".join(sections)


# Budget rules in column order: exact investment, the default band, any cash.
LOT_MIP_METHODS = ("mip_fully_invested", "mip_cash_band", "mip_cash_allowed")


def _pct_of_book(value: float) -> str:
    if pd.isna(value):
        return "—"
    pct = value * 100
    if abs(pct) >= 10:
        return f"{pct:.0f}%"
    return f"{pct:.2g}%" if abs(pct) >= 1e-3 else "0%"


def _mip_cell(row, greedy_error: float) -> str:
    """A MIP's tracking error, and how it compares with greedy's.

    Stated as "x% better/worse" rather than a ratio: at two decimals a 0.2%
    loss prints as 1.00×, indistinguishable from a win.
    """
    if row is None:
        return "—"  # this rule was not run
    if pd.isna(row["tracking_error"]):
        return "no solution"
    change = (row["tracking_error"] / greedy_error - 1.0) * 100
    versus = "same" if change == 0 else f"{abs(change):.2g}% {'worse' if change > 0 else 'better'}"
    cell = f"{_pct_of_book(row['tracking_error'])} ({versus})"
    return cell + " †" if row["status"] == "FeasibleFound" else cell


def lot_rounding_tables(results: Path) -> str:
    """MIP lot rounding against greedy, one table per GPU host that ran it."""
    sections = []
    for host in sorted(p for p in results.iterdir() if p.is_dir()):
        rows_csv, env_json = host / "lot_rounding.csv", host / "environment.json"
        if not (rows_csv.exists() and env_json.exists()):
            continue
        rows = pd.read_csv(rows_csv)
        if not set(LOT_MIP_METHODS) & set(rows["method"]):
            continue  # greedy alone is a baseline, not a comparison
        env = json.loads(env_json.read_text())
        lines = [
            f"**{env.get('gpu', 'unknown GPU')}** — ${rows['portfolio_value'].iloc[0] / 1e6:g}M book "
            f"— `benchmarks/results/{host.name}/`",
            "",
            "| assets | turnover cap | lot | greedy | MIP, fully invested | MIP, cash band | MIP, cash allowed "
            "| cash left: greedy / band / cash allowed | MIP solve: fully invested / band / cash allowed |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        time_limited = False
        rows["cap"] = rows["turnover_budget"].fillna(-1.0)  # uncapped sorts first
        for (n, cap, lot), case in rows.groupby(["n_assets", "cap", "lot_size"]):
            by = {r["method"]: r for _, r in case.iterrows()}
            greedy = by["greedy"]
            mips = [by.get(method) for method in LOT_MIP_METHODS]
            cells = [_mip_cell(r, greedy["tracking_error"]) for r in mips]
            time_limited |= any(cell.endswith("†") for cell in cells)
            cash_left = [_pct_of_book(np.nan if r is None else r["cash"]) for r in mips[1:]]
            solve = [_seconds(np.nan if r is None else r["solve_s"]) for r in mips]
            lines.append(
                f"| {n} | {'none' if cap < 0 else f'{cap:.2f}'} | {lot} | "
                f"{_pct_of_book(greedy['tracking_error'])} | {' | '.join(cells)} | "
                f"{' / '.join([_pct_of_book(greedy['cash']), *cash_left])} | {' / '.join(solve)} |"
            )
        if time_limited:
            lines += ["", "† hit the time limit: the best solution found is shown, not a proven optimum."]
        sections.append("\n".join(lines))
    if not sections:
        return ("_No lot-rounding results committed yet — run `benchmarks.run_lot_rounding` on a GPU "
                "host (see [docs/setup-wsl2.md](docs/setup-wsl2.md)). This table is generated from "
                "`benchmarks/results/`, so it fills in when they are._")
    return "\n\n".join(sections)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail instead of writing if stale")
    args = parser.parse_args()

    blocks = {
        "backtest-summary": backtest_table(RESULTS, BACKTESTS),
        "gpu-speedup": speedup_tables(RESULTS),
        "lot-rounding": lot_rounding_tables(RESULTS),
    }
    unused = [name for name in blocks
              if not any(_markers(name)[0] in doc.read_text() for doc in DOCS)]
    if unused:
        print(f"no doc contains a block for {unused}", file=sys.stderr)
        return 1
    try:
        changed = [doc.name for doc in DOCS if sync_file(doc, blocks, check=args.check)]
    except StaleBlocks as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"updated {', '.join(changed)}" if changed else "generated blocks are up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
