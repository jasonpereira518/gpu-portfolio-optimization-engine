"""Data-quality checks. A benchmark built on bad data is not a benchmark."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class QualityReport:
    n_rows: int
    n_cols: int
    missing_fraction: float
    all_nan_columns: list[str] = field(default_factory=list)
    stale_columns: list[str] = field(default_factory=list)
    calendar_gaps: list[str] = field(default_factory=list)
    suspected_splits: dict[str, list[str]] = field(default_factory=dict)
    nonpositive_prices: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.all_nan_columns or self.nonpositive_prices)

    def render(self) -> str:
        lines = [
            f"rows={self.n_rows} cols={self.n_cols} missing={self.missing_fraction:.4%}",
            f"all-NaN columns:      {len(self.all_nan_columns)} {self.all_nan_columns[:5]}",
            f"stale (constant) cols:{len(self.stale_columns)} {self.stale_columns[:5]}",
            f"calendar gaps:        {len(self.calendar_gaps)} {self.calendar_gaps[:5]}",
            f"suspected splits:     {len(self.suspected_splits)} "
            f"{list(self.suspected_splits)[:5]}",
            f"non-positive prices:  {len(self.nonpositive_prices)} {self.nonpositive_prices[:5]}",
            f"VERDICT: {'PASS' if self.ok else 'FAIL'}",
        ]
        return "\n".join(lines)


def validate_prices(
    prices: pd.DataFrame,
    max_missing: float = 0.05,
    split_threshold: float = 0.35,
    gap_days: int = 4,
) -> QualityReport:
    """Flag the failure modes that actually corrupt a portfolio backtest.

    ``split_threshold``: a single-day absolute return above this is far outside
    what a liquid equity does on its own, so it usually means an unadjusted
    split or a bad print. With ``auto_adjust=True`` these should be rare —
    finding many is a signal the adjustment did not apply.
    """
    n_rows, n_cols = prices.shape
    missing_fraction = float(prices.isna().to_numpy().mean())

    all_nan = [c for c in prices.columns if prices[c].isna().all()]
    clean = prices.drop(columns=all_nan)

    stale = [c for c in clean.columns if clean[c].nunique(dropna=True) <= 1]
    nonpositive = [c for c in clean.columns if (clean[c].dropna() <= 0).any()]

    gaps: list[str] = []
    if isinstance(prices.index, pd.DatetimeIndex) and len(prices.index) > 1:
        deltas = prices.index.to_series().diff().dt.days
        # A weekend is a legitimate 3-day step and a Monday/Friday market
        # holiday a 4-day one, so the default threshold is 4: flagging those
        # would bury the real gaps (halts, delistings) in ~9 rows a year of
        # noise, and a report nobody reads catches nothing.
        for ts, d in deltas[deltas > gap_days].items():
            gaps.append(f"{ts.date()} (+{int(d)}d)")

    returns = clean.pct_change()
    splits: dict[str, list[str]] = {}
    for col in clean.columns:
        hits = returns[col][returns[col].abs() > split_threshold].dropna()
        if len(hits):
            splits[col] = [str(t.date()) for t in hits.index[:5]]

    report = QualityReport(
        n_rows=n_rows,
        n_cols=n_cols,
        missing_fraction=missing_fraction,
        all_nan_columns=all_nan,
        stale_columns=stale,
        calendar_gaps=gaps,
        suspected_splits=splits,
        nonpositive_prices=nonpositive,
    )
    if missing_fraction > max_missing:
        log.warning("missing fraction %.2f%% exceeds tolerance", missing_fraction * 100)
    return report


def clean_prices(
    prices: pd.DataFrame, min_coverage: float = 0.95, ffill_limit: int = 5
) -> pd.DataFrame:
    """Drop thin columns, forward-fill short holes, drop leading all-NaN rows.

    Forward-filling is capped: a long hole means the name was not trading, and
    carrying a stale price across it manufactures a zero-volatility stretch
    that flatters the risk model.
    """
    coverage = prices.notna().mean()
    kept = coverage[coverage >= min_coverage].index
    out = prices[kept].ffill(limit=ffill_limit)
    out = out.dropna(how="all")
    return out.dropna(axis=1, how="any")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a cached price Parquet file.")
    parser.add_argument("parquet", help="path to a prices Parquet file")
    args = parser.parse_args()

    prices = pd.read_parquet(args.parquet)
    report = validate_prices(prices)
    print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
