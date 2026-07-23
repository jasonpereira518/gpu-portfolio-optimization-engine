"""CLI: pull a universe of prices and cache it as Parquet.

    python -m data.download_universe --source yfinance --n 500 --start 2014-01-01
    python -m data.download_universe --source synthetic --n 3000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from data.universe import CACHE_DIR, load_universe
from data.validate_data import validate_prices

log = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["yfinance", "synthetic"], default="yfinance")
    parser.add_argument("--n", type=int, default=500, help="number of tickers")
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--seed", type=int, default=0, help="synthetic source only")
    parser.add_argument("--out", type=Path, default=None, help="explicit output Parquet path")
    args = parser.parse_args()

    pd_data = load_universe(args.source, args.n, start=args.start, end=args.end, seed=args.seed)
    log.info("loaded %s prices %s", pd_data.label, pd_data.prices.shape)

    report = validate_prices(pd_data.prices)
    print(report.render())

    out = args.out or CACHE_DIR / f"prices_{pd_data.label}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd_data.prices.to_parquet(out)
    print(f"\nwrote {out}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
