"""Universe definition and price-data loading.

Two data sources:

* ``yfinance``  — real daily OHLCV, cached to Parquet. Requires network.
* ``synthetic`` — a factor-model price simulator. Requires nothing, is fully
  deterministic given a seed, and scales to universe sizes (3,000+ tickers)
  that free data sources will not serve. Used for the large-N benchmark points
  and for offline CI.

Every benchmark result records which source produced it; they are never mixed
inside one table.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent / "cache"
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

TRADING_DAYS = 252


@dataclass(frozen=True)
class PriceData:
    """Adjusted close prices plus provenance, so results stay traceable."""

    prices: pd.DataFrame  # dates x tickers, adjusted close
    source: str  # "yfinance" | "synthetic"
    universe_size: int

    def __post_init__(self) -> None:
        if self.prices.isna().all(axis=None):
            raise ValueError("price frame is entirely NaN")

    @property
    def label(self) -> str:
        return f"{self.source}-{self.universe_size}"


# --------------------------------------------------------------------------
# Universe lists
# --------------------------------------------------------------------------

def sp500_tickers(timeout: int = 30) -> list[str]:
    """Scrape the current S&P 500 constituent list from Wikipedia.

    Note: this is the *current* membership, so any backtest built on it has
    survivorship bias. That limitation is documented in the README rather than
    silently ignored.
    """
    import requests

    resp = requests.get(
        SP500_WIKI_URL, timeout=timeout, headers={"User-Agent": "gpu-portfolio-engine/0.1"}
    )
    resp.raise_for_status()
    table = pd.read_html(io.StringIO(resp.text))[0]
    tickers = [t.replace(".", "-") for t in table["Symbol"].astype(str)]
    return sorted(set(tickers))


# --------------------------------------------------------------------------
# Real data via yfinance
# --------------------------------------------------------------------------

def download_prices(
    tickers: list[str],
    start: str = "2014-01-01",
    end: str | None = None,
    cache_dir: Path = CACHE_DIR,
    refresh: bool = False,
    batch_size: int = 100,
) -> PriceData:
    """Download adjusted close prices, caching the result as Parquet.

    The cache key is derived from the ticker set and date range, so changing
    the universe does not silently reuse a stale file.
    """
    import hashlib

    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{sorted(tickers)}|{start}|{end}".encode()).hexdigest()[:12]
    path = cache_dir / f"prices_{len(tickers)}_{key}.parquet"

    if path.exists() and not refresh:
        log.info("loading cached prices from %s", path)
        prices = pd.read_parquet(path)
        return PriceData(prices, "yfinance", prices.shape[1])

    import yfinance as yf

    frames = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        log.info("downloading %d tickers (%d/%d)", len(batch), i + len(batch), len(tickers))
        raw = yf.download(
            batch,
            start=start,
            end=end,
            auto_adjust=True,  # split/dividend adjusted closes
            progress=False,
            threads=True,
        )
        if raw.empty:
            continue
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        if not isinstance(raw.columns, pd.MultiIndex):
            close.columns = batch
        frames.append(close)

    if not frames:
        raise RuntimeError("yfinance returned no data for any batch")

    prices = pd.concat(frames, axis=1).sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated()]

    # yfinance intermittently returns an empty series for individual tickers
    # (rate limiting, or a locked local cache DB) even when the batch succeeds.
    # Retry those once, singly, rather than silently shrinking the universe.
    empty = [c for c in prices.columns if prices[c].isna().all()]
    if empty:
        log.warning("retrying %d tickers that returned no data: %s", len(empty), empty[:10])
        for ticker in empty:
            try:
                retry = yf.download(
                    ticker, start=start, end=end, auto_adjust=True,
                    progress=False, threads=False,
                )
                if not retry.empty:
                    series = retry["Close"]
                    prices[ticker] = series.iloc[:, 0] if series.ndim > 1 else series
            except Exception as exc:
                log.warning("retry failed for %s: %s", ticker, exc)

    prices.to_parquet(path)
    log.info("cached %s -> %s", prices.shape, path)
    return PriceData(prices, "yfinance", prices.shape[1])


# --------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------

def synthetic_prices(
    n_assets: int,
    n_days: int = 10 * TRADING_DAYS,
    n_factors: int = 8,
    seed: int = 0,
    start: str = "2014-01-02",
) -> PriceData:
    """Generate prices from a k-factor return model.

    r_t = B f_t + e_t, with heterogeneous idiosyncratic vols and a spread of
    factor loadings. The resulting sample covariance matrix has realistic
    structure (a few dominant eigenvalues, a long noise tail), which matters:
    a covariance matrix built from i.i.d. noise would make the QP trivially
    easy and the benchmark meaningless.
    """
    rng = np.random.default_rng(seed)

    loadings = rng.normal(0.0, 1.0, size=(n_assets, n_factors))
    loadings[:, 0] = np.abs(loadings[:, 0]) * 0.8 + 0.4  # market factor: all positive
    factor_vol = np.sort(rng.uniform(0.004, 0.012, size=n_factors))[::-1]
    idio_vol = rng.uniform(0.008, 0.030, size=n_assets)
    drift = rng.normal(0.0003, 0.0004, size=n_assets)

    factors = rng.standard_normal((n_days, n_factors)) * factor_vol
    idio = rng.standard_normal((n_days, n_assets)) * idio_vol
    returns = drift + factors @ loadings.T + idio

    prices = 100.0 * np.exp(np.cumsum(returns, axis=0))
    dates = pd.bdate_range(start=start, periods=n_days)
    tickers = [f"SYN{i:05d}" for i in range(n_assets)]
    frame = pd.DataFrame(prices, index=dates, columns=tickers)
    return PriceData(frame, "synthetic", n_assets)


def load_universe(
    source: str,
    n_assets: int,
    start: str = "2014-01-01",
    end: str | None = None,
    seed: int = 0,
) -> PriceData:
    """Single entry point used by benchmarks and the backtest driver."""
    if source == "synthetic":
        return synthetic_prices(n_assets, seed=seed, start=start)
    if source == "yfinance":
        tickers = sp500_tickers()[:n_assets]
        return download_prices(tickers, start=start, end=end)
    raise ValueError(f"unknown source {source!r} (expected 'synthetic' or 'yfinance')")
