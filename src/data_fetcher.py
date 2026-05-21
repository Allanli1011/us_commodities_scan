"""Yahoo Finance daily-bar fetcher for futures.

- One parquet file per future at `data/prices/<ticker_safe>.parquet`,
  columns: Open / High / Low / Close / Volume (DatetimeIndex named `Date`).
- Incremental update: only request bars newer than the cached last_date.
- Parallel fetch via ThreadPoolExecutor (yfinance also supports batched
  `download(...)`, but per-ticker requests give us clean per-row error
  handling and a tidy progress bar).

Public API:
    update_prices(tickers, full_refresh=False) -> {ticker: status}
    load_prices(ticker)                         -> DataFrame | None
    refresh_all()                               -> universe DataFrame
"""
from __future__ import annotations

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .config import load_config, project_path
from .universe import _yahoo_safe_filename

logger = logging.getLogger(__name__)


# ─────────────── helpers ───────────────

def _price_path(ticker: str) -> Path:
    return project_path("data/prices") / f"{_yahoo_safe_filename(ticker)}.parquet"


def load_prices(ticker: str) -> pd.DataFrame | None:
    p = _price_path(ticker)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    return df


def _save_prices(ticker: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    path = _price_path(ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


# ─────────────── yfinance fetch ───────────────

def _normalise_yf_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """yfinance frame → canonical OHLCV with `Date` index."""
    if raw is None or raw.empty:
        return pd.DataFrame()

    # yfinance ≥ 0.2.40 may hand back MultiIndex columns even for one ticker
    # (level 0 = field, level 1 = ticker). Squash to single-level by keeping
    # the level that contains OHLC strings, then de-duplicating defensively.
    if isinstance(raw.columns, pd.MultiIndex):
        candidate_level = 0
        ohlc = {"Open", "High", "Low", "Close"}
        if not ohlc.intersection(raw.columns.get_level_values(0)):
            candidate_level = 1
        raw = raw.copy()
        raw.columns = raw.columns.get_level_values(candidate_level)
        raw = raw.loc[:, ~raw.columns.duplicated(keep="first")]

    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in raw.columns]
    if not keep or "Close" not in keep:
        return pd.DataFrame()

    df = raw[keep].copy()
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if "Volume" not in df.columns:
        df["Volume"] = 0
    return df


def _fetch_single(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Single-ticker yfinance pull via Ticker.history (cleaner columns than
    yf.download for single symbols) with retry."""
    import yfinance as yf

    cfg = load_config()["prices"]
    auto_adjust = bool(cfg.get("auto_adjust", False))
    prepost = bool(cfg.get("prepost", False))

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            t = yf.Ticker(ticker)
            raw = t.history(
                start=start,
                end=end,
                interval="1d",
                auto_adjust=auto_adjust,
                prepost=prepost,
                actions=False,
            )
            df = _normalise_yf_frame(raw)
            if not df.empty:
                return df
        except Exception as e:
            last_err = e
            wait = 0.4 + attempt * 0.6 + random.random() * 0.4
            logger.debug("[%s] try %d failed: %s; sleep %.1fs", ticker, attempt + 1, e, wait)
            time.sleep(wait)
    if last_err:
        logger.debug("yfinance failed for %s: %s", ticker, last_err)
    return pd.DataFrame()


# ─────────────── orchestration ───────────────

def update_prices(
    tickers: list[str],
    full_refresh: bool = False,
) -> dict[str, str]:
    """Update price caches. Status ∈ {ok, up_to_date, no_data, no_new_data}."""
    cfg = load_config()
    lookback_days = cfg["prices"]["lookback_days"]
    max_workers = int(cfg["prices"].get("max_workers", 6))
    sleep_sec = float(cfg["prices"].get("request_sleep_sec", 0.1))

    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    full_start = (today - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end_str = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # yfinance end is exclusive

    status: dict[str, str] = {}
    plan: list[tuple[str, str]] = []

    for t in tickers:
        existing = load_prices(t)
        if full_refresh or existing is None or existing.empty:
            plan.append((t, full_start))
            continue
        last_date = existing.index.max().normalize()
        if last_date >= today - pd.Timedelta(days=1):
            status[t] = "up_to_date"
            continue
        start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        plan.append((t, start))

    logger.info("Price plan: %d to fetch, %d up-to-date", len(plan), len(status))
    if not plan:
        return status

    def _worker(item: tuple[str, str]) -> tuple[str, pd.DataFrame]:
        t, start = item
        time.sleep(sleep_sec)
        new_df = _fetch_single(t, start, end_str)
        return t, new_df

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_worker, item) for item in plan]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="prices"):
            t, new_df = fut.result()
            if new_df is None or new_df.empty:
                existing = load_prices(t)
                status[t] = "no_new_data" if existing is not None else "no_data"
                continue
            existing = load_prices(t)
            if existing is None or existing.empty:
                _save_prices(t, new_df)
            else:
                combined = pd.concat([existing, new_df])
                combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                _save_prices(t, combined)
            status[t] = "ok"

    n_ok = sum(1 for s in status.values() if s == "ok")
    n_skip = sum(1 for s in status.values() if s in ("up_to_date", "no_new_data"))
    n_fail = sum(1 for s in status.values() if s == "no_data")
    logger.info("Price update done: %d ok, %d skip, %d no_data", n_ok, n_skip, n_fail)
    return status


def refresh_all(force_universe: bool = False) -> pd.DataFrame:
    """End-to-end: build universe → incremental price update."""
    from .universe import build_universe
    universe = build_universe(force_refresh=force_universe)
    logger.info("Universe loaded: %d futures", len(universe))
    update_prices(universe["ticker"].tolist())
    return universe


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--force-universe", action="store_true", help="rebuild universe metadata")
    p.add_argument("--full", action="store_true", help="re-download full history (ignore cache)")
    p.add_argument("--limit", type=int, default=0, help="only first N tickers (debug)")
    args = p.parse_args()

    from .universe import build_universe
    universe = build_universe(force_refresh=args.force_universe)
    tickers = universe["ticker"].tolist()
    if args.limit > 0:
        tickers = tickers[: args.limit]
    update_prices(tickers, full_refresh=args.full)
    print(f"\n✅ Updated {len(tickers)} futures")
